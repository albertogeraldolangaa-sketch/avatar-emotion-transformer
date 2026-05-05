from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32
DEVICE = torch.device('cpu')


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return target
    return lerp(current, target, 1.0 - math.exp(-dt / tau))


def _hash_text(text: str, dim: int = 128) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=DTYPE)
    text = (text or '').lower()
    for token in re.findall(r"[\wÀ-ÿÇç]+", text):
        h = abs(hash(token))
        vec[h % dim] += 1.0
        vec[(h // 7) % dim] += 0.5
        vec[(h // 31) % dim] += 0.25
    n = torch.linalg.norm(vec).clamp_min(1e-6)
    return vec / n


def _safe_tensor(value: Any, size: int, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    if value is None:
        return torch.zeros(size, device=device, dtype=dtype)
    if isinstance(value, torch.Tensor):
        v = value.detach().to(device=device, dtype=dtype).flatten()
    elif isinstance(value, (list, tuple)):
        v = torch.tensor(list(value), device=device, dtype=dtype).flatten()
    else:
        try:
            v = torch.tensor([float(value)], device=device, dtype=dtype)
        except Exception:
            v = torch.zeros(1, device=device, dtype=dtype)
    if v.numel() < size:
        v = torch.cat([v, torch.zeros(size - v.numel(), device=device, dtype=dtype)], dim=0)
    return v[:size]


@dataclass
class MemoryVector:
    vector: torch.Tensor
    label: str = ''
    source: str = ''
    importance: float = 0.5
    timestamp: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMapItem:
    name: str
    vector: torch.Tensor
    kind: str = 'generic'
    path: str = ''
    meta: Dict[str, Any] = field(default_factory=dict)


class VectorMemoryBank(nn.Module):
    def __init__(self, dim: int = 256, capacity: int = 256, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.register_buffer('bank', torch.zeros(capacity, dim, device=device, dtype=dtype))
        self.register_buffer('importance', torch.zeros(capacity, device=device, dtype=dtype))
        self.register_buffer('age', torch.zeros(capacity, device=device, dtype=dtype))
        self.register_buffer('used', torch.zeros(capacity, device=device, dtype=dtype))
        self._cursor = 0
        self.records: List[MemoryVector] = []

    def write(self, vector: torch.Tensor, importance: float = 0.5, label: str = '', source: str = '', meta: Optional[Dict[str, Any]] = None) -> None:
        vec = vector.detach().to(device=self.device, dtype=self.dtype).flatten()
        if vec.numel() < self.dim:
            vec = torch.cat([vec, torch.zeros(self.dim - vec.numel(), device=self.device, dtype=self.dtype)], dim=0)
        self.bank[self._cursor].copy_(vec[:self.dim])
        self.importance[self._cursor] = float(clamp(importance, 0.0, 1.0))
        self.age[self._cursor] = 0.0
        self.used[self._cursor] = 1.0
        self._cursor = (self._cursor + 1) % self.capacity
        self.records.append(MemoryVector(vector=vec[:self.dim].clone(), label=label, source=source, importance=float(importance), timestamp=time.time(), meta=dict(meta or {})))
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity:]

    def decay(self, dt: float) -> None:
        dt = float(max(1e-6, dt))
        self.age.add_(dt)
        self.importance.mul_(torch.exp(-0.0025 * torch.tensor(dt, device=self.device, dtype=self.dtype)))

    def query(self, vector: torch.Tensor, top_k: int = 5) -> List[MemoryVector]:
        if not self.records:
            return []
        q = vector.detach().to(device=self.device, dtype=self.dtype).flatten()
        if q.numel() < self.dim:
            q = torch.cat([q, torch.zeros(self.dim - q.numel(), device=self.device, dtype=self.dtype)], dim=0)
        q = q[:self.dim]
        bank = self.bank[self.used > 0.5]
        if bank.numel() == 0:
            return []
        scores = F.cosine_similarity(bank, q.unsqueeze(0), dim=-1)
        top = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
        out: List[MemoryVector] = []
        active_records = [r for i, r in enumerate(self.records[-bank.shape[0]:])]
        for idx in top:
            rec = active_records[idx]
            out.append(rec)
        return out

    def summary(self) -> torch.Tensor:
        if self.used.sum() < 1:
            return torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        weights = self.importance * self.used
        norm = weights.sum().clamp_min(1e-6)
        return (self.bank * weights.unsqueeze(-1)).sum(dim=0) / norm


class SystemMapEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, hidden_dim: int = 256, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.out_dim = out_dim
        self.in_dim = 192
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _pack(self, **kwargs: Any) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for key in [
            'mouse', 'window', 'workspace', 'audio', 'emotion', 'focus', 'content', 'history',
            'clipboard', 'filesystem', 'network', 'vision', 'time', 'gesture', 'memory', 'safety',
        ]:
            v = kwargs.get(key)
            if key == 'mouse':
                parts.append(_safe_tensor(v, 16, self.device, self.dtype))
            elif key == 'window':
                parts.append(_safe_tensor(v, 20, self.device, self.dtype))
            elif key == 'workspace':
                parts.append(_safe_tensor(v, 20, self.device, self.dtype))
            elif key == 'audio':
                parts.append(_safe_tensor(v, 12, self.device, self.dtype))
            elif key == 'emotion':
                parts.append(_safe_tensor(v, 8, self.device, self.dtype))
            elif key == 'focus':
                parts.append(_safe_tensor(v, 8, self.device, self.dtype))
            elif key == 'content':
                parts.append(_safe_tensor(v, 28, self.device, self.dtype))
            elif key == 'history':
                parts.append(_safe_tensor(v, 24, self.device, self.dtype))
            elif key == 'clipboard':
                parts.append(_safe_tensor(v, 8, self.device, self.dtype))
            elif key == 'filesystem':
                parts.append(_safe_tensor(v, 16, self.device, self.dtype))
            elif key == 'network':
                parts.append(_safe_tensor(v, 12, self.device, self.dtype))
            elif key == 'vision':
                parts.append(_safe_tensor(v, 12, self.device, self.dtype))
            elif key == 'time':
                parts.append(_safe_tensor(v, 8, self.device, self.dtype))
            elif key == 'gesture':
                parts.append(_safe_tensor(v, 4, self.device, self.dtype))
            elif key == 'memory':
                parts.append(_safe_tensor(v, 10, self.device, self.dtype))
            elif key == 'safety':
                parts.append(_safe_tensor(v, 8, self.device, self.dtype))
        packed = torch.cat(parts, dim=0)
        if packed.numel() < self.in_dim:
            packed = torch.cat([packed, torch.zeros(self.in_dim - packed.numel(), device=self.device, dtype=self.dtype)], dim=0)
        return packed[:self.in_dim]

    def forward(self, **kwargs: Any) -> torch.Tensor:
        return torch.tanh(self.proj(self._pack(**kwargs)))


class LatentIntentRouter(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, num_experts: int = 8, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU())
            for _ in range(num_experts)
        ])
        self.intent_head = nn.Linear((hidden_dim // 4) * num_experts, 64)
        self.action_head = nn.Linear((hidden_dim // 4) * num_experts, 12)
        self.confidence_head = nn.Linear((hidden_dim // 4) * num_experts, 1)
        self.register_buffer('last_gate', torch.zeros(num_experts, device=device, dtype=dtype))

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(latent)
        gate = torch.softmax(self.gate(h), dim=-1)
        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=0)
        mixed = (expert_outs * gate.unsqueeze(-1)).reshape(-1)
        self.last_gate.copy_(gate.detach())
        return {
            'intent_vector': torch.tanh(self.intent_head(mixed)),
            'action_logits': self.action_head(mixed),
            'confidence': torch.sigmoid(self.confidence_head(mixed)).squeeze(-1),
            'gate': gate,
            'expert_mix': mixed,
        }


class WebContextTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def forward(self, *, url: str = '', title: str = '', text: str = '', dom: str = '', trend: str = '') -> torch.Tensor:
        raw = torch.cat([
            _hash_text(url, 32),
            _hash_text(title, 24),
            _hash_text(text, 40),
            _hash_text(dom, 24),
            _hash_text(trend, 8),
        ], dim=0)
        return torch.tanh(self.encoder(raw))


class CreationTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.head = nn.Linear(128, 48)

    def forward(self, description: str, context: Optional[str] = None) -> Dict[str, Any]:
        vec = torch.cat([_hash_text(description, 96), _hash_text(context or '', 32)], dim=0)
        h = self.encoder(vec)
        out = self.head(h)
        code_kind = int(torch.argmax(out[:4]).item())
        return {
            'kind': ['project', 'svg', 'document', 'music'][code_kind],
            'structure': out.detach().cpu().tolist(),
            'summary': description[:240],
        }


class PedagogicalCorrectionTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.scorer = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, student_text: str, exemplar_text: str = '', rubric: str = '') -> Dict[str, Any]:
        a = _hash_text(student_text, 64)
        b = _hash_text(exemplar_text, 64)
        c = _hash_text(rubric, 64)
        h = torch.cat([a + b, c], dim=0)
        logits = self.scorer(h)
        score = float(torch.sigmoid(logits[0]).item())
        return {
            'score': score,
            'needs_review': bool(score < 0.62),
            'feedback_vector': logits.detach().cpu().tolist(),
        }


class SafetyGate(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.head = nn.Sequential(
            nn.Linear(48, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 6),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, float]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] < 48:
            pad = torch.zeros(features.shape[0], 48 - features.shape[-1], device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > 48:
            features = features[..., :48]
        out = torch.sigmoid(self.head(features)).squeeze(0)
        return {
            'allow_clickthrough': float(out[0].item()),
            'allow_topmost': float(out[1].item()),
            'allow_external_action': float(out[2].item()),
            'privacy_mask': float(out[3].item()),
            'full_focus_bias': float(out[4].item()),
            'hold_position': float(out[5].item()),
        }


@dataclass
class AvatarSuperSnapshot:
    latent_confidence: float = 0.0
    workspace_density: float = 0.0
    safety_allow_clickthrough: float = 0.0
    safety_allow_topmost: float = 0.0
    safety_allow_external_action: float = 0.0
    safe_to_click: bool = False
    target_alpha: float = 1.0
    target_scale: float = 1.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    motion_bias_x: float = 0.0
    motion_bias_y: float = 0.0
    intent_focus: float = 0.0
    intent_avoid: float = 0.0
    intent_creation: float = 0.0
    web_relevance: float = 0.0
    pedagogy_score: float = 0.0
    memory_score: float = 0.0
    creation_kind: str = 'project'
    action_path: str = 'idle'


class AvatarSuperSystems(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.encoder = SystemMapEncoder(device=device, dtype=dtype)
        self.router = LatentIntentRouter(device=device, dtype=dtype)
        self.web = WebContextTransformer(device=device, dtype=dtype)
        self.creator = CreationTransformer(device=device, dtype=dtype)
        self.teacher = PedagogicalCorrectionTransformer(device=device, dtype=dtype)
        self.safety = SafetyGate(device=device, dtype=dtype)
        self.memory = VectorMemoryBank(device=device, dtype=dtype)
        self.register_buffer('last_latent', torch.zeros(256, device=device, dtype=dtype))
        self.register_buffer('last_workspace', torch.zeros(256, device=device, dtype=dtype))
        self.register_buffer('last_gate', torch.zeros(8, device=device, dtype=dtype))
        self.snapshot = AvatarSuperSnapshot()

    def _feature_pack(self, perception: Any, emotion: Any, behavior_state: str, plan: Any, motion_data: Dict[str, Any], dt: float) -> Dict[str, Any]:
        return {
            'mouse': [
                float(getattr(perception, 'mouse_x', 0.0)), float(getattr(perception, 'mouse_y', 0.0)),
                float(getattr(perception, 'mouse_dx', 0.0)), float(getattr(perception, 'mouse_dy', 0.0)),
                float(getattr(perception, 'mouse_speed', 0.0)), float(getattr(perception, 'mouse_distance_to_face', 0.0)),
                float(getattr(perception, 'click', False)), float(getattr(perception, 'double_click', False)),
            ],
            'window': [
                float(motion_data.get('space_target_x', 0.0)), float(motion_data.get('space_target_y', 0.0)),
                float(motion_data.get('space_alpha', 1.0)), float(motion_data.get('space_scale', 1.0)),
                float(motion_data.get('space_workspace_density', 0.0)), float(motion_data.get('space_occupancy_pressure', 0.0)),
                float(motion_data.get('space_drag_active', False)), float(motion_data.get('space_fullscreen_pressure', 0.0)),
            ],
            'workspace': [
                float(getattr(perception, 'focus', True)), float(getattr(perception, 'silence_s', 0.0)), float(getattr(perception, 'audio_rms', 0.0)),
                float(getattr(perception, 'audio_peak', 0.0)), float(getattr(perception, 'turn_index', 0)), float(getattr(perception, 'window_active', True)),
            ],
            'audio': [float(getattr(perception, 'audio_rms', 0.0)), float(getattr(perception, 'audio_peak', 0.0)), float(getattr(perception, 'audio_attack', 0.0)), float(getattr(perception, 'audio_release', 0.0))],
            'emotion': [float(getattr(emotion, 'intensity', 0.0)), float(getattr(emotion, 'energy', 0.0)), float(getattr(emotion, 'arousal', 0.0)), float(getattr(emotion, 'curiosity', 0.0))],
            'focus': [float(getattr(emotion, 'attention', 0.0)), float(getattr(emotion, 'confidence', 0.0)), float(getattr(emotion, 'tension', 0.0)), float(getattr(emotion, 'fatigue', 0.0))],
            'content': [float(len(getattr(perception, 'user_text', '') or '') / 256.0), float(getattr(perception, 'user_text_len', 0)), float(bool(plan)), float(getattr(plan, 'intensity', 0.0) if plan else 0.0)],
            'history': [float(getattr(perception, 'turn_index', 0)), float(motion_data.get('space_window_count', 0)), float(motion_data.get('space_drag_active', False)), float(motion_data.get('space_fullscreen_pressure', 0.0))],
            'clipboard': [float(getattr(perception, 'input_active', False)), float(getattr(perception, 'speaking', False))],
            'filesystem': [float(motion_data.get('space_occupied_count', 0)), float(motion_data.get('space_heat_mean', 0.0)), float(motion_data.get('space_heat_peak', 0.0))],
            'network': [float(getattr(perception, 'window_active', True)), float(getattr(perception, 'focus', True)), float(motion_data.get('latent_web_relevance', 0.0))],
            'vision': [float(motion_data.get('space_target_z', 0.0)), float(motion_data.get('space_cursor_locked', False)), float(motion_data.get('space_topmost_hint', 0.0))],
            'time': [float(dt), float(getattr(perception, 'silence_s', 0.0)), float(getattr(perception, 'turn_index', 0))],
            'gesture': [float(getattr(plan, 'gesture_strength', 0.0) if plan else 0.0), float(motion_data.get('space_motion_energy', 0.0))],
            'memory': [float(self.memory.summary().abs().mean().item() if self.memory.used.sum() > 0 else 0.0), float(len(self.memory.records)), float(motion_data.get('latent_memory_score', 0.0))],
            'safety': [float(motion_data.get('space_allow_clickthrough', 1.0)), float(motion_data.get('space_allow_topmost', 1.0)), float(motion_data.get('space_privacy_mask', 0.0))],
        }

    def step(self, perception: Any, emotion: Any, behavior_state: str, plan: Any, motion_data: Optional[Dict[str, Any]] = None, dt: float = 0.016) -> Dict[str, Any]:
        motion_data = dict(motion_data or {})
        features = self._feature_pack(perception, emotion, behavior_state, plan, motion_data, dt)
        latent = self.encoder(**features)
        self.last_latent.copy_(latent.detach())
        workspace = self.web(
            url=str(motion_data.get('latent_url', '')),
            title=str(motion_data.get('latent_title', '')),
            text=str(getattr(perception, 'user_text', '') or motion_data.get('latent_text', '')),
            dom=str(motion_data.get('latent_dom', '')),
            trend=str(motion_data.get('latent_trend', '')),
        )
        self.last_workspace.copy_(torch.cat([workspace, torch.zeros(256 - workspace.numel(), device=self.device, dtype=self.dtype)], dim=0)[:256])
        routed = self.router(latent)
        gate = self.safety(torch.cat([latent[:32], workspace[:16], routed['intent_vector'][:32]], dim=0))
        self.last_gate.copy_(routed['gate'].detach())

        intent_vec = routed['intent_vector']
        action_logits = routed['action_logits']
        action_idx = int(torch.argmax(action_logits).item())
        actions = ['idle', 'approach', 'avoid', 'focus', 'front', 'hide', 'creation', 'web', 'pedagogy', 'observe', 'move', 'hold']
        action_path = actions[action_idx % len(actions)]

        latency = float(routed['confidence'].item())
        memory_score = float(self.memory.summary().abs().mean().item() if self.memory.used.sum() > 0 else 0.0)
        creation = self.creator(str(motion_data.get('latent_creation_prompt', motion_data.get('latent_text', ''))), context=str(motion_data.get('latent_creation_context', '')))
        pedagogy = self.teacher(str(getattr(perception, 'user_text', '') or ''), str(motion_data.get('latent_exemplar', '')), str(motion_data.get('latent_rubric', '')))

        self.memory.write(latent, importance=max(0.2, float(gate['full_focus_bias'])), label=action_path, source='latent', meta={'gate': gate})
        self.memory.decay(dt)

        workspace_density = float(clamp(
            0.35 * float(motion_data.get('space_workspace_density', 0.0)) +
            0.25 * float(motion_data.get('space_occupancy_pressure', 0.0)) +
            0.18 * float(getattr(perception, 'silence_s', 0.0)) / 18.0 +
            0.22 * float(getattr(perception, 'mouse_speed', 0.0)) / 1200.0,
            0.0, 1.0))
        safe_to_click = float(gate['privacy_mask']) < 0.55 and float(gate['allow_clickthrough']) > 0.55
        alpha = clamp(1.0 - 0.45 * workspace_density + 0.12 * float(gate['allow_topmost']), 0.08, 1.0)
        scale = clamp(0.82 + 0.42 * (1.0 - workspace_density) + 0.10 * float(latency), 0.65, 1.55)
        target_x = float(motion_data.get('space_target_x', 0.0)) + (float(intent_vec[0].item()) * 18.0)
        target_y = float(motion_data.get('space_target_y', 0.0)) + (float(intent_vec[1].item()) * 18.0)
        target_z = clamp(float(motion_data.get('space_target_z', 0.5)) + 0.14 * float(gate['allow_topmost']), 0.0, 1.0)

        self.snapshot = AvatarSuperSnapshot(
            latent_confidence=latency,
            workspace_density=workspace_density,
            safety_allow_clickthrough=float(gate['allow_clickthrough']),
            safety_allow_topmost=float(gate['allow_topmost']),
            safety_allow_external_action=float(gate['allow_external_action']),
            safe_to_click=safe_to_click,
            target_alpha=alpha,
            target_scale=scale,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            motion_bias_x=float(intent_vec[2].item()) * 0.35,
            motion_bias_y=float(intent_vec[3].item()) * 0.35,
            intent_focus=float(intent_vec[4].item()),
            intent_avoid=float(intent_vec[5].item()),
            intent_creation=float(intent_vec[6].item()),
            web_relevance=float(workspace.abs().mean().item()),
            pedagogy_score=float(pedagogy['score']),
            memory_score=memory_score,
            creation_kind=creation['kind'],
            action_path=action_path,
        )

        motion_data.update({
            'super_latent_confidence': self.snapshot.latent_confidence,
            'super_workspace_density': self.snapshot.workspace_density,
            'super_safe_to_click': self.snapshot.safe_to_click,
            'super_allow_clickthrough': self.snapshot.safety_allow_clickthrough,
            'super_allow_topmost': self.snapshot.safety_allow_topmost,
            'super_allow_external_action': self.snapshot.safety_allow_external_action,
            'super_target_alpha': self.snapshot.target_alpha,
            'super_target_scale': self.snapshot.target_scale,
            'super_target_x': self.snapshot.target_x,
            'super_target_y': self.snapshot.target_y,
            'super_target_z': self.snapshot.target_z,
            'super_motion_bias_x': self.snapshot.motion_bias_x,
            'super_motion_bias_y': self.snapshot.motion_bias_y,
            'super_intent_focus': self.snapshot.intent_focus,
            'super_intent_avoid': self.snapshot.intent_avoid,
            'super_intent_creation': self.snapshot.intent_creation,
            'super_web_relevance': self.snapshot.web_relevance,
            'super_pedagogy_score': self.snapshot.pedagogy_score,
            'super_memory_score': self.snapshot.memory_score,
            'super_creation_kind': self.snapshot.creation_kind,
            'super_action_path': self.snapshot.action_path,
        })
        return motion_data


def build_super_systems(device: torch.device = DEVICE, dtype: torch.dtype = DTYPE) -> AvatarSuperSystems:
    return AvatarSuperSystems(device=device, dtype=dtype)


__all__ = [
    'AvatarSuperSystems', 'AvatarSuperSnapshot', 'SystemMapEncoder', 'LatentIntentRouter',
    'WebContextTransformer', 'CreationTransformer', 'PedagogicalCorrectionTransformer',
    'SafetyGate', 'VectorMemoryBank', 'SystemMapItem', 'MemoryVector', 'build_super_systems',
]
