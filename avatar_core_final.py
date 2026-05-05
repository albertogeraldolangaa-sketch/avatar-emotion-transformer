 
import json
import math
import os
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from animation_fluidity_transformers import (
        AnimationFluidityTransformer,
        AnimationFluidityInput,
    )
except Exception:
    AnimationFluidityTransformer = None
    AnimationFluidityInput = None

try:
    from occupancy_visibility_transformer import (
        OccupancyVisibilityController,
        SpaceOccupancyInput,
        SpaceOccupancyState,
    )
except Exception:
    OccupancyVisibilityController = None
    SpaceOccupancyInput = None
    SpaceOccupancyState = None

DTYPE = torch.float32
DEVICE = torch.device("cpu")


# =========================
# Utilities
# =========================

def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def clamp11(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


def smooth_update(x: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
    a = torch.as_tensor(alpha, device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
    return x + (target - x) * a


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    return torch.norm(x, p=2, dim=dim, keepdim=keepdim).clamp_min(eps)


def to_tensor(x, *, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class AvatarMode(str, Enum):
    idle = "idle"
    listening = "listening"
    thinking = "thinking"
    speaking = "speaking"
    reacting = "reacting"
    curious = "curious"
    surprised = "surprised"
    shy = "shy"
    observe = "observe"
    retreat = "retreat"
    peek = "peek"
    lean_in = "lean_in"
    rest = "rest"


# =========================
# Dataclasses
# =========================


@dataclass
class PerceptionInput:
    text_tokens: torch.Tensor
    conversation_embedding: torch.Tensor
    emotion_state: torch.Tensor
    memory_short: torch.Tensor
    memory_consolidated: torch.Tensor
    audio_features: torch.Tensor
    mouse_features: torch.Tensor
    click_focus_features: torch.Tensor
    silence_features: torch.Tensor
    repetition_features: torch.Tensor
    contextual_salience: torch.Tensor
    body_state: torch.Tensor
    face_state: torch.Tensor


@dataclass
class PolicyOutput:
    text: torch.Tensor
    emotion: torch.Tensor
    intensity: torch.Tensor
    intention: torch.Tensor
    attention_target: torch.Tensor
    gaze_target: torch.Tensor
    gesture: torch.Tensor
    gesture_strength: torch.Tensor
    body_posture: torch.Tensor
    motion_speed: torch.Tensor
    pause_ms: torch.Tensor
    silence_response: torch.Tensor
    confidence: torch.Tensor
    curiosity: torch.Tensor
    tension: torch.Tensor
    memory_update: torch.Tensor
    state_transition: torch.Tensor


@dataclass
class AvatarBehaviorFrame:
    mode: AvatarMode
    policy: PolicyOutput
    motion: torch.Tensor
    mouth: torch.Tensor
    gaze: torch.Tensor
    attention_weights: torch.Tensor
    memory_write: torch.Tensor


@dataclass
class AvatarPersistentState:
    mode: str = AvatarMode.idle.value
    step: int = 0
    hidden_emotion: List[float] = field(default_factory=lambda: [0.05, 0.20, 0.10, 0.12, 0.10, 0.55])
    body: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.10, 0.10, 0.0, 0.0])
    face: List[float] = field(default_factory=lambda: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    last_attention_target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    last_gaze_target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


# =========================
# State Manager
# =========================


class StateManager:
    def __init__(self, path: str | Path = "avatar_state.json", device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        self.path = Path(path)
        self.device = device
        self.dtype = dtype

    def load(self) -> AvatarPersistentState:
        if not self.path.exists():
            return AvatarPersistentState()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return AvatarPersistentState(**data)

    def save(self, state: AvatarPersistentState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def tensorize_state(state: AvatarPersistentState, device=DEVICE, dtype=DTYPE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emotion = torch.tensor(state.hidden_emotion, device=device, dtype=dtype)
        body = torch.tensor(state.body, device=device, dtype=dtype)
        face = torch.tensor(state.face, device=device, dtype=dtype)
        return emotion, body, face

    @staticmethod
    def detensorize_state(state: AvatarPersistentState, emotion: torch.Tensor, body: torch.Tensor, face: torch.Tensor, mode: AvatarMode, step: int, attention_target: torch.Tensor, gaze_target: torch.Tensor) -> AvatarPersistentState:
        state.hidden_emotion = emotion.detach().cpu().tolist()
        state.body = body.detach().cpu().tolist()
        state.face = face.detach().cpu().tolist()
        state.mode = mode.value
        state.step = int(step)
        state.last_attention_target = attention_target.detach().cpu().tolist()
        state.last_gaze_target = gaze_target.detach().cpu().tolist()
        return state


# =========================
# Perception Engine
# =========================


class PerceptionEngine(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.text_proj = nn.Linear(32, 32)
        self.context_proj = nn.Linear(24, 24)
        self.audio_proj = nn.Linear(12, 12)
        self.mouse_proj = nn.Linear(8, 8)
        self.env_proj = nn.Linear(12, 12)
        self.body_proj = nn.Linear(12, 12)
        self.face_proj = nn.Linear(12, 12)
        self.fusion = nn.Linear(32 + 24 + 12 + 8 + 12 + 12 + 12 + 8, 64)

    def forward(self, inp: PerceptionInput) -> torch.Tensor:
        text = self.text_proj(inp.text_tokens)
        context = self.context_proj(inp.conversation_embedding)
        audio = self.audio_proj(inp.audio_features)
        mouse = self.mouse_proj(inp.mouse_features)
        env = self.env_proj(torch.cat([inp.click_focus_features, inp.silence_features, inp.repetition_features, inp.contextual_salience], dim=-1))
        body = self.body_proj(inp.body_state)
        face = self.face_proj(inp.face_state)
        emotion = inp.emotion_state
        x = torch.cat([text, context, audio, mouse, env, body, face, emotion], dim=-1)
        return torch.tanh(self.fusion(x))


# =========================
# Memory Engine
# =========================


class MemoryEngine(nn.Module):
    def __init__(self, d_model: int = 64, short_size: int = 8, consolidated_size: int = 32, emotional_size: int = 16, preference_size: int = 16, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.short_size = short_size
        self.consolidated_size = consolidated_size
        self.emotional_size = emotional_size
        self.preference_size = preference_size
        self.d_model = d_model

        self.write_gate = nn.Linear(d_model, 1)
        self.importance_head = nn.Linear(d_model, 1)
        self.recall_query = nn.Linear(d_model, d_model)
        self.consolidation_proj = nn.Linear(d_model, d_model)
        self.register_buffer("short_term", torch.zeros(short_size, d_model, device=device, dtype=dtype))
        self.register_buffer("consolidated", torch.zeros(consolidated_size, d_model, device=device, dtype=dtype))
        self.register_buffer("emotional", torch.zeros(emotional_size, d_model, device=device, dtype=dtype))
        self.register_buffer("preferences", torch.zeros(preference_size, d_model, device=device, dtype=dtype))
        self.register_buffer("short_age", torch.zeros(short_size, device=device, dtype=dtype))
        self.register_buffer("cons_age", torch.zeros(consolidated_size, device=device, dtype=dtype))
        self.register_buffer("emo_age", torch.zeros(emotional_size, device=device, dtype=dtype))
        self.register_buffer("pref_age", torch.zeros(preference_size, device=device, dtype=dtype))
        self._short_cursor = 0
        self._cons_cursor = 0
        self._emo_cursor = 0
        self._pref_cursor = 0

    def decay_memory(self, dt: torch.Tensor) -> None:
        decay = torch.exp(-0.04 * dt)
        self.short_term.mul_(decay)
        self.consolidated.mul_(torch.exp(-0.005 * dt))
        self.emotional.mul_(torch.exp(-0.02 * dt))
        self.preferences.mul_(torch.exp(-0.003 * dt))
        self.short_age.add_(dt)
        self.cons_age.add_(dt)
        self.emo_age.add_(dt)
        self.pref_age.add_(dt)

    def write(self, x: torch.Tensor, emotional_weight: torch.Tensor, preference_weight: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        self.decay_memory(dt)
        write_score = torch.sigmoid(self.write_gate(x)).squeeze(-1)
        importance = torch.sigmoid(self.importance_head(x)).squeeze(-1)
        strength = (0.35 * write_score + 0.45 * importance + 0.20 * emotional_weight).clamp(0.0, 1.0)

        self.short_term[self._short_cursor].copy_(x)
        self.short_age[self._short_cursor] = 0.0
        self._short_cursor = (self._short_cursor + 1) % self.short_size

        if strength > 0.55:
            target = torch.tanh(self.consolidation_proj(x))
            self.consolidated[self._cons_cursor].copy_(target)
            self.cons_age[self._cons_cursor] = 0.0
            self._cons_cursor = (self._cons_cursor + 1) % self.consolidated_size

        if emotional_weight > 0.50:
            self.emotional[self._emo_cursor].copy_(x)
            self.emo_age[self._emo_cursor] = 0.0
            self._emo_cursor = (self._emo_cursor + 1) % self.emotional_size

        if preference_weight > 0.50:
            self.preferences[self._pref_cursor].copy_(x)
            self.pref_age[self._pref_cursor] = 0.0
            self._pref_cursor = (self._pref_cursor + 1) % self.preference_size

        return strength

    def retrieve(self, query: torch.Tensor, k: int = 4) -> torch.Tensor:
        q = self.recall_query(query)
        banks = torch.cat([self.short_term, self.consolidated, self.emotional, self.preferences], dim=0)
        if banks.numel() == 0:
            return torch.zeros_like(q)
        scores = torch.matmul(banks, q)
        topk = torch.topk(scores, k=min(k, scores.shape[0]), largest=True).indices
        selected = banks[topk]
        weights = torch.softmax(scores[topk], dim=0).unsqueeze(-1)
        return torch.sum(selected * weights, dim=0)

    def summary(self) -> torch.Tensor:
        return torch.cat(
            [
                self.short_term.mean(dim=0),
                self.consolidated.mean(dim=0),
                self.emotional.mean(dim=0),
                self.preferences.mean(dim=0),
            ],
            dim=-1,
        )


# =========================
# Emotion Engine
# =========================


class EmotionEngine(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer("state", torch.tensor([0.05, 0.20, 0.02, 0.10, 0.05, 0.55], device=device, dtype=dtype))
        self.inertia = nn.Parameter(torch.tensor(0.82, device=device, dtype=dtype), requires_grad=False)
        self.recovery = nn.Parameter(torch.tensor(0.04, device=device, dtype=dtype), requires_grad=False)
        self.modulator = nn.Linear(64, 6)

    def forward(self, perception: torch.Tensor, memory_summary: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([perception, memory_summary[:32], context[:8]], dim=-1)
        drive = torch.tanh(self.modulator(x))
        target = torch.stack(
            [
                0.60 * self.state[0] + 0.25 * drive[0] + 0.10 * context.mean(),
                0.55 * self.state[1] + 0.28 * drive[1] + 0.12 * safe_norm(perception) / math.sqrt(perception.numel()),
                0.50 * self.state[2] + 0.25 * drive[2] + 0.10 * memory_summary.abs().mean(),
                0.58 * self.state[3] + 0.22 * drive[3] + 0.10 * context.std(unbiased=False),
                0.62 * self.state[4] + 0.24 * drive[4] + 0.08 * perception.abs().mean(),
                0.66 * self.state[5] + 0.18 * drive[5] - 0.10 * self.state[3],
            ]
        )
        self.state.copy_(smooth_update(self.state, torch.tanh(target), 1.0 - self.inertia))
        self.state[:5] = clamp11(self.state[:5])
        self.state[5] = clamp01(self.state[5].unsqueeze(0)).squeeze(0)
        self.state.mul_(1.0 - self.recovery * 0.02)
        return self.state


# =========================
# Attention Engine
# =========================


class AttentionEngine(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.head = nn.Linear(64 + 6 + 32, 10)
        self.register_buffer("last_weights", torch.zeros(10, device=device, dtype=dtype))

    def forward(self, perception: torch.Tensor, emotion: torch.Tensor, memory_summary: torch.Tensor, repetition: torch.Tensor, fatigue: torch.Tensor, novelty: torch.Tensor, priority: torch.Tensor) -> torch.Tensor:
        x = torch.cat([perception, emotion, memory_summary[:32]], dim=-1)
        logits = self.head(x)
        logits = logits + torch.tensor([
            novelty,
            priority,
            -fatigue,
            repetition,
            emotion[1],
            emotion[0],
            emotion[3],
            emotion[4],
            emotion[2],
            perception.mean(),
        ], device=self.device, dtype=self.dtype)
        weights = torch.softmax(logits, dim=-1)
        self.last_weights.copy_(weights)
        return weights


# =========================
# Transformer Policy
# =========================


@dataclass
class TransformerScaleConfig:
    d_model: int = 3072
    nhead: int = 24
    num_layers: int = 30
    dim_feedforward: int = 12288
    dropout: float = 0.05


@dataclass
class SpatialDynamicsConfig:
    body_scale_min: float = 0.68
    body_scale_max: float = 1.55
    alpha_min: float = 0.00
    alpha_max: float = 1.00
    placement_range: float = 0.48
    occupancy_bias: float = 0.38
    visibility_softness: float = 0.42
    front_idle_threshold: float = 5.5
    path_smoothing: float = 0.20
    mouse_escape_radius: float = 240.0



class TransformerPolicy(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        nhead: int = 16,
        num_layers: int = 24,
        dim_feedforward: int = 8192,
        dropout: float = 0.05,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.input_proj = nn.Linear(64 + 64 + 6 + 10 + 12, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_type = nn.Embedding(8, d_model)
        self.pos_embed = nn.Parameter(torch.randn(8, d_model, device=device, dtype=dtype) * 0.02)

        self.text_head = nn.Linear(d_model, 32)
        self.emotion_head = nn.Linear(d_model, 6)
        self.intensity_head = nn.Linear(d_model, 1)
        self.intention_head = nn.Linear(d_model, 8)
        self.attention_target_head = nn.Linear(d_model, 4)
        self.gaze_target_head = nn.Linear(d_model, 3)
        self.gesture_head = nn.Linear(d_model, 10)
        self.gesture_strength_head = nn.Linear(d_model, 1)
        self.body_posture_head = nn.Linear(d_model, 6)
        self.motion_speed_head = nn.Linear(d_model, 1)
        self.pause_ms_head = nn.Linear(d_model, 1)
        self.silence_response_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)
        self.curiosity_head = nn.Linear(d_model, 1)
        self.tension_head = nn.Linear(d_model, 1)
        self.memory_update_head = nn.Linear(d_model, 16)
        self.state_transition_head = nn.Linear(d_model, len(AvatarMode))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(tokens)
        t = self.token_type(torch.arange(h.shape[1], device=h.device))
        h = h + self.pos_embed[: h.shape[1]] + t
        h = self.transformer(h)
        return h[:, -1, :]

    def decode(self, h: torch.Tensor, vocab_size: int = 32) -> PolicyOutput:
        text = self.text_head(h)
        emotion = torch.tanh(self.emotion_head(h))
        intensity = torch.sigmoid(self.intensity_head(h)).squeeze(-1)
        intention = torch.softmax(self.intention_head(h), dim=-1)
        attention_target = torch.softmax(self.attention_target_head(h), dim=-1)
        gaze_target = torch.tanh(self.gaze_target_head(h))
        gesture = torch.softmax(self.gesture_head(h), dim=-1)
        gesture_strength = torch.sigmoid(self.gesture_strength_head(h)).squeeze(-1)
        body_posture = torch.tanh(self.body_posture_head(h))
        motion_speed = torch.sigmoid(self.motion_speed_head(h)).squeeze(-1)
        pause_ms = torch.relu(self.pause_ms_head(h)).squeeze(-1) * 900.0 + 80.0
        silence_response = torch.sigmoid(self.silence_response_head(h)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        curiosity = torch.sigmoid(self.curiosity_head(h)).squeeze(-1)
        tension = torch.sigmoid(self.tension_head(h)).squeeze(-1)
        memory_update = torch.tanh(self.memory_update_head(h))
        state_transition = torch.softmax(self.state_transition_head(h), dim=-1)
        return PolicyOutput(
            text=text,
            emotion=emotion,
            intensity=intensity,
            intention=intention,
            attention_target=attention_target,
            gaze_target=gaze_target,
            gesture=gesture,
            gesture_strength=gesture_strength,
            body_posture=body_posture,
            motion_speed=motion_speed,
            pause_ms=pause_ms,
            silence_response=silence_response,
            confidence=confidence,
            curiosity=curiosity,
            tension=tension,
            memory_update=memory_update,
            state_transition=state_transition,
        )


# =========================
# Behavior Engine
# =========================


class BehaviorEngine(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mode_to_index = {m: i for i, m in enumerate(AvatarMode)}
        self.register_buffer("mode_state", torch.zeros(len(AvatarMode), device=device, dtype=dtype))
        self.transition = nn.Linear(len(AvatarMode), len(AvatarMode), bias=False)
        self.action_gate = nn.Linear(16, 8)

    def choose_mode(self, policy: PolicyOutput, emotion: torch.Tensor, attention: torch.Tensor, silence: torch.Tensor, repetition_penalty: torch.Tensor) -> AvatarMode:
        state_probs = policy.state_transition
        mix = torch.stack(
            [
                state_probs[self.mode_to_index[AvatarMode.idle]],
                state_probs[self.mode_to_index[AvatarMode.listening]] + 0.20 * attention[0],
                state_probs[self.mode_to_index[AvatarMode.thinking]] + 0.20 * policy.curiosity,
                state_probs[self.mode_to_index[AvatarMode.speaking]] + 0.25 * policy.intensity,
                state_probs[self.mode_to_index[AvatarMode.reacting]] + 0.20 * emotion[3],
                state_probs[self.mode_to_index[AvatarMode.curious]] + 0.18 * policy.curiosity,
                state_probs[self.mode_to_index[AvatarMode.surprised]] + 0.14 * emotion[1],
                state_probs[self.mode_to_index[AvatarMode.shy]] + 0.14 * (1.0 - policy.confidence),
                state_probs[self.mode_to_index[AvatarMode.observe]] + 0.12 * attention[1],
                state_probs[self.mode_to_index[AvatarMode.retreat]] + 0.12 * repetition_penalty,
                state_probs[self.mode_to_index[AvatarMode.peek]] + 0.08 * attention[2],
                state_probs[self.mode_to_index[AvatarMode.lean_in]] + 0.12 * policy.curiosity,
                state_probs[self.mode_to_index[AvatarMode.rest]] + 0.14 * silence,
            ],
            dim=0,
        )
        idx = torch.argmax(mix).item()
        return list(AvatarMode)[idx]

    def continuous_transition(self, current: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
        return smooth_update(current, target, alpha)

    def build_body_face(self, policy: PolicyOutput, emotion: torch.Tensor, attention: torch.Tensor, current_body: torch.Tensor, current_face: torch.Tensor, dt: torch.Tensor, repetition_penalty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        body_target = torch.stack(
            [
                0.60 * policy.gesture_strength + 0.15 * policy.motion_speed,
                0.55 * policy.intensity + 0.18 * emotion[1],
                0.50 * policy.body_posture[0] + 0.15 * attention[0],
                0.50 * policy.body_posture[1] + 0.10 * policy.curiosity,
                0.50 * policy.body_posture[2] - 0.10 * repetition_penalty,
                0.50 * policy.body_posture[3] + 0.10 * policy.confidence,
            ]
        )
        face_target = torch.stack(
            [
                0.62 * emotion[0] + 0.10 * policy.confidence,
                0.62 * emotion[1] + 0.10 * policy.intensity,
                0.56 * emotion[2] + 0.14 * policy.curiosity,
                0.58 * emotion[3] + 0.10 * attention[1],
                0.55 * emotion[4] + 0.08 * repetition_penalty,
                0.55 * emotion[5] + 0.10 * policy.silence_response,
            ]
        )
        body = smooth_update(current_body, body_target, 1.0 - torch.exp(-3.0 * dt))
        face = smooth_update(current_face, face_target, 1.0 - torch.exp(-4.0 * dt))
        return clamp11(body), clamp01(face)


# =========================
# Spatial / Visibility Transformers
# =========================


class ScaleTransformer(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 32, config: Optional[SpatialDynamicsConfig] = None, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.config = config or SpatialDynamicsConfig()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("smooth_scale", torch.tensor(1.0, device=device, dtype=dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.to(device=self.device, dtype=self.dtype)
        raw = torch.sigmoid(self.net(x).squeeze(-1))
        goal = self.config.body_scale_min + raw * (self.config.body_scale_max - self.config.body_scale_min)
        if x.numel() >= 4:
            attention_hint = torch.clamp(0.55 + 0.12 * x[..., 0] + 0.10 * x[..., 1] - 0.08 * x[..., 2], 0.78, 1.18)
            goal = goal * attention_hint
        self.smooth_scale.copy_(smooth_update(self.smooth_scale, goal, 1.0 - torch.exp(torch.tensor(-3.4, device=self.device, dtype=self.dtype))))
        return self.smooth_scale


class PlacementTransformer(nn.Module):
    def __init__(self, in_dim: int = 10, hidden_dim: int = 32, config: Optional[SpatialDynamicsConfig] = None, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.config = config or SpatialDynamicsConfig()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        self.register_buffer("last_pos_x", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("last_pos_y", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("last_alpha", torch.tensor(1.0, device=device, dtype=dtype))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = features.to(device=self.device, dtype=self.dtype)
        raw = self.net(x)
        pos = torch.tanh(raw[:2]) * self.config.placement_range
        front = torch.sigmoid(raw[2])
        alpha = torch.sigmoid(raw[3])
        drift = torch.tanh(raw[4]) * 0.12
        pos_x = pos[0] + drift * front
        pos_y = pos[1] + drift * (1.0 - front)
        alpha = self.config.alpha_min + alpha * (self.config.alpha_max - self.config.alpha_min)
        self.last_pos_x.copy_(smooth_update(self.last_pos_x, pos_x, 0.08))
        self.last_pos_y.copy_(smooth_update(self.last_pos_y, pos_y, 0.08))
        self.last_alpha.copy_(smooth_update(self.last_alpha, alpha, 0.08))
        return self.last_pos_x, self.last_pos_y, self.last_alpha


class VisibilityTransformer(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 24, config: Optional[SpatialDynamicsConfig] = None, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.config = config or SpatialDynamicsConfig()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("alpha_state", torch.tensor(1.0, device=device, dtype=dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.to(device=self.device, dtype=self.dtype)
        raw = torch.sigmoid(self.net(x).squeeze(-1))
        target = self.config.alpha_min + raw * (self.config.alpha_max - self.config.alpha_min)
        if x.numel() >= 3:
            target = torch.clamp(target + 0.06 * x[..., 2] - 0.05 * torch.abs(x[..., 0]), self.config.alpha_min, self.config.alpha_max)
        self.alpha_state.copy_(smooth_update(self.alpha_state, target, 0.12))
        return self.alpha_state


# =========================
# Motion Engine
# =========================


class MotionEngine(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.config = SpatialDynamicsConfig()
        self.scale_transformer = ScaleTransformer(device=device, dtype=dtype, config=self.config)
        self.placement_transformer = PlacementTransformer(device=device, dtype=dtype, config=self.config)
        self.visibility_transformer = VisibilityTransformer(device=device, dtype=dtype, config=self.config)
        self.animation_fluidity = AnimationFluidityTransformer(device=device, dtype=dtype) if AnimationFluidityTransformer is not None else None
        self.register_buffer("breathing_phase", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("micro_phase", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("last_scale", torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer("last_alpha", torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer("last_place_x", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("last_place_y", torch.tensor(0.0, device=device, dtype=dtype))

    def _pad_features(self, features: Optional[torch.Tensor], size: int) -> torch.Tensor:
        if features is None:
            return torch.zeros(size, device=self.device, dtype=self.dtype)
        vec = features.flatten().to(device=self.device, dtype=self.dtype)
        if vec.numel() >= size:
            return vec[:size]
        return torch.cat([vec, torch.zeros(size - vec.numel(), device=self.device, dtype=self.dtype)], dim=0)

    def forward(self, body: torch.Tensor, emotion: torch.Tensor, attention: torch.Tensor, dt: torch.Tensor, mode: AvatarMode, spatial_features: Optional[torch.Tensor] = None, workspace_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.breathing_phase.add_(dt * (0.8 + 0.4 * emotion[1]))
        self.micro_phase.add_(dt * (1.5 + 1.2 * attention.mean()))

        breathing = 0.024 * torch.sin(self.breathing_phase)
        micro = 0.010 * torch.sin(3.0 * self.micro_phase)
        idle_drift = 0.018 * torch.tanh(torch.tensor([emotion[0], emotion[1], emotion[3], emotion[4], emotion[5], attention[0]], device=self.device, dtype=self.dtype))

        mode_bias = {
            AvatarMode.idle: 0.00,
            AvatarMode.listening: 0.02,
            AvatarMode.thinking: -0.01,
            AvatarMode.speaking: 0.03,
            AvatarMode.reacting: 0.04,
            AvatarMode.curious: 0.03,
            AvatarMode.surprised: 0.05,
            AvatarMode.shy: -0.02,
            AvatarMode.observe: 0.01,
            AvatarMode.retreat: -0.05,
            AvatarMode.peek: 0.02,
            AvatarMode.lean_in: 0.05,
            AvatarMode.rest: -0.03,
        }[mode]
        target = body.clone()
        target = target + breathing * 0.6 + micro * 0.4 + idle_drift * 0.5
        target[0] = target[0] + mode_bias
        target[1] = target[1] + 0.5 * mode_bias

        spatial_vec = self._pad_features(spatial_features, 8)
        workspace_vec = self._pad_features(workspace_features, 10)
        combined_scale_features = torch.cat([spatial_vec[:6], workspace_vec[:2]], dim=0)
        scale_goal = self.scale_transformer(combined_scale_features)
        place_x, place_y, alpha_goal = self.placement_transformer(torch.cat([spatial_vec, workspace_vec[:2]], dim=0))
        vis_goal = self.visibility_transformer(combined_scale_features)
        alpha_goal = torch.clamp(0.62 * alpha_goal + 0.38 * vis_goal, 0.0, 1.0)

        target_scale = 0.82 + 0.18 * scale_goal
        target = target * target_scale
        target[0] = target[0] + float(place_x) * 0.12
        target[1] = target[1] + float(place_y) * 0.12

        fluidity: Dict[str, float] = {}
        if self.animation_fluidity is not None:
            fluidity = self.animation_fluidity.step(AnimationFluidityInput(
                dt=float(dt.item()) if isinstance(dt, torch.Tensor) else float(dt),
                state_name=str(mode.value),
                mode_name=str(mode.value),
                intensity=float(emotion[0].abs().item()),
                energy=float(emotion[1].item()),
                arousal=float(emotion[2].item()),
                curiosity=float(emotion[4].item()),
                tension=float(emotion[3].item()),
                attention=float(attention.mean().item()),
                silence=float(1.0 - float(attention.mean().item())),
                speaking=(mode.value == 'speaking'),
                gesture_strength=float(abs(float(target[0].item())) * 0.15 + abs(float(target[1].item())) * 0.15),
                motion_speed=float(scale_goal.item()),
                mouse_speed=0.0,
                mouse_pressure=0.0,
                drag_pressure=0.0,
                fullscreen_pressure=0.0,
                workspace_density=0.0,
                occupancy_pressure=0.0,
                visibility_alpha=float(self.last_alpha.item()),
                body_scale=float(scale_goal.item()),
                body_lean_x=float(target[0].item()),
                body_lean_y=float(target[1].item()),
                head_x=float(target[0].item()),
                head_y=float(target[1].item()),
                rotation=float(target[2].item()) if target.numel() > 2 else 0.0,
                target_x=float(self.last_place_x.item()),
                target_y=float(self.last_place_y.item()),
                target_z=float(self.last_alpha.item()),
                target_alpha=float(self.last_alpha.item()),
                transition_bias=0.5,
                allow_micro_motion=True,
            ))
            target[0] = target[0] + torch.tensor(fluidity.get('secondary_x', 0.0) * 0.20 + fluidity.get('micro_x', 0.0) * 0.10, device=self.device, dtype=self.dtype)
            target[1] = target[1] + torch.tensor(fluidity.get('secondary_y', 0.0) * 0.20 + fluidity.get('micro_y', 0.0) * 0.10, device=self.device, dtype=self.dtype)
            target = target * torch.tensor(fluidity.get('scale', 1.0), device=self.device, dtype=self.dtype)

        self.last_scale.copy_(scale_goal.detach())
        self.last_alpha.copy_(alpha_goal.detach())
        self.last_place_x.copy_(place_x.detach())
        self.last_place_y.copy_(place_y.detach())
        return clamp11(target)


# =========================
# Audio Sync Engine
# =========================


class AudioSyncEngine(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer("mouth_state", torch.tensor(0.05, device=device, dtype=dtype))
        self.register_buffer("attack_state", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("decay_state", torch.tensor(0.0, device=device, dtype=dtype))

    def forward(self, audio_features: torch.Tensor, emotion: torch.Tensor, policy: PolicyOutput, dt: torch.Tensor) -> torch.Tensor:
        energy, envelope, attack, decay, pause, silence, voiced = audio_features
        target = 0.08 + 0.62 * energy + 0.16 * envelope + 0.12 * attack + 0.10 * voiced - 0.20 * silence
        target = target * (0.85 + 0.15 * policy.intensity)
        target = target * (0.92 + 0.08 * emotion[1])
        self.mouth_state.copy_(smooth_update(self.mouth_state, clamp01(target.unsqueeze(0)).squeeze(0), 1.0 - torch.exp(-8.0 * dt)))
        self.attack_state.copy_(smooth_update(self.attack_state, attack, 1.0 - torch.exp(-10.0 * dt)))
        self.decay_state.copy_(smooth_update(self.decay_state, decay, 1.0 - torch.exp(-10.0 * dt)))
        return self.mouth_state


# =========================
# Renderer Adapter
# =========================


class RendererAdapter:
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        self.device = device
        self.dtype = dtype

    def pack(self, mode: AvatarMode, body: torch.Tensor, face: torch.Tensor, gaze_target: torch.Tensor, mouth: torch.Tensor, motion: torch.Tensor, policy: PolicyOutput, motion_meta: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor | str]:
        payload: Dict[str, torch.Tensor | str] = {
            "mode": mode.value,
            "body": body.detach().clone(),
            "face": face.detach().clone(),
            "gaze_target": gaze_target.detach().clone(),
            "mouth": mouth.detach().clone(),
            "motion": motion.detach().clone(),
            "confidence": policy.confidence.detach().clone(),
            "curiosity": policy.curiosity.detach().clone(),
            "tension": policy.tension.detach().clone(),
            "pause_ms": policy.pause_ms.detach().clone(),
        }
        if motion_meta:
            payload["motion_meta"] = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in motion_meta.items()}
        return payload


# =========================
# LLM Controller
# =========================


class LLMController(nn.Module):
    def __init__(self, vocab_size: int = 256, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.adapter = nn.Linear(32 + 6 + 6 + 10, vocab_size)

    def forward(self, policy: PolicyOutput, emotion: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        x = torch.cat([policy.text, policy.emotion, emotion, attention[:10]], dim=-1)
        logits = self.adapter(x)
        return logits


# =========================
# Core Avatar Brain
# =========================


class AvatarTransformerBrain(nn.Module):
    def __init__(self, state_path: str = "avatar_state.json", device: torch.device = DEVICE, dtype: torch.dtype = DTYPE, scale: Optional[TransformerScaleConfig] = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.scale = scale or TransformerScaleConfig()
        self.state_manager = StateManager(state_path, device=device, dtype=dtype)
        self.perception_engine = PerceptionEngine(device=device, dtype=dtype)
        self.memory_engine = MemoryEngine(device=device, dtype=dtype)
        self.emotion_engine = EmotionEngine(device=device, dtype=dtype)
        self.attention_engine = AttentionEngine(device=device, dtype=dtype)
        self.transformer_policy = TransformerPolicy(
            d_model=self.scale.d_model,
            nhead=self.scale.nhead,
            num_layers=self.scale.num_layers,
            dim_feedforward=self.scale.dim_feedforward,
            dropout=self.scale.dropout,
            device=device,
            dtype=dtype,
        )
        self.behavior_engine = BehaviorEngine(device=device, dtype=dtype)
        self.motion_engine = MotionEngine(device=device, dtype=dtype)
        self.scale_transformer = self.motion_engine.scale_transformer
        self.placement_transformer = self.motion_engine.placement_transformer
        self.visibility_transformer = self.motion_engine.visibility_transformer
        self.audio_sync_engine = AudioSyncEngine(device=device, dtype=dtype)
        self.occupancy_controller = OccupancyVisibilityController(device=device, dtype=dtype) if OccupancyVisibilityController is not None else None
        self.llm_controller = LLMController(device=device, dtype=dtype)
        self.renderer = RendererAdapter(device=device, dtype=dtype)

        self.hidden_state = self.state_manager.load()
        emotion, body, face = self.state_manager.tensorize_state(self.hidden_state, device=device, dtype=dtype)
        self.register_buffer("emotion_state", emotion)
        self.register_buffer("body_state", body)
        self.register_buffer("face_state", face)
        self.register_buffer("last_attention_target", torch.tensor(self.hidden_state.last_attention_target, device=device, dtype=dtype))
        self.register_buffer("last_gaze_target", torch.tensor(self.hidden_state.last_gaze_target, device=device, dtype=dtype))
        self.register_buffer("step_counter", torch.tensor(float(self.hidden_state.step), device=device, dtype=dtype))
        self.register_buffer("silence_clock", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("repeat_clock", torch.tensor(0.0, device=device, dtype=dtype))

    def build_perception_input(self, obs: Dict[str, torch.Tensor]) -> PerceptionInput:
        return PerceptionInput(
            text_tokens=obs["text_tokens"],
            conversation_embedding=obs["conversation_embedding"],
            emotion_state=self.emotion_state,
            memory_short=self.memory_engine.short_term.flatten()[:64],
            memory_consolidated=self.memory_engine.consolidated.flatten()[:64],
            audio_features=obs["audio_features"],
            mouse_features=obs["mouse_features"],
            click_focus_features=obs["click_focus_features"],
            silence_features=obs["silence_features"],
            repetition_features=obs["repetition_features"],
            contextual_salience=obs["contextual_salience"],
            body_state=self.body_state,
            face_state=self.face_state,
        )

    def pack_policy_tokens(self, perception: torch.Tensor, memory_summary: torch.Tensor, emotion: torch.Tensor, attention: torch.Tensor, env_vec: torch.Tensor) -> torch.Tensor:
        base = torch.cat([perception, memory_summary[:64], emotion, attention, env_vec], dim=-1)
        token_a = base
        token_b = torch.roll(base, shifts=8, dims=0)
        token_c = torch.roll(base, shifts=16, dims=0)
        token_d = torch.roll(base, shifts=24, dims=0)
        tokens = torch.stack([token_a, token_b, token_c, token_d], dim=0).unsqueeze(0)
        return tokens

    def _build_space_input(self, obs: Dict[str, torch.Tensor], perception: torch.Tensor, emotion: torch.Tensor, attention: torch.Tensor, policy: PolicyOutput, mode: AvatarMode, dt: float) -> Optional[SpaceOccupancyInput]:
        if self.occupancy_controller is None or SpaceOccupancyInput is None:
            return None
        screen_w = float(obs.get('screen_width', torch.tensor(1920.0)).item() if isinstance(obs.get('screen_width'), torch.Tensor) else obs.get('screen_width', 1920.0))
        screen_h = float(obs.get('screen_height', torch.tensor(1080.0)).item() if isinstance(obs.get('screen_height'), torch.Tensor) else obs.get('screen_height', 1080.0))
        mouse = obs.get('mouse_features')
        mouse_x = float(mouse[0].item()) if mouse is not None and mouse.numel() > 0 else screen_w * 0.5
        mouse_y = float(mouse[1].item()) if mouse is not None and mouse.numel() > 1 else screen_h * 0.5
        mouse_speed = float(torch.norm(mouse, p=2).item()) if mouse is not None and mouse.numel() > 0 else 0.0
        density = float(obs.get('workspace_density', torch.tensor(0.0)).item() if isinstance(obs.get('workspace_density'), torch.Tensor) else obs.get('workspace_density', 0.0))
        occupancy = float(obs.get('occupancy_pressure', torch.tensor(0.0)).item() if isinstance(obs.get('occupancy_pressure'), torch.Tensor) else obs.get('occupancy_pressure', 0.0))
        fullscreen = float(obs.get('fullscreen_pressure', torch.tensor(0.0)).item() if isinstance(obs.get('fullscreen_pressure'), torch.Tensor) else obs.get('fullscreen_pressure', 0.0))
        heatmap = obs.get('screen_heatmap')
        occupied = obs.get('occupied_rects')
        return SpaceOccupancyInput(
            avatar_x=float(obs.get('avatar_x', torch.tensor(0.0)).item() if isinstance(obs.get('avatar_x'), torch.Tensor) else obs.get('avatar_x', 0.0)),
            avatar_y=float(obs.get('avatar_y', torch.tensor(0.0)).item() if isinstance(obs.get('avatar_y'), torch.Tensor) else obs.get('avatar_y', 0.0)),
            avatar_scale=float(obs.get('avatar_scale', torch.tensor(1.0)).item() if isinstance(obs.get('avatar_scale'), torch.Tensor) else obs.get('avatar_scale', 1.0)),
            avatar_alpha=float(obs.get('avatar_alpha', torch.tensor(1.0)).item() if isinstance(obs.get('avatar_alpha'), torch.Tensor) else obs.get('avatar_alpha', 1.0)),
            mouse_x=mouse_x,
            mouse_y=mouse_y,
            mouse_dx=float(obs.get('mouse_dx', torch.tensor(0.0)).item() if isinstance(obs.get('mouse_dx'), torch.Tensor) else obs.get('mouse_dx', 0.0)),
            mouse_dy=float(obs.get('mouse_dy', torch.tensor(0.0)).item() if isinstance(obs.get('mouse_dy'), torch.Tensor) else obs.get('mouse_dy', 0.0)),
            mouse_speed=mouse_speed,
            mouse_pressure=clamp(1.0 - float(obs.get('mouse_distance_norm', torch.tensor(0.0)).item() if isinstance(obs.get('mouse_distance_norm'), torch.Tensor) else obs.get('mouse_distance_norm', 0.0)), 0.0, 1.0),
            focus=bool(obs.get('focus', True)),
            click_pressure=1.0 if bool(obs.get('click', False)) else 0.0,
            fullscreen_pressure=fullscreen,
            silence_pressure=float(obs.get('silence_pressure', torch.tensor(0.0)).item() if isinstance(obs.get('silence_pressure'), torch.Tensor) else obs.get('silence_pressure', 0.0)),
            workspace_density=density,
            occupancy_pressure=occupancy,
            idle_seconds=float(obs.get('idle_seconds', torch.tensor(0.0)).item() if isinstance(obs.get('idle_seconds'), torch.Tensor) else obs.get('idle_seconds', 0.0)),
            time_since_input=float(obs.get('time_since_input', torch.tensor(0.0)).item() if isinstance(obs.get('time_since_input'), torch.Tensor) else obs.get('time_since_input', 0.0)),
            screen_width=screen_w,
            screen_height=screen_h,
            safe_margin=24.0,
            float_bias=float(policy.motion_speed.item()) if hasattr(policy.motion_speed, 'item') else float(policy.motion_speed),
            attention_pressure=float(attention.mean().item()) if hasattr(attention, 'mean') else 0.0,
            front_bias=0.62 if float(obs.get('idle_seconds', torch.tensor(0.0)).item() if isinstance(obs.get('idle_seconds'), torch.Tensor) else obs.get('idle_seconds', 0.0)) > 5.0 else 0.35,
            mouse_hot_radius=260.0,
            cursor_locked=bool(obs.get('cursor_locked', False)),
            heatmap=heatmap,
            occupied_rects=occupied,
            active_window_rect=obs.get('active_window_rect'),
            dragged_window_rect=obs.get('dragged_window_rect'),
            fullscreen_window_rect=obs.get('fullscreen_window_rect'),
            z_order_hint=float(obs.get('z_order_hint', 0.0)),
            interaction_intensity=float(obs.get('interaction_intensity', 0.0)),
            window_velocity_x=float(obs.get('window_velocity_x', 0.0)),
            window_velocity_y=float(obs.get('window_velocity_y', 0.0)),
            talking=bool(obs.get('talking', False)),
            window_count=int(obs.get('window_count', 0)),
            drag_active=bool(obs.get('drag_active', False)),
            allow_return_to_last_safe=bool(obs.get('allow_return_to_last_safe', True)),
            hold_lock_seconds=float(obs.get('hold_lock_seconds', 0.85)),
            preferred_slot_index=int(obs.get('preferred_slot_index', -1)),
            return_bias=float(obs.get('return_bias', 0.35)),
            occupied_rect_bias=float(obs.get('occupied_rect_bias', 0.60)),
            workspace_focus_bias=float(obs.get('workspace_focus_bias', 0.25)),
        )



    def step(self, obs: Dict[str, torch.Tensor], dt: float = 0.016) -> AvatarBehaviorFrame:
        dt_t = torch.tensor(float(dt), device=self.device, dtype=self.dtype)
        input_bundle = self.build_perception_input(obs)
        perception = self.perception_engine(input_bundle)
        memory_summary = self.memory_engine.summary()
        memory_context = self.memory_engine.retrieve(perception)
        combined_context = torch.cat([perception, memory_summary[:32], memory_context[:32]], dim=-1)

        audio = obs["audio_features"]
        mouse = obs["mouse_features"]
        repetition = obs["repetition_features"][0]
        silence = obs["silence_features"][0]
        novelty = obs["contextual_salience"].mean() + 0.15 * torch.norm(mouse, p=2)
        priority = obs["contextual_salience"].max() + 0.20 * audio[0] + 0.12 * obs["click_focus_features"].max()
        fatigue = self.emotion_state[4]

        emotion = self.emotion_engine(perception, memory_summary, combined_context[:8])
        attention_weights = self.attention_engine(perception, emotion, memory_summary, repetition, fatigue, novelty, priority)

        memory_vec = torch.cat([perception, memory_summary[:32], emotion, attention_weights], dim=-1)
        env_vec = torch.cat([audio, mouse, obs["click_focus_features"], obs["silence_features"], obs["repetition_features"], obs["contextual_salience"]], dim=-1)
        tokens = self.pack_policy_tokens(perception, memory_summary, emotion, attention_weights, env_vec)
        policy_hidden = self.transformer_policy(tokens)
        policy = self.transformer_policy.decode(policy_hidden)

        self.repeat_clock.copy_(smooth_update(self.repeat_clock, repetition, 1.0 - torch.exp(-2.5 * dt_t)))
        self.silence_clock.copy_(smooth_update(self.silence_clock, silence, 1.0 - torch.exp(-2.5 * dt_t)))

        mode = self.behavior_engine.choose_mode(policy, emotion, attention_weights, silence, repetition)
        body, face = self.behavior_engine.build_body_face(policy, emotion, attention_weights, self.body_state, self.face_state, dt_t, repetition)
        workspace_features = obs.get("workspace_features")
        if workspace_features is None:
            workspace_features = torch.tensor([
                float(perception[0].item()) if perception.numel() > 0 else 0.0,
                float(emotion[1].item()),
                float(audio[0].item()) if audio.numel() > 0 else 0.0,
                float(mouse[0].item()) if mouse.numel() > 0 else 0.0,
                float(silence.item()),
                float(perception.abs().mean().item()),
                float(priority.item()),
                float(novelty.item()),
                float(attention_weights.mean().item()),
                float(self.emotion_state[4].item()),
            ], device=self.device, dtype=self.dtype)
        spatial_features = obs.get("spatial_features")
        if spatial_features is None:
            spatial_features = torch.tensor([
                float(emotion.intensity.item()),
                float(emotion.arousal.item()),
                float(audio[0].item()) if audio.numel() > 0 else 0.0,
                float(torch.clamp(1.0 - self.silence_clock / 18.0, 0.0, 1.0).item()),
                float(torch.clamp(torch.norm(mouse, p=2) / 4.0, 0.0, 1.0).item()),
                float(torch.clamp(1.0 - repetition + 0.10 * self.repeat_clock, 0.0, 1.0).item()),
                float(attention_weights[0].item()),
                float(attention_weights[1].item()),
            ], device=self.device, dtype=self.dtype)

        motion = self.motion_engine(body, emotion, attention_weights, dt_t, mode, spatial_features=spatial_features, workspace_features=workspace_features)
        space_inp = self._build_space_input(obs, perception, emotion, attention_weights, policy, mode, dt)
        if space_inp is not None and self.occupancy_controller is not None:
            space_state = self.occupancy_controller.step(space_inp, dt=dt)
            motion = motion.clone()
            motion[0] = clamp11(motion[0] + torch.tensor(space_state.slide_x * 0.12, device=self.device, dtype=self.dtype))
            motion[1] = clamp11(motion[1] + torch.tensor(space_state.slide_y * 0.12, device=self.device, dtype=self.dtype))
            motion[3] = clamp01(motion[3] * torch.tensor(space_state.target_scale, device=self.device, dtype=self.dtype))
            motion[4] = clamp01(motion[4] * torch.tensor(space_state.target_scale, device=self.device, dtype=self.dtype))
            self.last_attention_target = torch.tensor([space_state.target_x, space_state.target_y, space_state.target_alpha, space_state.target_scale], device=self.device, dtype=self.dtype)
        else:
            self.last_attention_target = torch.tensor([0.0, 0.0, 1.0, 1.0], device=self.device, dtype=self.dtype)
        mouth = self.audio_sync_engine(audio, emotion, policy, dt_t)

        gaze_target = torch.stack([
            attention_weights[0] * (1.0 - obs["mouse_features"][0]),
            attention_weights[1] * obs["mouse_features"][1],
            attention_weights[2] * (1.0 - repetition),
        ])
        attention_target = torch.stack([
            attention_weights[0] + 0.15 * policy.curiosity,
            attention_weights[1] + 0.10 * policy.confidence,
            attention_weights[2] + 0.10 * policy.tension,
            attention_weights[3] + 0.10 * policy.silence_response,
        ])

        emotion_update = smooth_update(self.emotion_state, emotion, 1.0 - torch.exp(-2.0 * dt_t))
        body_update = smooth_update(self.body_state, body, 1.0 - torch.exp(-3.0 * dt_t))
        face_update = smooth_update(self.face_state, face, 1.0 - torch.exp(-4.0 * dt_t))
        self.emotion_state.copy_(emotion_update)
        self.body_state.copy_(body_update)
        self.face_state.copy_(face_update)

        memory_write = self.memory_engine.write(
            x=memory_vec,
            emotional_weight=policy.emotion.abs().mean(),
            preference_weight=policy.confidence,
            dt=dt_t,
        )

        anti_loop = 0.04 * torch.sigmoid(repetition + self.repeat_clock)
        self.emotion_state[4] = clamp01(self.emotion_state[4] + 0.02 * anti_loop)
        self.emotion_state[1] = clamp01(self.emotion_state[1] + 0.01 * policy.intensity - 0.01 * self.emotion_state[4])

        self.step_counter.add_(1.0)
        self.hidden_state = self.state_manager.detensorize_state(
            self.hidden_state,
            self.emotion_state,
            self.body_state,
            self.face_state,
            mode,
            int(self.step_counter.item()),
            attention_target,
            gaze_target,
        )
        self.state_manager.save(self.hidden_state)

        return AvatarBehaviorFrame(
            mode=mode,
            policy=policy,
            motion=motion,
            mouth=mouth,
            gaze=gaze_target,
            attention_weights=attention_weights,
            memory_write=memory_write,
        )

    def render_state(self) -> Dict[str, torch.Tensor | str]:
        mode = AvatarMode(self.hidden_state.mode)
        dummy_policy = PolicyOutput(
            text=torch.zeros(32, device=self.device, dtype=self.dtype),
            emotion=torch.zeros(6, device=self.device, dtype=self.dtype),
            intensity=torch.zeros((), device=self.device, dtype=self.dtype),
            intention=torch.zeros(8, device=self.device, dtype=self.dtype),
            attention_target=torch.zeros(4, device=self.device, dtype=self.dtype),
            gaze_target=torch.zeros(3, device=self.device, dtype=self.dtype),
            gesture=torch.zeros(10, device=self.device, dtype=self.dtype),
            gesture_strength=torch.zeros((), device=self.device, dtype=self.dtype),
            body_posture=torch.zeros(6, device=self.device, dtype=self.dtype),
            motion_speed=torch.zeros((), device=self.device, dtype=self.dtype),
            pause_ms=torch.zeros((), device=self.device, dtype=self.dtype),
            silence_response=torch.zeros((), device=self.device, dtype=self.dtype),
            confidence=torch.zeros((), device=self.device, dtype=self.dtype),
            curiosity=torch.zeros((), device=self.device, dtype=self.dtype),
            tension=torch.zeros((), device=self.device, dtype=self.dtype),
            memory_update=torch.zeros(16, device=self.device, dtype=self.dtype),
            state_transition=torch.zeros(len(AvatarMode), device=self.device, dtype=self.dtype),
        )
        motion_meta = {"scale": self.motion_engine.last_scale, "alpha": self.motion_engine.last_alpha, "place_x": self.motion_engine.last_place_x, "place_y": self.motion_engine.last_place_y, "fluidity": self.motion_engine.animation_fluidity.to_dict() if getattr(self.motion_engine, "animation_fluidity", None) is not None else {}}
        return self.renderer.pack(mode, self.body_state, self.face_state, self.last_gaze_target, torch.tensor(0.0, device=self.device, dtype=self.dtype), torch.zeros(6, device=self.device, dtype=self.dtype), dummy_policy, motion_meta=motion_meta)


# Factory helpers for 1B-class scale

def build_avatar_brain_1b(state_path: str = "avatar_state.json", device: torch.device = DEVICE, dtype: torch.dtype = DTYPE) -> AvatarTransformerBrain:
    scale = TransformerScaleConfig(d_model=3072, nhead=24, num_layers=30, dim_feedforward=12288, dropout=0.05)
    return AvatarTransformerBrain(state_path=state_path, device=device, dtype=dtype, scale=scale)


def estimate_transformer_params(scale: TransformerScaleConfig) -> int:
    d = scale.d_model
    ff = scale.dim_feedforward
    layers = scale.num_layers
    per_layer = (4 * d * d) + (2 * d * ff) + (8 * d)
    embed = (64 + 64 + 6 + 10 + 12) * d
    output_heads = (32 + 6 + 1 + 8 + 4 + 3 + 10 + 1 + 6 + 1 + 1 + 1 + 1 + 16 + len(AvatarMode)) * d
    return int(embed + layers * per_layer + output_heads)


# =========================
# Demo / Example Execution
# =========================


def build_observation(
    text_tokens: torch.Tensor,
    conversation_embedding: torch.Tensor,
    audio_features: torch.Tensor,
    mouse_features: torch.Tensor,
    click_focus_features: torch.Tensor,
    silence_features: torch.Tensor,
    repetition_features: torch.Tensor,
    contextual_salience: torch.Tensor,
    body_state: torch.Tensor,
    face_state: torch.Tensor,
    spatial_features: Optional[torch.Tensor] = None,
    workspace_features: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    obs = {
        "text_tokens": text_tokens,
        "conversation_embedding": conversation_embedding,
        "audio_features": audio_features,
        "mouse_features": mouse_features,
        "click_focus_features": click_focus_features,
        "silence_features": silence_features,
        "repetition_features": repetition_features,
        "contextual_salience": contextual_salience,
        "body_state": body_state,
        "face_state": face_state,
    }
    if spatial_features is not None:
        obs["spatial_features"] = spatial_features
    if workspace_features is not None:
        obs["workspace_features"] = workspace_features
    return obs


def build_spatial_features(
    *,
    intensity: float,
    arousal: float,
    audio_drive: float,
    silence_drive: float,
    mouse_pressure: float,
    repetition: float,
    attention_mouse: float,
    attention_click: float,
    visibility_alpha: float,
    occupancy_pressure: float,
) -> torch.Tensor:
    return torch.tensor([
        float(intensity),
        float(arousal),
        float(audio_drive),
        float(silence_drive),
        float(mouse_pressure),
        float(repetition),
        float(attention_mouse),
        float(attention_click),
        float(visibility_alpha),
        float(occupancy_pressure),
    ], device=DEVICE, dtype=DTYPE)


def build_workspace_features(
    *,
    workspace_density: float,
    fullscreen_pressure: float,
    mouse_pressure: float,
    click_pressure: float,
    silence_pressure: float,
    idle_seconds: float,
    active_window_width: float = 0.0,
    active_window_height: float = 0.0,
    z_order_hint: float = 0.5,
    talking: float = 0.0,
) -> torch.Tensor:
    return torch.tensor([
        float(workspace_density),
        float(fullscreen_pressure),
        float(mouse_pressure),
        float(click_pressure),
        float(silence_pressure),
        float(idle_seconds),
        float(active_window_width),
        float(active_window_height),
        float(z_order_hint),
        float(talking),
    ], device=DEVICE, dtype=DTYPE)



# ======================================================================
# Transformed Avatar Brain2 merged below
# ======================================================================

import json
import math
import queue
import random
import threading
import time
import pickle
import os
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageTk

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from model_manager import ModelManager
except Exception:
    ModelManager = None

try:
    from tts_local import LocalTTSManager
except Exception:
    LocalTTSManager = None

# ----------------------------------------------------------------------
#  CONSTANTES E UTILITÁRIOS
# ----------------------------------------------------------------------

STATE_NAMES = [
    "idle", "listening", "thinking", "speaking", "reacting",
    "curious", "surprised", "shy", "hide", "peek", "retreat",
    "lean_in", "observe", "rest"
]

EMOTIONS = [
    "neutral", "curious", "happy", "calm", "focused",
    "surprised", "shy", "confident", "tired", "annoyed"
]

GESTURES = [
    "none", "nod", "tilt_left", "tilt_right", "lean_in", "retreat",
    "peek", "micro_shrug", "soft_smile", "glance_away", "present"
]

GAZE_TARGETS = ["user", "mouse", "away", "internal", "edge", "text"]


def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-dt / tau)
    return lerp(current, target, k)


def soft_noise(seed: float, t: float, freq: float = 1.0) -> float:
    return (math.sin(t * freq + seed) * 0.5 +
            math.sin(t * freq * 1.71 + seed * 1.37) * 0.32 +
            math.sin(t * freq * 2.63 + seed * 0.73) * 0.18)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def text_summary(text: str, max_len: int = 120) -> str:
    t = " ".join((text or "").strip().split())
    return t[:max_len]


# ----------------------------------------------------------------------
#  DATACLASSES PARA ESTADOS E PERCEPÇÃO
# ----------------------------------------------------------------------

@dataclass
class PerceptionSnapshot:
    t: float = 0.0
    dt: float = 0.016
    mouse_x: int = 0
    mouse_y: int = 0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    mouse_distance_to_face: float = 0.0
    click: bool = False
    double_click: bool = False
    focus: bool = True
    silence_s: float = 0.0
    audio_rms: float = 0.0
    audio_peak: float = 0.0
    audio_attack: float = 0.0
    audio_voiced: bool = False
    user_text: str = ""
    user_text_len: int = 0
    speaking: bool = False
    window_active: bool = True
    turn_index: int = 0


@dataclass
class InternalState:
    """Estado interno contínuo do avatar."""
    valence: float = 0.0
    arousal: float = 0.35
    dominance: float = 0.4
    attention: float = 0.48
    curiosity: float = 0.32
    confidence: float = 0.52
    tension: float = 0.12
    fatigue: float = 0.08
    social_drive: float = 0.28
    energy: float = 0.45
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.valence, self.arousal, self.dominance, self.attention,
            self.curiosity, self.confidence, self.tension, self.fatigue,
            self.social_drive, self.energy
        ], dtype=torch.float32)
    
    def from_tensor(self, t: torch.Tensor):
        self.valence = float(t[0])
        self.arousal = float(t[1])
        self.dominance = float(t[2])
        self.attention = float(t[3])
        self.curiosity = float(t[4])
        self.confidence = float(t[5])
        self.tension = float(t[6])
        self.fatigue = float(t[7])
        self.social_drive = float(t[8])
        self.energy = float(t[9])


@dataclass
class MemoryTurn:
    t: float
    speaker: str
    text: str
    emotion: str = "neutral"
    intention: str = "none"
    state: str = "idle"
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None


@dataclass
class TransformerOutput:
    text: str = ""
    emotion: str = "neutral"
    intensity: float = 0.35
    intention: str = "respond"
    attention_target: str = "user"
    gaze_target: str = "user"
    gesture: str = "none"
    gesture_strength: float = 0.35
    body_posture: Dict[str, float] = field(default_factory=dict)
    motion_speed: float = 0.45
    pause_ms: int = 120
    silence_response: bool = False
    confidence: float = 0.55
    curiosity: float = 0.5
    tension: float = 0.2
    memory_update: Dict[str, Any] = field(default_factory=dict)
    state_transition: str = "speaking"


# ----------------------------------------------------------------------
#  TRANSFORMER ARCHITECTURE
# ----------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class AvatarTransformer(nn.Module):
    """
    Transformer que atua como o cérebro decisor do avatar.
    Recebe embedding de contexto e produz ações estruturadas.
    """
    def __init__(self, 
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 vocab_size: int = 50000,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.state_embedding = nn.Linear(10, d_model)  # 10 dimensões do InternalState
        self.stimulus_embedding = nn.Linear(12, d_model)  # 12 estímulos de percepção
        self.memory_embedding = nn.Linear(128, d_model)  # embedding de memória consolidada
        
        # Posicional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.text_head = nn.Linear(d_model, vocab_size)
        self.emotion_head = nn.Linear(d_model, len(EMOTIONS))
        self.intensity_head = nn.Linear(d_model, 1)
        self.intention_head = nn.Linear(d_model, 8)  # 8 intenções
        self.gaze_head = nn.Linear(d_model, len(GAZE_TARGETS))
        self.gesture_head = nn.Linear(d_model, len(GESTURES))
        self.gesture_strength_head = nn.Linear(d_model, 1)
        self.motion_speed_head = nn.Linear(d_model, 1)
        self.pause_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)
        self.curiosity_head = nn.Linear(d_model, 1)
        self.tension_head = nn.Linear(d_model, 1)
        self.state_transition_head = nn.Linear(d_model, len(STATE_NAMES))
        
        # Memory projection
        self.memory_projection = nn.Linear(d_model, 128)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                token_ids: torch.Tensor,
                state: torch.Tensor,
                stimulus: torch.Tensor,
                memory: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len] tokens do texto do usuário
            state: [batch, 10] estado interno atual
            stimulus: [batch, 12] vetor de estímulos
            memory: [batch, 128] embedding de memória consolidada
            attention_mask: [batch, seq_len] máscara de atenção
        
        Returns:
            Dicionário com todas as saídas estruturadas
        """
        batch_size = token_ids.shape[0]
        seq_len = token_ids.shape[1]
        
        # Embeddings
        token_emb = self.token_embedding(token_ids)  # [batch, seq_len, d_model]
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch, 1, d_model]
        stimulus_emb = self.stimulus_embedding(stimulus).unsqueeze(1)  # [batch, 1, d_model]
        memory_emb = self.memory_embedding(memory).unsqueeze(1)  # [batch, 1, d_model]
        
        # Concatenar todas as fontes de informação
        combined = torch.cat([token_emb, state_emb, stimulus_emb, memory_emb], dim=1)
        
        # Adicionar posicional encoding
        combined = self.pos_encoding(combined)
        combined = self.dropout(combined)
        
        # Passar pelo transformer
        for layer in self.layers:
            combined = layer(combined, attention_mask)
        
        # Extrair o token [CLS] (último token combinado)
        cls_embedding = combined[:, -1, :]  # [batch, d_model]
        
        # Gerar saídas
        outputs = {
            'text_logits': self.text_head(cls_embedding),
            'emotion_logits': self.emotion_head(cls_embedding),
            'intensity': torch.sigmoid(self.intensity_head(cls_embedding)).squeeze(-1),
            'intention_logits': self.intention_head(cls_embedding),
            'gaze_logits': self.gaze_head(cls_embedding),
            'gesture_logits': self.gesture_head(cls_embedding),
            'gesture_strength': torch.sigmoid(self.gesture_strength_head(cls_embedding)).squeeze(-1),
            'motion_speed': torch.sigmoid(self.motion_speed_head(cls_embedding)).squeeze(-1),
            'pause_ms': torch.sigmoid(self.pause_head(cls_embedding)).squeeze(-1) * 500,  # 0-500ms
            'confidence': torch.sigmoid(self.confidence_head(cls_embedding)).squeeze(-1),
            'curiosity': torch.sigmoid(self.curiosity_head(cls_embedding)).squeeze(-1),
            'tension': torch.sigmoid(self.tension_head(cls_embedding)).squeeze(-1),
            'state_transition_logits': self.state_transition_head(cls_embedding),
            'memory_update': self.memory_projection(cls_embedding)
        }
        
        return outputs
    
    def generate_action(self, 
                        token_ids: torch.Tensor,
                        state: torch.Tensor,
                        stimulus: torch.Tensor,
                        memory: torch.Tensor,
                        temperature: float = 0.7) -> TransformerOutput:
        """
        Gera uma ação estruturada a partir das entradas.
        """
        with torch.no_grad():
            outputs = self.forward(token_ids, state, stimulus, memory)
        
        # Decodificar saídas
        emotion_idx = torch.argmax(outputs['emotion_logits'], dim=-1).item()
        intention_idx = torch.argmax(outputs['intention_logits'], dim=-1).item()
        gaze_idx = torch.argmax(outputs['gaze_logits'], dim=-1).item()
        gesture_idx = torch.argmax(outputs['gesture_logits'], dim=-1).item()
        state_idx = torch.argmax(outputs['state_transition_logits'], dim=-1).item()
        
        # Mapeamento de intenções (simplificado)
        intentions = ["respond", "ask", "explain", "empathize", "react", "observe", "wait", "initiate"]
        
        return TransformerOutput(
            text="",  # Será gerado pelo LLM
            emotion=EMOTIONS[emotion_idx],
            intensity=float(outputs['intensity'].item()),
            intention=intentions[intention_idx] if intention_idx < len(intentions) else "respond",
            attention_target=GAZE_TARGETS[gaze_idx],
            gaze_target=GAZE_TARGETS[gaze_idx],
            gesture=GESTURES[gesture_idx],
            gesture_strength=float(outputs['gesture_strength'].item()),
            body_posture={},
            motion_speed=float(outputs['motion_speed'].item()),
            pause_ms=int(outputs['pause_ms'].item()),
            silence_response=False,
            confidence=float(outputs['confidence'].item()),
            curiosity=float(outputs['curiosity'].item()),
            tension=float(outputs['tension'].item()),
            memory_update={},
            state_transition=STATE_NAMES[state_idx]
        )


# ----------------------------------------------------------------------
#  PERCEPTION ENGINE
# ----------------------------------------------------------------------

class PerceptionEngine:
    """Processa estímulos sensoriais em vetor de características."""
    
    def __init__(self):
        self.last_mouse_pos = (0, 0)
        self.last_mouse_time = 0.0
        
    def process(self, perception: PerceptionSnapshot) -> torch.Tensor:
        """
        Converte percepção bruta em vetor de características.
        Dimensão: 12
        """
        # Normalizar distâncias (0-1)
        mouse_dist_norm = min(1.0, perception.mouse_distance_to_face / 500.0)
        mouse_speed_norm = min(1.0, perception.mouse_speed / 1000.0)
        
        features = torch.tensor([
            mouse_dist_norm,                    # 0
            mouse_speed_norm,                   # 1
            1.0 if perception.click else 0.0,   # 2
            1.0 if perception.double_click else 0.0,  # 3
            perception.audio_rms,               # 4
            perception.audio_peak,              # 5
            perception.audio_attack,            # 6
            min(1.0, perception.silence_s / 30.0),  # 7
            1.0 if perception.audio_voiced else 0.0, # 8
            1.0 if perception.speaking else 0.0,     # 9
            min(1.0, perception.user_text_len / 100.0),  # 10
            1.0 if perception.focus else 0.0     # 11
        ], dtype=torch.float32)
        
        return features.unsqueeze(0)  # [batch=1, 12]


# ----------------------------------------------------------------------
#  MEMORY ENGINE
# ----------------------------------------------------------------------

class MemoryEngine:
    """Memória em camadas com consolidação e embeddings."""
    
    def __init__(self, embedding_dim: int = 128, short_limit: int = 24, long_limit: int = 200):
        self.short_term: Deque[MemoryTurn] = deque(maxlen=short_limit)
        self.long_term: List[MemoryTurn] = []
        self.long_limit = long_limit
        self.embedding_dim = embedding_dim
        self.topics: Counter[str] = Counter()
        self.user_preferences: Dict[str, Any] = {}
        self.session_notes: List[str] = []
        self.last_user_text: str = ""
        self.last_assistant_text: str = ""
        
        # Embedding model placeholder (usaria um modelo real)
        self.embedding_model = None
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding para texto (placeholder - implementar com modelo real)."""
        # Simulação: hash simples para embedding
        if not text:
            return np.zeros(self.embedding_dim)
        
        # Gerar embedding determinístico baseado no texto
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(self.embedding_dim) * 0.1
        return emb / np.linalg.norm(emb)
    
    def remember_turn(self, speaker: str, text: str, emotion: str = "neutral",
                      intention: str = "none", state: str = "idle",
                      importance: float = 0.5, t: float = 0.0) -> None:
        embedding = self._get_embedding(text)
        turn = MemoryTurn(
            t=t, speaker=speaker, text=text, emotion=emotion,
            intention=intention, state=state, importance=importance,
            embedding=embedding
        )
        self.short_term.append(turn)
        
        if speaker == "user":
            self.last_user_text = text
        else:
            self.last_assistant_text = text
            
        topic = self._extract_topic(text)
        if topic:
            self.topics[topic] += 1
    
    def _extract_topic(self, text: str) -> str:
        if not text:
            return ""
        words = [w.strip(".,!?;:()[]{}\"'\"").lower() for w in text.split()]
        words = [w for w in words if len(w) > 3]
        if not words:
            return ""
        stop = {"que", "para", "com", "uma", "isso", "esta", "esse", "sobre", "como"}
        filtered = [w for w in words if w not in stop]
        return filtered[0] if filtered else words[0]
    
    def consolidate(self, current_time: float, importance_threshold: float = 0.65) -> None:
        """Consolida memórias importantes para longo prazo."""
        for turn in list(self.short_term):
            if turn.importance >= importance_threshold:
                self.long_term.append(turn)
        
        if len(self.long_term) > self.long_limit:
            self.long_term = self.long_term[-self.long_limit:]
    
    def get_consolidated_embedding(self) -> torch.Tensor:
        """Gera embedding consolidado da memória."""
        if not self.long_term and not self.short_term:
            return torch.zeros(1, self.embedding_dim)
        
        # Média dos embeddings das memórias mais importantes
        all_memories = list(self.short_term) + self.long_term
        important_memories = sorted(all_memories, key=lambda x: x.importance, reverse=True)[:10]
        
        if not important_memories:
            return torch.zeros(1, self.embedding_dim)
        
        embeddings = [m.embedding for m in important_memories if m.embedding is not None]
        if not embeddings:
            return torch.zeros(1, self.embedding_dim)
        
        avg_embedding = np.mean(embeddings, axis=0)
        return torch.from_numpy(avg_embedding).float().unsqueeze(0)
    
    def get_context(self) -> str:
        """Retorna resumo de contexto para LLM."""
        last_user = text_summary(self.last_user_text, 90)
        last_assistant = text_summary(self.last_assistant_text, 90)
        topics = ", ".join([f"{k}" for k, _ in self.topics.most_common(3)])
        return f"user={last_user} | assistant={last_assistant} | topics={topics}"


# ----------------------------------------------------------------------
#  EMOTION ENGINE
# ----------------------------------------------------------------------

class EmotionEngine:
    """Sistema emocional contínuo com inércia e recuperação."""
    
    def __init__(self):
        self.state = InternalState()
        self.emotion_label = "neutral"
        self._inertia = 0.15
        
    def update(self, stimulus: torch.Tensor, plan: Optional[TransformerOutput] = None,
               dt: float = 0.016) -> InternalState:
        """Atualiza estado emocional baseado em estímulos e plano."""
        
        # Extrair estímulos
        mouse_dist = float(stimulus[0, 0])
        audio_energy = float(stimulus[0, 4])
        silence = float(stimulus[0, 7])
        speaking = float(stimulus[0, 9]) > 0.5
        
        # Atualizar alvos baseados em estímulos
        target_valence = 0.0
        target_arousal = 0.35
        target_curiosity = 0.32
        target_confidence = 0.52
        target_tension = 0.12
        
        # Mouse próximo aumenta curiosidade e tensão
        if mouse_dist < 0.3:
            target_curiosity += 0.2
            target_tension += 0.1
        
        # Áudio aumenta arousal
        if audio_energy > 0.3:
            target_arousal += audio_energy * 0.4
        
        # Silêncio prolongado diminui arousal e aumenta fadiga
        if silence > 0.5:
            target_arousal -= 0.15
            self.state.fatigue += 0.01 * dt * 30
        
        # Falando aumenta confiança
        if speaking:
            target_confidence += 0.1
        
        # Plano pode modificar alvos
        if plan:
            if plan.emotion == "happy":
                target_valence += 0.3
                target_arousal += 0.2
            elif plan.emotion == "curious":
                target_curiosity += 0.25
            elif plan.emotion == "confident":
                target_confidence += 0.2
            elif plan.emotion == "shy":
                target_confidence -= 0.15
                target_tension += 0.1
            
            target_curiosity += plan.curiosity * 0.3
            target_tension += plan.tension * 0.2
        
        # Aplicar suavização
        tau = 0.2
        self.state.valence = exp_smooth(self.state.valence, clamp(target_valence, -1, 1), dt, tau)
        self.state.arousal = exp_smooth(self.state.arousal, clamp(target_arousal, 0, 1), dt, tau)
        self.state.curiosity = exp_smooth(self.state.curiosity, clamp(target_curiosity, 0, 1), dt, tau)
        self.state.confidence = exp_smooth(self.state.confidence, clamp(target_confidence, 0, 1), dt, tau)
        self.state.tension = exp_smooth(self.state.tension, clamp(target_tension, 0, 1), dt, tau)
        self.state.fatigue = exp_smooth(self.state.fatigue, clamp(self.state.fatigue, 0, 1), dt, tau * 1.5)
        
        # Energia
        target_energy = 0.45 + self.state.arousal * 0.3 - self.state.fatigue * 0.2
        self.state.energy = exp_smooth(self.state.energy, clamp(target_energy, 0, 1), dt, tau)
        
        # Determinar label de emoção
        self.emotion_label = self._get_emotion_label()
        
        return self.state
    
    def _get_emotion_label(self) -> str:
        """Converte estado contínuo para label discreta."""
        if self.state.fatigue > 0.7:
            return "tired"
        if self.state.tension > 0.7:
            return "annoyed"
        if self.state.arousal > 0.7 and self.state.valence < -0.1:
            return "surprised"
        if self.state.confidence > 0.7 and self.state.valence > 0.15:
            return "confident"
        if self.state.curiosity > 0.6 and self.state.arousal > 0.4:
            return "curious"
        if self.state.valence > 0.2 and self.state.arousal > 0.4:
            return "happy"
        if self.state.arousal < 0.25 and abs(self.state.valence) < 0.15:
            return "calm"
        if self.state.confidence > 0.55 and self.state.arousal > 0.4:
            return "focused"
        if self.state.valence < -0.2:
            return "shy"
        return "neutral"


# ----------------------------------------------------------------------
#  ATTENTION ENGINE
# ----------------------------------------------------------------------

class AttentionEngine:
    """Sistema de atenção com saliência e foco dinâmico."""
    
    def __init__(self):
        self.target = "user"
        self.target_point = (0.0, 0.0)
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.blink_timer = 0.0
        self.next_blink = random.uniform(2.0, 4.8)
        self.blink_amount = 0.0
        self.saccade_seed = random.random() * 100.0
        
        # Saliencia dos estímulos
        self.salience = {
            "mouse": 0.0, "click": 0.0, "speech": 0.0, "silence": 0.0, "text": 0.0
        }
        self.salience_decay = 0.95
        
    def update(self, perception: PerceptionSnapshot, state: InternalState,
               plan: Optional[TransformerOutput], dt: float) -> str:
        """Atualiza alvo de atenção."""
        
        # Atualizar saliência
        if perception.mouse_distance_to_face < 300:
            self.salience["mouse"] = 1.0
        else:
            self.salience["mouse"] *= self.salience_decay ** dt
        
        if perception.click or perception.double_click:
            self.salience["click"] = 1.0
        else:
            self.salience["click"] *= self.salience_decay ** dt
        
        if perception.audio_voiced or perception.user_text_len > 0:
            self.salience["speech"] = 1.0
        else:
            self.salience["speech"] *= self.salience_decay ** dt
        
        if perception.silence_s > 5.0:
            self.salience["silence"] = min(1.0, self.salience["silence"] + 0.2 * dt)
        else:
            self.salience["silence"] *= self.salience_decay ** dt
        
        # Decidir alvo
        if plan and plan.attention_target:
            self.target = plan.attention_target
        elif self.salience["click"] > 0.5:
            self.target = "user"
        elif self.salience["speech"] > 0.4:
            self.target = "user"
        elif self.salience["mouse"] > 0.6 and state.curiosity > 0.4:
            self.target = "mouse"
        elif self.salience["silence"] > 0.7 and state.curiosity > 0.5:
            self.target = "internal"
        elif state.curiosity > 0.6:
            self.target = "mouse"
        elif state.attention < 0.3:
            self.target = "away"
        else:
            self.target = "user"
        
        return self.target
    
    def target_point_for(self, target: str, perception: PerceptionSnapshot) -> Tuple[float, float]:
        """Retorna ponto de olhar para o alvo."""
        w, h = 1024.0, 1024.0
        if target == "user":
            return w / 2, h / 2 - 55
        if target == "mouse":
            return float(perception.mouse_x), float(perception.mouse_y)
        if target == "away":
            return w * 0.70, h * 0.34
        if target == "internal":
            return w * 0.42, h * 0.30
        return w / 2, h / 2
    
    def update_gaze(self, dt: float, target_point: Tuple[float, float],
                    face_center: Tuple[float, float]) -> Tuple[float, float]:
        """Atualiza posição do olhar com micro-sacadas."""
        tx, ty = target_point
        cx, cy = face_center
        dx = tx - cx
        dy = ty - cy
        
        # Micro-sacadas
        sacc = soft_noise(self.saccade_seed, time.time(), 2.5)
        dx += sacc * 8.0
        dy += sacc * 5.0
        
        self.gaze_x = exp_smooth(self.gaze_x, clamp(dx / 30.0, -1.0, 1.0), dt, 0.08)
        self.gaze_y = exp_smooth(self.gaze_y, clamp(dy / 28.0, -1.0, 1.0), dt, 0.08)
        
        return self.gaze_x, self.gaze_y
    
    def update_blink(self, dt: float, state: InternalState) -> float:
        """Atualiza estado de piscar."""
        self.blink_timer += dt
        if self.blink_amount <= 0.0 and self.blink_timer >= self.next_blink:
            self.blink_amount = 1.0
            self.blink_timer = 0.0
            base = 2.5 if state.arousal > 0.6 else 3.5
            self.next_blink = random.uniform(base, base + 2.0)
        
        if self.blink_amount > 0.0:
            self.blink_amount = max(0.0, self.blink_amount - dt * 12.0)
        
        return self.blink_amount


# ----------------------------------------------------------------------
#  BEHAVIOR ENGINE
# ----------------------------------------------------------------------

class BehaviorEngine:
    """Máquina de estados com transições suaves."""
    
    def __init__(self):
        self.current_state = "idle"
        self.state_enter_t = 0.0
        self.state_dwell = 0.0
        self.force_target_state: Optional[str] = None
        
        self.state_profiles = {
            "idle": {"speed": 0.30, "attention": 0.45, "energy": 0.35},
            "listening": {"speed": 0.38, "attention": 0.78, "energy": 0.42},
            "thinking": {"speed": 0.22, "attention": 0.38, "energy": 0.28},
            "speaking": {"speed": 0.78, "attention": 0.88, "energy": 0.76},
            "reacting": {"speed": 0.95, "attention": 0.92, "energy": 0.90},
            "curious": {"speed": 0.42, "attention": 0.82, "energy": 0.52},
            "surprised": {"speed": 0.96, "attention": 1.00, "energy": 0.95},
            "shy": {"speed": 0.25, "attention": 0.48, "energy": 0.33},
            "hide": {"speed": 0.18, "attention": 0.28, "energy": 0.22},
            "peek": {"speed": 0.34, "attention": 0.70, "energy": 0.41},
            "retreat": {"speed": 0.28, "attention": 0.36, "energy": 0.24},
            "lean_in": {"speed": 0.55, "attention": 0.86, "energy": 0.66},
            "observe": {"speed": 0.22, "attention": 0.50, "energy": 0.29},
            "rest": {"speed": 0.14, "attention": 0.22, "energy": 0.16},
        }
        
    def update(self, perception: PerceptionSnapshot, state: InternalState,
               plan: Optional[TransformerOutput], t: float) -> str:
        """Atualiza estado atual baseado em percepção e plano."""
        
        if self.force_target_state:
            self.current_state = self.force_target_state
            self.force_target_state = None
            self.state_enter_t = t
        
        elif plan and plan.state_transition:
            if plan.state_transition != self.current_state and t - self.state_enter_t > 0.3:
                self.current_state = plan.state_transition
                self.state_enter_t = t
        
        else:
            # Decisão baseada em heurística
            if perception.speaking:
                candidate = "speaking"
            elif perception.audio_voiced:
                candidate = "listening"
            elif perception.mouse_distance_to_face < 150 and state.confidence < 0.55:
                candidate = "retreat"
            elif perception.mouse_distance_to_face < 200 and state.curiosity > 0.48:
                candidate = "curious"
            elif perception.silence_s > 8.0:
                candidate = "observe"
            elif perception.silence_s > 14.0:
                candidate = "rest"
            else:
                candidate = "idle"
            
            if candidate != self.current_state and t - self.state_enter_t > 0.5:
                self.current_state = candidate
                self.state_enter_t = t
        
        self.state_dwell = t - self.state_enter_t
        return self.current_state
    
    def state_profile(self) -> Dict[str, float]:
        return self.state_profiles.get(self.current_state, self.state_profiles["idle"])


# ----------------------------------------------------------------------
#  MOTION ENGINE
# ----------------------------------------------------------------------

class MotionEngine:
    """Gera movimento contínuo e expressões faciais."""
    
    def __init__(self):
        self.seed = random.random() * 200.0
        self.body_lean_x = 0.0
        self.body_lean_y = 0.0
        self.head_x = 0.0
        self.head_y = 0.0
        self.eye_open = 1.0
        self.mouth_open = 0.0
        self.brow_left = 0.0
        self.brow_right = 0.0
        self.mouth_curve = 0.0
        
    def update(self, perception: PerceptionSnapshot, state: InternalState,
               behavior: BehaviorEngine, attention: AttentionEngine,
               plan: Optional[TransformerOutput], dt: float) -> Dict[str, Any]:
        """Atualiza todos os parâmetros de movimento."""
        
        profile = behavior.state_profile()
        t = perception.t
        
        # Respiração contínua
        breath = 0.5 + 0.5 * math.sin(t * 1.05 + self.seed)
        
        # Micro-movimentos
        micro_x = soft_noise(self.seed, t, 0.3) * 2.0
        micro_y = soft_noise(self.seed + 1, t, 0.35) * 1.5
        
        # Baseado em estado
        speed_factor = profile["speed"] * (0.6 + 0.4 * state.energy)
        
        # Movimento corporal
        target_lean_x = micro_x * 5.0 * state.curiosity
        target_lean_y = micro_y * 4.0 - state.fatigue * 3.0 + (breath - 0.5) * 6.0
        
        if behavior.current_state == "lean_in":
            target_lean_y -= 8.0
        elif behavior.current_state in {"retreat", "hide"}:
            target_lean_y += 6.0
            target_lean_x -= 4.0
        
        self.body_lean_x = exp_smooth(self.body_lean_x, target_lean_x, dt, 0.22)
        self.head_x = exp_smooth(self.head_x, target_head_x, dt, 0.18)
        
        # Cabeça
        target_head_x = micro_x * 4.0 + (perception.mouse_x - 512) / 512.0 * 8.0 * attention.salience["mouse"]
        target_head_y = micro_y * 3.0 - state.fatigue * 2.0
        
        self.head_x = exp_smooth(self.head_x, target_head_x, dt, 0.1)
        self.head_y = exp_smooth(self.head_y, target_head_y, dt, 0.1)
        
        # Expressões faciais baseadas em emoção
        if state.valence > 0.2:
            self.mouth_curve = 0.3 + state.valence * 0.4
            self.brow_left = -0.1
            self.brow_right = -0.1
        elif state.valence < -0.2:
            self.mouth_curve = -0.1
            self.brow_left = 0.15
            self.brow_right = 0.15
        else:
            self.mouth_curve = 0.1
            self.brow_left = 0.0
            self.brow_right = 0.0

        lips_together = clamp(0.62 - self.mouth_open * 0.55 - state.energy * 0.04, 0.0, 1.0)
        lip_corner_up = clamp(max(0.0, state.valence) * 0.55 + max(0.0, state.confidence - 0.45) * 0.22, 0.0, 1.0)
        lip_corner_down = clamp(max(0.0, -state.valence) * 0.35 + state.tension * 0.20, 0.0, 1.0)
        jaw_clench = clamp(state.tension * 0.55 + max(0.0, -state.valence) * 0.18, 0.0, 1.0)
        tongue_out = clamp(max(0.0, state.curiosity - 0.45) * 0.18 + (0.12 if state.current_state == 'thinking' else 0.0), 0.0, 1.0)
        tongue_tip_interdental = clamp(0.12 if state.current_state in {'thinking', 'speaking'} and state.curiosity > 0.42 else 0.0, 0.0, 1.0)
        upper_lip_raise = clamp(max(0.0, state.valence) * 0.26 + (0.12 if state.emotion == 'happy' else 0.0), 0.0, 1.0)
        lower_lip_depress = clamp(max(0.0, -state.valence) * 0.22 + (0.10 if state.current_state in {'reacting', 'surprised'} else 0.0), 0.0, 1.0)
        cheek_puff = clamp(max(0.0, state.arousal - 0.45) * 0.12 + (0.08 if state.emotion == 'surprised' else 0.0), 0.0, 1.0)
        cheek_suck = clamp(max(0.0, state.confidence - 0.55) * 0.10 + max(0.0, state.valence) * 0.10, 0.0, 1.0)
        mouth_corner_stretch = clamp(0.10 + max(0.0, state.valence) * 0.16 + state.intensity * 0.10, 0.0, 1.0)

        # Abertura dos olhos
        target_eye_open = 1.0 - state.fatigue * 0.3 + state.arousal * 0.2
        target_eye_open = clamp(target_eye_open, 0.6, 1.3)
        self.eye_open = exp_smooth(self.eye_open, target_eye_open, dt, 0.15)
        
        # Abertura da boca (baseada em áudio)
        if perception.speaking:
            target_mouth = perception.audio_rms * 0.8 + perception.audio_attack * 0.3
            target_mouth = clamp(target_mouth, 0.05, 0.85)
        else:
            target_mouth = 0.02 + state.energy * 0.05
        
        self.mouth_open = exp_smooth(self.mouth_open, target_mouth, dt, 0.08)
        
        return {
            "body_lean_x": self.body_lean_x,
            "body_lean_y": self.body_lean_y,
            "head_x": self.head_x,
            "head_y": self.head_y,
            "eye_open": self.eye_open,
            "mouth_open": self.mouth_open,
            "brow_left": self.brow_left,
            "brow_right": self.brow_right,
            "mouth_curve": self.mouth_curve,
            "breath": breath,
            "lips_together": lips_together,
            "lip_corner_up": lip_corner_up,
            "lip_corner_down": lip_corner_down,
            "jaw_clench": jaw_clench,
            "tongue_out": tongue_out,
            "tongue_tip_interdental": tongue_tip_interdental,
            "upper_lip_raise": upper_lip_raise,
            "lower_lip_depress": lower_lip_depress,
            "cheek_puff": cheek_puff,
            "cheek_suck": cheek_suck,
            "mouth_corner_stretch": mouth_corner_stretch
        }


# ----------------------------------------------------------------------
#  AUDIO SYNC ENGINE
# ----------------------------------------------------------------------

class AudioSyncEngine:
    """Sincronização de áudio em tempo real."""
    
    def __init__(self):
        self.energy = 0.0
        self.peak = 0.0
        self.attack = 0.0
        self.voiced = False
        self.speaking = False
        self.proxy_active = False
        self.proxy_time = 0.0
        self.proxy_duration = 0.0
        
    def feed(self, chunk: Optional[np.ndarray]) -> None:
        """Processa chunk de áudio."""
        if chunk is None:
            self.voiced = False
            return
        
        data = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return
        
        rms = float(np.sqrt(np.mean(np.square(data))))
        peak = float(np.max(np.abs(data)))
        
        self.energy = exp_smooth(self.energy, rms, 0.016, 0.08)
        self.peak = peak
        self.attack = max(0.0, (rms - self.energy) * 5.0)
        self.voiced = rms > 0.02 or peak > 0.05
        
    def start_proxy(self, text: str, duration: float = 2.0) -> None:
        """Inicia proxy de áudio (quando não há TTS)."""
        self.proxy_active = True
        self.proxy_time = 0.0
        self.proxy_duration = duration
        
    def update(self, dt: float, is_speaking: bool) -> None:
        """Atualiza estado de áudio."""
        self.speaking = is_speaking
        
        if self.proxy_active:
            self.proxy_time += dt
            if self.proxy_time >= self.proxy_duration:
                self.proxy_active = False
    
    def mouth_open(self, state: InternalState, plan: Optional[TransformerOutput]) -> float:
        """Calcula abertura da boca baseada em áudio e estado."""
        if self.speaking or self.proxy_active:
            base = self.energy * 0.7 + self.attack * 0.3
            base = clamp(base, 0.05, 0.85)
        else:
            base = 0.02 + state.energy * 0.05
        
        if plan and plan.state_transition == "speaking":
            base *= 1.2
        
        return clamp(base, 0.0, 1.0)


# ----------------------------------------------------------------------
#  LLM CONTROLLER
# ----------------------------------------------------------------------

class LLMController:
    """Controlador LLM para geração de texto estruturado."""
    
    def __init__(self):
        self.model = None
        self.available = False
        self.persona = (
            "Você é um avatar digital vivo, com presença, intenção e continuidade psicológica. "
            "Responda de forma natural, com emoção apropriada ao contexto. "
            "Suas respostas devem ser concisas e vivas."
        )
        
    def load(self) -> None:
        """Carrega modelo LLM."""
        if ModelManager is not None:
            try:
                self.model = ModelManager(models_folder="models")
                self.available = True
            except Exception:
                self.available = False
    
    def generate(self, user_text: str, context: str, state: InternalState,
                 transformer_plan: TransformerOutput) -> str:
        """Gera texto baseado no plano do transformer."""
        
        if self.available and self.model:
            try:
                prompt = f"{self.persona}\n\nContexto: {context}\n\n"
                prompt += f"Estado emocional: {transformer_plan.emotion} (intensidade {transformer_plan.intensity:.2f})\n"
                prompt += f"Intenção: {transformer_plan.intention}\n\n"
                prompt += f"Usuário: {user_text}\n\nAvatar:"
                
                result = self.model.generate(prompt, max_tokens=80, temperature=0.8)
                return result.strip()
            except Exception:
                pass
        
        # Fallback baseado no estado
        if transformer_plan.emotion == "happy":
            return f"Que bom! {transformer_plan.text or 'Estou feliz com isso!'}"
        elif transformer_plan.emotion == "curious":
            return f"Interessante! {transformer_plan.text or 'Me conta mais sobre isso.'}"
        elif transformer_plan.emotion == "confident":
            return transformer_plan.text or "Entendi. Vou ajudar com isso."
        elif transformer_plan.emotion == "shy":
            return transformer_plan.text or "Ah, entendi..."
        else:
            return transformer_plan.text or "Entendi. Vou refletir sobre isso."


# ----------------------------------------------------------------------
#  RENDERER
# ----------------------------------------------------------------------

class AvatarRenderer:
    """Renderizador visual do avatar."""
    
    def __init__(self, render_res: int = 1400, display_res: int = 900):
        self.render_res = render_res
        self.display_res = display_res
        self.cx = render_res // 2
        self.cy = render_res // 2 + 45
        self.head_radius = int(render_res * 0.22)
        
    def render(self, t: float, perception: PerceptionSnapshot, state: InternalState,
               motion: Dict[str, Any], attention: AttentionEngine,
               audio: AudioSyncEngine, plan: Optional[TransformerOutput]) -> Image.Image:
        """Renderiza o avatar com expressões atuais."""
        
        img = Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Corpo
        torso_cx = self.cx + motion["body_lean_x"] * 40
        torso_cy = self.cy + 255 + motion["body_lean_y"] * 24
        draw.ellipse((torso_cx - 210, torso_cy - 245, torso_cx + 210, torso_cy + 245),
                     fill=(18, 72, 112, 184))
        
        # Cabeça
        head_x = self.cx + motion["head_x"] * 12
        head_y = self.cy - 15 + motion["head_y"] * 10
        radius = self.head_radius
        
        # Gradiente da cabeça
        for i in range(58, 0, -1):
            r = radius * (i / 58) ** 0.92
            color = (20 + int(14 * (1 - i/58)),
                     88 + int(78 * (1 - i/58)),
                     180 + int(38 * (1 - i/58)))
            draw.ellipse((head_x - r, head_y - r, head_x + r, head_y + r),
                         fill=color)
        
        # Olhos
        eye_y = head_y - int(radius * 0.22)
        left_eye_x = head_x - int(radius * 0.34)
        right_eye_x = head_x + int(radius * 0.34)
        eye_h = 50 * state.arousal * motion["eye_open"]
        
        for ex in [left_eye_x, right_eye_x]:
            draw.ellipse((ex - 58, eye_y - eye_h, ex + 58, eye_y + eye_h),
                         fill=(240, 245, 255))
            
            # Íris com direção do olhar
            iris_cx = ex + attention.gaze_x * 12
            iris_cy = eye_y + attention.gaze_y * 10
            draw.ellipse((iris_cx - 30, iris_cy - 30, iris_cx + 30, iris_cy + 30),
                         fill=(12, 70, 150))
            draw.ellipse((iris_cx - 10, iris_cy - 10, iris_cx + 10, iris_cy + 10),
                         fill=(0, 0, 0))
        
        # Sobrancelhas
        brow_lift = -20 * motion["brow_left"]
        self._draw_brow(draw, left_eye_x, eye_y - 34, brow_lift, -5)
        self._draw_brow(draw, right_eye_x, eye_y - 34, -20 * motion["brow_right"], 5)
        
        # Boca
        mouth_y = head_y + 102
        mouth_w = 82 * (1.0 + 0.12 * motion.get("mouth_corner_stretch", 0.0) - 0.06 * motion.get("cheek_puff", 0.0) + 0.05 * motion.get("cheek_suck", 0.0))
        curve = motion["mouth_curve"] + 0.16 * motion.get("lip_corner_up", 0.0) - 0.12 * motion.get("lip_corner_down", 0.0) + 0.08 * motion.get("upper_lip_raise", 0.0) - 0.08 * motion.get("lower_lip_depress", 0.0)
        open_amt = clamp(motion["mouth_open"] * (1.0 - 0.32 * motion.get("lips_together", 0.0)) + 0.08 * motion.get("jaw_clench", 0.0) + 0.05 * motion.get("tongue_out", 0.0), 0.0, 1.0)
        
        pts = []
        for i in range(-9, 10):
            x = head_x + i * 8
            u = i / 9.0
            arch = math.sin((u + 1.0) * math.pi * 0.5)
            yy = mouth_y + curve * 18 * arch - open_amt * 8 * (1 - abs(u)) + 1.4 * motion.get("lower_lip_depress", 0.0) - 1.1 * motion.get("upper_lip_raise", 0.0)
            pts.append((x, yy))
        
        draw.line(pts, fill=(112, 52, 64), width=8)
        if motion.get("tongue_out", 0.0) > 0.08 or motion.get("tongue_tip_interdental", 0.0) > 0.08:
            tw = mouth_w * (0.15 + 0.12 * motion.get("tongue_out", 0.0))
            th = (18 + 38 * open_amt) * (0.15 + 0.10 * motion.get("tongue_tip_interdental", 0.0))
            draw.ellipse((head_x - tw, mouth_y + 14, head_x + tw, mouth_y + 14 + th), fill=(195, 120, 128))
        
        # Piscar
        if attention.blink_amount > 0.01:
            lid_h = eye_h * attention.blink_amount * 1.55
            for ex in [left_eye_x, right_eye_x]:
                draw.rounded_rectangle((ex - 64, eye_y - lid_h, ex + 64, eye_y + lid_h),
                                       radius=22, fill=(80, 112, 145))
        
        # Redimensionar para display
        img = img.resize((self.display_res, self.display_res), Image.Resampling.LANCZOS)
        return img
    
    def _draw_brow(self, draw: ImageDraw.Draw, bx: float, by: float, lift: float, tilt: float):
        pts = []
        for i in range(18):
            u = i / 17.0
            x = bx - 40 + 80 * u
            arch = math.sin(u * math.pi)
            y = by + lift + tilt * (u - 0.5) - arch * 12
            pts.append((x, y))
        draw.line(pts, fill=(22, 44, 70), width=10)


# ----------------------------------------------------------------------
#  STATE MANAGER
# ----------------------------------------------------------------------

class StateManager:
    """Gerencia persistência de estado entre sessões."""
    
    def __init__(self, save_path: str = "avatar_state.pkl"):
        self.save_path = save_path
        self.last_save = 0.0
        
    def save(self, memory: MemoryEngine, state: InternalState, t: float) -> None:
        """Salva estado atual em disco."""
        if t - self.last_save > 60.0:  # Salvar a cada minuto
            try:
                data = {
                    'memory_short': list(memory.short_term),
                    'memory_long': memory.long_term,
                    'topics': dict(memory.topics),
                    'preferences': memory.user_preferences,
                    'state': state,
                    'timestamp': t
                }
                with open(self.save_path, 'wb') as f:
                    pickle.dump(data, f)
                self.last_save = t
            except Exception:
                pass
    
    def load(self, memory: MemoryEngine, state: InternalState) -> bool:
        """Carrega estado anterior do disco."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                memory.short_term.extend(data.get('memory_short', []))
                memory.long_term.extend(data.get('memory_long', []))
                memory.topics.update(data.get('topics', {}))
                memory.user_preferences.update(data.get('preferences', {}))
                loaded_state = data.get('state')
                if loaded_state:
                    state.valence = loaded_state.valence
                    state.arousal = loaded_state.arousal
                    state.curiosity = loaded_state.curiosity
                    state.confidence = loaded_state.confidence
                return True
            except Exception:
                pass
        return False


# ----------------------------------------------------------------------
#  AVATAR APP - INTEGRAÇÃO PRINCIPAL
# ----------------------------------------------------------------------

class AvatarApp:
    """Aplicação principal do avatar."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Avatar Vivo - Transformer Brain")
        self.root.overrideredirect(True)
        self.root.configure(bg="#ffffff")
        try:
            self.root.attributes("-transparentcolor", "#ffffff")
        except Exception:
            pass
        
        self.display_res = 900
        self.root.geometry(f"{self.display_res}x{self.display_res}+120+90")
        
        self.canvas = tk.Canvas(self.root, width=self.display_res, height=self.display_res,
                                bg="#ffffff", highlightthickness=0, bd=0)
        self.canvas.pack()
        
        # Módulos
        self.perception = PerceptionEngine()
        self.memory = MemoryEngine()
        self.emotion = EmotionEngine()
        self.attention = AttentionEngine()
        self.behavior = BehaviorEngine()
        self.motion = MotionEngine()
        self.audio = AudioSyncEngine()
        self.llm = LLMController()
        self.renderer = AvatarRenderer()
        self.state_manager = StateManager()
        
        # Transformer
        self.transformer = AvatarTransformer(
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024
        )
        
        # Estado
        self.perception_data = PerceptionSnapshot()
        self.plan: Optional[TransformerOutput] = None
        self.is_speaking = False
        self.start_time = time.time()
        self.last_frame_t = self.start_time
        self.input_queue: queue.Queue = queue.Queue()
        
        # UI
        self.photo = None
        self.drag = {"x": 0, "y": 0}
        self.mouse_x = self.display_res // 2
        self.mouse_y = self.display_res // 2
        
        # Carregar estado salvo
        self.state_manager.load(self.memory, self.emotion.state)
        self.llm.load()
        
        self._setup_window()
        self._bind_events()
        self._start_console_thread()
        
    def _setup_window(self):
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))
    
    def _bind_events(self):
        self.root.bind("<Motion>", self._on_motion)
        self.root.bind("<Button-1>", self._on_click)
        self.root.bind("<Double-Button-1>", self._on_double_click)
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<ButtonPress-3>", self._drag_start)
        self.root.bind("<B3-Motion>", self._drag_move)
    
    def _drag_start(self, event):
        self.drag["x"] = event.x_root
        self.drag["y"] = event.y_root
    
    def _drag_move(self, event):
        dx = event.x_root - self.drag["x"]
        dy = event.y_root - self.drag["y"]
        geom = self.root.geometry()
        size_part, pos_part = geom.split("+", 1)
        w, h = map(int, size_part.split("x"))
        x, y = map(int, pos_part.split("+"))
        self.root.geometry(f"{w}x{h}+{x + dx}+{y + dy}")
        self.drag["x"] = event.x_root
        self.drag["y"] = event.y_root
    
    def _on_motion(self, event):
        self.mouse_x = clamp(event.x, 0, self.display_res)
        self.mouse_y = clamp(event.y, 0, self.display_res)
    
    def _on_click(self, _event):
        self.perception_data.click = True
    
    def _on_double_click(self, _event):
        self.perception_data.double_click = True
    
    def _on_focus_in(self, _event):
        self.perception_data.focus = True
    
    def _on_focus_out(self, _event):
        self.perception_data.focus = False
    
    def _start_console_thread(self):
        def loop():
            while True:
                try:
                    text = input("Você: ").strip()
                    if text:
                        if text.lower() in {"/quit", "/exit"}:
                            self.root.after(0, self.close)
                            break
                        self.input_queue.put(text)
                except Exception:
                    break
        threading.Thread(target=loop, daemon=True).start()
    
    def _distance_to_face(self) -> float:
        fx, fy = self.display_res // 2, self.display_res // 2 - 20
        return math.hypot(self.mouse_x - fx, self.mouse_y - fy)
    
    def _update_perception(self, t: float, dt: float, user_text: str = ""):
        click_recent = (t - self.last_click_time) < 0.18 if hasattr(self, 'last_click_time') else False
        double_recent = (t - self.last_click_time) < 0.08 if hasattr(self, 'last_click_time') else False
        
        if self.audio.voiced or self.is_speaking:
            self.silence_start = t
        silence_s = max(0.0, t - self.silence_start) if hasattr(self, 'silence_start') else 0.0
        
        self.perception_data = PerceptionSnapshot(
            t=t, dt=dt,
            mouse_x=self.mouse_x, mouse_y=self.mouse_y,
            mouse_distance_to_face=self._distance_to_face(),
            click=click_recent, double_click=double_recent,
            focus=self.perception_data.focus,
            silence_s=silence_s,
            audio_rms=self.audio.energy,
            audio_peak=self.audio.peak,
            audio_attack=self.audio.attack,
            audio_voiced=self.audio.voiced,
            user_text=user_text, user_text_len=len(user_text),
            speaking=self.is_speaking,
            turn_index=getattr(self, 'turn_index', 0)
        )
    
    def _begin_response(self, user_text: str):
        t = time.time() - self.start_time
        self.turn_index = getattr(self, 'turn_index', 0) + 1
        
        # Registrar na memória
        self.memory.remember_turn("user", user_text, emotion=self.emotion.emotion_label,
                                  state=self.behavior.current_state, t=t)
        
        # Preparar entradas para o transformer
        state_tensor = self.emotion.state.to_tensor().unsqueeze(0)
        stimulus_tensor = self.perception.process(self.perception_data)
        memory_tensor = self.memory.get_consolidated_embedding()
        
        # Criar token IDs placeholder (usar embedding de texto real)
        token_ids = torch.zeros(1, 10, dtype=torch.long)
        
        # Gerar plano do transformer
        self.plan = self.transformer.generate_action(
            token_ids, state_tensor, stimulus_tensor, memory_tensor
        )
        
        # Gerar texto com LLM
        context = self.memory.get_context()
        text = self.llm.generate(user_text, context, self.emotion.state, self.plan)
        self.plan.text = text
        
        # Registrar resposta na memória
        self.memory.remember_turn("assistant", text, emotion=self.plan.emotion,
                                  intention=self.plan.intention, state=self.plan.state_transition,
                                  t=t)
        
        # Forçar transição de estado
        self.behavior.force_target_state = self.plan.state_transition
        self.is_speaking = True
        self.audio.speaking = True
        self.audio.reset()
        
        # Proxy de áudio
        duration = max(1.2, len(text) * 0.05 + self.plan.pause_ms / 1000.0)
        self.audio.start_proxy(text, duration)
        
        # TTS (se disponível)
        if hasattr(self, 'tts') and self.tts:
            try:
                audio_gen = self.tts.synthesize(text)
                if audio_gen and sd:
                    threading.Thread(target=self._play_audio, args=(audio_gen,), daemon=True).start()
            except Exception:
                pass
        
        self.root.after(int(duration * 1000), self._stop_speaking)
    
    def _play_audio(self, audio_gen):
        try:
            for chunk in audio_gen:
                if chunk is not None:
                    self.audio.feed(chunk)
                    if sd:
                        sd.play(chunk, samplerate=24000)
                        sd.wait()
        except Exception:
            pass
    
    def _stop_speaking(self):
        self.is_speaking = False
        self.audio.speaking = False
        self.audio.reset()
    
    def _update_logic(self, user_text: str = ""):
        now = time.time()
        dt = min(0.033, max(0.008, now - self.last_frame_t))
        self.last_frame_t = now
        t = now - self.start_time
        
        self._update_perception(t, dt, user_text)
        
        # Processar estímulos
        stimulus_tensor = self.perception.process(self.perception_data)
        
        # Atualizar emoção
        self.emotion.update(stimulus_tensor, self.plan, dt)
        
        # Atualizar atenção
        attention_target = self.attention.update(self.perception_data, self.emotion.state, self.plan, dt)
        
        # Atualizar comportamento
        state_name = self.behavior.update(self.perception_data, self.emotion.state, self.plan, t)
        
        # Atualizar movimento
        motion_data = self.motion.update(self.perception_data, self.emotion.state,
                                          self.behavior, self.attention, self.plan, dt)
        
        # Atualizar áudio
        self.audio.update(dt, self.is_speaking)
        
        # Consolidar memória
        if t - getattr(self, '_last_consolidate', 0) > 5.0:
            self.memory.consolidate(t)
            self._last_consolidate = t
        
        # Salvar estado
        self.state_manager.save(self.memory, self.emotion.state, t)
        
        return motion_data
    
    def _draw(self):
        t = time.time() - self.start_time
        
        # Processar input do usuário
        pending_text = ""
        if not self.input_queue.empty():
            try:
                txt = self.input_queue.get_nowait()
                if txt:
                    pending_text = txt
                    self.last_click_time = t
                    self._begin_response(txt)
            except Exception:
                pass
        
        # Atualizar lógica
        motion_data = self._update_logic(pending_text)
        
        # Renderizar
        frame = self.renderer.render(t, self.perception_data, self.emotion.state,
                                      motion_data, self.attention, self.audio, self.plan)
        
        self.photo = ImageTk.PhotoImage(frame)
        self.canvas.delete("all")
        self.canvas.create_image(self.display_res // 2, self.display_res // 2,
                                 anchor="center", image=self.photo)
        
        self.root.after(16, self._draw)
    
    def run(self):
        self.silence_start = time.time()
        self.last_click_time = 0.0
        self._last_consolidate = 0.0
        self.turn_index = 0
        self._draw()
        self.root.mainloop()
    
    def close(self):
        try:
            self.state_manager.save(self.memory, self.emotion.state, time.time() - self.start_time)
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


def main():
    app = AvatarAppV2()
    app.run()



# ----------------------------------------------------------------------
# Unified helpers / compatibility
# ----------------------------------------------------------------------

def build_unified_avatar_brain(*args, **kwargs):
    """Compatibility entrypoint. Uses the latest merged AvatarApp/brain symbols."""
    return AvatarApp(*args, **kwargs)

__all__ = [name for name in globals().keys() if not name.startswith("_")]

# ----------------------------------------------------------------------
# Third-module phoneme → viseme bridge (professional lip-sync)
# ----------------------------------------------------------------------
try:
    from phoneme_viseme_sync import install_phoneme_viseme_sync
    install_phoneme_viseme_sync(AudioSyncEngine, AvatarRenderer, globals())
except Exception:
    pass

try:
    from avatar_orchestrator_experts import patch_all as _patch_all_experts
    _patch_all_experts(globals())
except Exception:
    pass

