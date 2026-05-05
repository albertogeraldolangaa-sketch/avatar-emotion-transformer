from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

DTYPE = torch.float32
DEVICE = torch.device('cpu')

try:
    from denise_math import clamp01, clamp11, exp_decay, cosine_similarity, safe_norm, op_179, project_179, normalize
except Exception:  # pragma: no cover - standalone fallback
    def clamp01(x):
        return max(0.0, min(1.0, float(x)))
    def clamp11(x):
        return max(-1.0, min(1.0, float(x)))
    def exp_decay(value: torch.Tensor, rate: float, dt: float) -> torch.Tensor:
        return value * torch.exp(torch.tensor(-rate * dt, device=value.device, dtype=value.dtype))
    def cosine_similarity(a, b, eps=1e-8):
        a = torch.as_tensor(a, dtype=DTYPE, device=DEVICE)
        b = torch.as_tensor(b, dtype=DTYPE, device=DEVICE)
        return torch.sum(a * b, dim=-1) / ((torch.norm(a, dim=-1) * torch.norm(b, dim=-1)) + eps)
    def safe_norm(x, dim=-1, keepdim=False, eps=1e-8):
        if isinstance(x, torch.Tensor):
            return torch.norm(x, p=2, dim=dim, keepdim=keepdim).clamp_min(eps)
        return max(float(torch.norm(torch.as_tensor(x, dtype=DTYPE))), eps)
    def op_179(x, y):
        x = torch.as_tensor(x, dtype=DTYPE, device=DEVICE)
        y = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
        return torch.tanh((x + y) * 1.4)
    def project_179(x, basis):
        x = torch.as_tensor(x, dtype=DTYPE, device=DEVICE)
        basis = torch.as_tensor(basis, dtype=DTYPE, device=DEVICE)
        coeff = torch.matmul(x, basis)
        return torch.matmul(coeff, basis.T)
    def normalize(v, dim=-1, eps=1e-8):
        v = torch.as_tensor(v, dtype=DTYPE, device=DEVICE)
        return v / (torch.norm(v, dim=dim, keepdim=True).clamp_min(eps))


@dataclass
class MemoryRecord:
    t: float
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    embedding: Optional[torch.Tensor] = None
    tags: List[str] = field(default_factory=list)
    decay_rate: float = 0.02
    recall_count: int = 0
    last_recall: float = 0.0

    def score(self, now: float) -> float:
        age = max(0.0, now - self.t)
        decay = math.exp(-self.decay_rate * age)
        recency = 1.0 / (1.0 + age * 0.15)
        return float(clamp01(self.importance * (0.62 * decay + 0.38 * recency)))


class VectorMemoryBank:
    def __init__(self, dim: int = 512, max_items: int = 256, name: str = 'bank'):
        self.dim = dim
        self.max_items = max_items
        self.name = name
        self.records: List[MemoryRecord] = []
        self.vectors: List[torch.Tensor] = []

    def _trim(self) -> None:
        overflow = max(0, len(self.records) - self.max_items)
        if overflow:
            self.records = self.records[overflow:]
            self.vectors = self.vectors[overflow:]

    def push(self, record: MemoryRecord, vector: Optional[torch.Tensor] = None) -> None:
        if vector is not None:
            vector = torch.as_tensor(vector, dtype=DTYPE, device=DEVICE).flatten()
            if vector.numel() != self.dim:
                vector = torch.nn.functional.pad(vector, (0, max(0, self.dim - vector.numel())))[: self.dim]
        self.records.append(record)
        self.vectors.append(vector if vector is not None else torch.zeros(self.dim, dtype=DTYPE, device=DEVICE))
        self._trim()

    def decay(self, now: float, rate: float = 0.015) -> None:
        for rec in self.records:
            rec.importance = float(max(0.01, rec.importance * math.exp(-rate * max(0.0, now - rec.last_recall))))

    def query(self, query: torch.Tensor, top_k: int = 5, min_importance: float = 0.08) -> List[Tuple[float, MemoryRecord]]:
        if not self.records:
            return []
        q = torch.as_tensor(query, dtype=DTYPE, device=DEVICE).flatten()
        if q.numel() != self.dim:
            q = torch.nn.functional.pad(q, (0, max(0, self.dim - q.numel())))[: self.dim]
        q = normalize(q)
        now = time.time()
        scored: List[Tuple[float, MemoryRecord]] = []
        for rec, vec in zip(self.records, self.vectors):
            if rec.importance < min_importance:
                continue
            sim = float(cosine_similarity(q, normalize(vec)).clamp(-1.0, 1.0)) if vec is not None else 0.0
            score = 0.68 * rec.score(now) + 0.32 * max(0.0, sim)
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, rec in scored[:top_k]:
            rec.recall_count += 1
            rec.last_recall = now
        return scored[:top_k]

    def summary(self) -> torch.Tensor:
        if not self.vectors:
            return torch.zeros(self.dim, dtype=DTYPE, device=DEVICE)
        stack = torch.stack(self.vectors, dim=0)
        weights = torch.tensor([max(0.01, r.importance) for r in self.records], dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        return (stack * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)


class DeniseMemory:
    """Memória comercial com short_bank, emo_bank e vetor de contexto."""

    def __init__(self, dim: int = 512, short_size: int = 96, emo_size: int = 48, long_size: int = 256):
        self.dim = dim
        self.short_bank = VectorMemoryBank(dim=dim, max_items=short_size, name='short_bank')
        self.emo_bank = VectorMemoryBank(dim=dim, max_items=emo_size, name='emo_bank')
        self.long_bank = VectorMemoryBank(dim=dim, max_items=long_size, name='long_bank')
        self.meta: Dict[str, Any] = {
            'last_text': '',
            'last_emotion': 'neutral',
            'last_state': 'idle',
            'last_user': '',
            'last_assistant': '',
        }
        self.timeline: List[MemoryRecord] = []

    def _text_vector(self, text: str) -> torch.Tensor:
        text = (text or '').strip()
        if not text:
            return torch.zeros(self.dim, dtype=DTYPE, device=DEVICE)
        codes = torch.tensor([ord(c) % 256 for c in text[: self.dim]], dtype=DTYPE, device=DEVICE)
        if codes.numel() < self.dim:
            codes = torch.nn.functional.pad(codes, (0, self.dim - codes.numel()))
        basis = torch.eye(self.dim, dtype=DTYPE, device=DEVICE)
        proj = project_179(codes, basis)
        return normalize(op_179(proj, torch.roll(proj, shifts=1)))

    def remember_input(self, *, t: float, text: str = '', emotion: str = 'neutral', state: str = 'idle', speaker: str = 'user', features: Optional[torch.Tensor] = None, importance: float = 0.5, tags: Optional[Sequence[str]] = None) -> MemoryRecord:
        vec = self._text_vector(text) if features is None else torch.as_tensor(features, dtype=DTYPE, device=DEVICE).flatten()
        if vec.numel() != self.dim:
            vec = torch.nn.functional.pad(vec, (0, max(0, self.dim - vec.numel())))[: self.dim]
        record = MemoryRecord(t=t, kind='input', payload={'text': text, 'emotion': emotion, 'state': state, 'speaker': speaker}, importance=clamp01(importance), embedding=vec, tags=list(tags or []), last_recall=t)
        self.short_bank.push(record, vec)
        self.long_bank.push(record, vec)
        if emotion != 'neutral':
            self.remember_emotion(t=t, emotion=emotion, state=state, features=vec, importance=min(1.0, importance + 0.15), tags=tags)
        if speaker == 'assistant':
            self.meta['last_assistant'] = text
        else:
            self.meta['last_user'] = text
        self.meta['last_text'] = text
        self.meta['last_emotion'] = emotion
        self.meta['last_state'] = state
        self.timeline.append(record)
        return record

    def remember_emotion(self, *, t: float, emotion: str, state: str = 'idle', features: Optional[torch.Tensor] = None, importance: float = 0.55, tags: Optional[Sequence[str]] = None) -> MemoryRecord:
        vec = torch.zeros(self.dim, dtype=DTYPE, device=DEVICE) if features is None else torch.as_tensor(features, dtype=DTYPE, device=DEVICE).flatten()
        if vec.numel() != self.dim:
            vec = torch.nn.functional.pad(vec, (0, max(0, self.dim - vec.numel())))[: self.dim]
        record = MemoryRecord(t=t, kind='emotion', payload={'emotion': emotion, 'state': state}, importance=clamp01(importance), embedding=vec, tags=list(tags or []), last_recall=t)
        self.emo_bank.push(record, vec)
        self.timeline.append(record)
        return record

    def remember_state(self, *, t: float, state: str, importance: float = 0.4, payload: Optional[Dict[str, Any]] = None, features: Optional[torch.Tensor] = None, tags: Optional[Sequence[str]] = None) -> MemoryRecord:
        vec = torch.zeros(self.dim, dtype=DTYPE, device=DEVICE) if features is None else torch.as_tensor(features, dtype=DTYPE, device=DEVICE).flatten()
        if vec.numel() != self.dim:
            vec = torch.nn.functional.pad(vec, (0, max(0, self.dim - vec.numel())))[: self.dim]
        record = MemoryRecord(t=t, kind='state', payload=dict(payload or {}, state=state), importance=clamp01(importance), embedding=vec, tags=list(tags or []), last_recall=t)
        self.short_bank.push(record, vec)
        self.timeline.append(record)
        return record

    def decay(self, now: float) -> None:
        self.short_bank.decay(now)
        self.emo_bank.decay(now, rate=0.02)
        self.long_bank.decay(now, rate=0.01)

    def recall(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[float, MemoryRecord]]:
        q = torch.as_tensor(query, dtype=DTYPE, device=DEVICE).flatten()
        if q.numel() != self.dim:
            q = torch.nn.functional.pad(q, (0, max(0, self.dim - q.numel())))[: self.dim]
        scored = self.short_bank.query(q, top_k=top_k) + self.emo_bank.query(q, top_k=top_k) + self.long_bank.query(q, top_k=top_k)
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def context_vector(self) -> torch.Tensor:
        return op_179(self.short_bank.summary(), self.emo_bank.summary())

    def digest(self) -> Dict[str, Any]:
        return {
            'last_text': self.meta['last_text'],
            'last_emotion': self.meta['last_emotion'],
            'last_state': self.meta['last_state'],
            'short_items': len(self.short_bank.records),
            'emo_items': len(self.emo_bank.records),
            'long_items': len(self.long_bank.records),
        }

    def serialize(self) -> Dict[str, Any]:
        def pack_bank(bank: VectorMemoryBank) -> Dict[str, Any]:
            return {
                'records': [asdict(r) for r in bank.records],
                'vectors': [v.detach().cpu().tolist() for v in bank.vectors],
            }
        return {
            'short_bank': pack_bank(self.short_bank),
            'emo_bank': pack_bank(self.emo_bank),
            'long_bank': pack_bank(self.long_bank),
            'meta': dict(self.meta),
        }

    def load_state(self, data: Dict[str, Any]) -> None:
        def unpack_bank(bank: VectorMemoryBank, src: Dict[str, Any]) -> None:
            bank.records.clear(); bank.vectors.clear()
            for rec_data, vec in zip(src.get('records', []), src.get('vectors', [])):
                rec = MemoryRecord(**rec_data)
                bank.push(rec, torch.tensor(vec, dtype=DTYPE, device=DEVICE))
        unpack_bank(self.short_bank, data.get('short_bank', {}))
        unpack_bank(self.emo_bank, data.get('emo_bank', {}))
        unpack_bank(self.long_bank, data.get('long_bank', {}))
        self.meta.update(data.get('meta', {}))


__all__ = ['MemoryRecord', 'VectorMemoryBank', 'DeniseMemory']
