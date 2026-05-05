from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

DTYPE = torch.float32
DEVICE = torch.device('cpu')


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return target
    return lerp(current, target, 1.0 - math.exp(-dt / tau))


def safe_hypot(x: float, y: float) -> float:
    return math.hypot(x, y)


# -----------------------------------------------------------------------------
# Workspace / spatial intelligence
# -----------------------------------------------------------------------------

@dataclass(init=False)
class SpaceOccupancyInput:
    avatar_x: float = 0.0
    avatar_y: float = 0.0
    avatar_scale: float = 1.0
    avatar_alpha: float = 1.0

    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    mouse_pressure: float = 0.0

    focus: bool = True
    click_pressure: float = 0.0
    fullscreen_pressure: float = 0.0
    silence_pressure: float = 0.0
    workspace_density: float = 0.0
    occupancy_pressure: float = 0.0
    idle_seconds: float = 0.0
    time_since_input: float = 0.0

    screen_width: float = 1920.0
    screen_height: float = 1080.0
    safe_margin: float = 24.0

    float_bias: float = 0.0
    attention_pressure: float = 0.0
    front_bias: float = 0.5
    mouse_hot_radius: float = 260.0
    cursor_locked: bool = False

    heatmap: Optional[Sequence[Sequence[float]]] = None
    occupied_rects: Optional[Sequence[Tuple[float, float, float, float]]] = None
    active_window_rect: Optional[Tuple[float, float, float, float]] = None
    dragged_window_rect: Optional[Tuple[float, float, float, float]] = None
    fullscreen_window_rect: Optional[Tuple[float, float, float, float]] = None
    z_order_hint: float = 0.0
    interaction_intensity: float = 0.0
    window_velocity_x: float = 0.0
    window_velocity_y: float = 0.0
    talking: bool = False
    window_count: int = 0
    drag_active: bool = False
    allow_return_to_last_safe: bool = True
    hold_lock_seconds: float = 0.85
    preferred_slot_index: int = -1
    return_bias: float = 0.35
    occupied_rect_bias: float = 0.60
    workspace_focus_bias: float = 0.25
    extra: Dict[str, Any] = field(default_factory=dict)

    _ALIASES = {
        'drag_window_rect': 'dragged_window_rect',
        'moving_window_rect': 'dragged_window_rect',
        'window_rect': 'active_window_rect',
        'window': 'active_window_rect',
        'fullscreen_rect': 'fullscreen_window_rect',
        'cursor_pressed': 'drag_active',
    }

    def __init__(self, **kwargs: Any):
        values = {f.name: getattr(type(self), f.name) for f in self.__dataclass_fields__.values() if f.init is False or True}  # type: ignore[attr-defined]
        # initialize defaults from class attributes
        for name in [
            'avatar_x','avatar_y','avatar_scale','avatar_alpha','mouse_x','mouse_y','mouse_dx','mouse_dy','mouse_speed','mouse_pressure',
            'focus','click_pressure','fullscreen_pressure','silence_pressure','workspace_density','occupancy_pressure','idle_seconds','time_since_input',
            'screen_width','screen_height','safe_margin','float_bias','attention_pressure','front_bias','mouse_hot_radius','cursor_locked',
            'heatmap','occupied_rects','active_window_rect','dragged_window_rect','fullscreen_window_rect','z_order_hint','interaction_intensity',
            'window_velocity_x','window_velocity_y','talking','window_count','drag_active','allow_return_to_last_safe','hold_lock_seconds',
            'preferred_slot_index','return_bias','occupied_rect_bias','workspace_focus_bias'
        ]:
            setattr(self, name, kwargs.pop(name, getattr(type(self), name)))
        self.extra = {}
        for k, v in list(kwargs.items()):
            alias = self._ALIASES.get(k, k)
            if hasattr(self, alias):
                setattr(self, alias, v)
            else:
                self.extra[k] = v

    def __repr__(self) -> str:
        return f"SpaceOccupancyInput(avatar=({self.avatar_x:.1f},{self.avatar_y:.1f}), mouse=({self.mouse_x:.1f},{self.mouse_y:.1f}), alpha={self.avatar_alpha:.2f})"


@dataclass
class SpaceOccupancyState:
    target_x: float = 0.0
    target_y: float = 0.0
    target_alpha: float = 1.0
    target_scale: float = 1.0
    visible: bool = True
    topmost: bool = False
    layer_mode: str = 'respect_space'
    target_z: float = 0.5
    anchor_x: float = 0.0
    anchor_y: float = 0.0
    current_slot_id: int = 0
    locked_slot_id: int = 0
    home_slot_id: int = 0
    last_safe_slot_id: int = 0
    last_safe_x: float = 0.0
    last_safe_y: float = 0.0
    locked_until: float = 0.0
    dodge_strength: float = 0.0
    slide_x: float = 0.0
    slide_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    spring_x: float = 0.0
    spring_y: float = 0.0
    spring_vx: float = 0.0
    spring_vy: float = 0.0
    reason: str = 'idle'
    float_phase: float = 0.0
    float_amplitude: float = 0.0
    smooth_time: float = 0.18
    hover_bias: float = 0.0
    overlap_score: float = 0.0
    visibility_score: float = 1.0
    escape_score: float = 0.0
    pulse: float = 0.0
    slot_stability: float = 0.0
    motion_curvature: float = 0.0
    path_curvature: float = 0.0
    alpha_floor: float = 0.12
    alpha_ceiling: float = 1.0
    transition_alpha: float = 1.0
    click_through: bool = False
    click_blocking: bool = True
    z_policy: str = 'normal'
    hold_reason: str = 'none'
    float_bias: float = 0.0
    expert_name: str = 'default'

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class MotionIntent:
    approach: float = 0.0
    avoid: float = 0.0
    focus: float = 0.0
    idle: float = 0.0
    respect: float = 0.0
    explore: float = 0.0
    overlay: float = 0.0
    hide: float = 0.0
    front: float = 0.0
    return_home: float = 0.0


@dataclass
class MotionTarget:
    x: float = 0.0
    y: float = 0.0
    alpha: float = 1.0
    scale: float = 1.0
    z: float = 0.5
    mode: str = 'respect_space'
    slot_id: int = 0
    click_through: bool = False
    topmost: bool = False
    reason: str = 'idle'


class ExpertRouter(nn.Module):
    def __init__(self, input_dim: int = 24, hidden: int = 64, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 8),
        )
        self.register_buffer('last_gate', torch.zeros(8, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        y = self.net(x)
        gate = torch.softmax(y.squeeze(0), dim=-1)
        self.last_gate.copy_(gate)
        return gate


class SpaceOccupancyTransformer(nn.Module):
    def __init__(self, input_dim: int = 24, hidden_dim: int = 96, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.router = ExpertRouter(input_dim=input_dim, hidden=hidden_dim, device=device, dtype=dtype)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 8)),
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 8)),
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 8)),
        ])
        self.register_buffer('last_output', torch.zeros(8, device=device, dtype=dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] < 24:
            pad = torch.zeros(*features.shape[:-1], 24 - features.shape[-1], device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > 24:
            features = features[..., :24]
        x = features.squeeze(0)
        gate = self.router(x)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        mixed = torch.sum(outputs * gate[: len(self.experts)].view(-1, 1), dim=0)
        self.last_output.copy_(mixed)
        return mixed

    def decode(self, out: torch.Tensor) -> Dict[str, float]:
        if out.ndim > 1:
            out = out.squeeze(0)
        return {
            'target_x': float(torch.tanh(out[0]).item()),
            'target_y': float(torch.tanh(out[1]).item()),
            'alpha': float(torch.sigmoid(out[2]).item()),
            'scale': float(0.72 + 0.78 * torch.sigmoid(out[3]).item()),
            'dodge': float(torch.sigmoid(out[4]).item()),
            'hover': float(torch.sigmoid(out[5]).item()),
            'z': float(torch.sigmoid(out[6]).item()),
            'visibility': float(torch.sigmoid(out[7]).item()),
        }


class OccupancyVisibilityController(nn.Module):
    def __init__(self, screen_width: float = 1920.0, screen_height: float = 1080.0, avatar_size: float = 900.0, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)
        self.avatar_size = float(avatar_size)
        self.device = device
        self.dtype = dtype
        self.model = SpaceOccupancyTransformer(device=device, dtype=dtype)
        self.state = SpaceOccupancyState(target_x=self.screen_width * 0.60, target_y=self.screen_height * 0.55)
        self.float_phase = random.random() * math.tau
        self._last_mouse = (self.screen_width * 0.5, self.screen_height * 0.5)
        self._last_target = (self.state.target_x, self.state.target_y)
        self._slot_bank: List[Tuple[float, float]] = []
        self.last_decision = self.state.as_dict()

    def reset(self) -> None:
        self.state = SpaceOccupancyState(target_x=self.screen_width * 0.60, target_y=self.screen_height * 0.55)
        self.float_phase = random.random() * math.tau
        self._slot_bank = []
        self.last_decision = self.state.as_dict()

    def _rect_area(self, rect: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = rect
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _rect_center(self, rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = rect
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _rect_overlap(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        return self._rect_area((x1, y1, x2, y2))

    def _heat_at(self, x: float, y: float, heatmap: Optional[Sequence[Sequence[float]]]) -> float:
        if not heatmap:
            return 0.0
        try:
            rows = len(heatmap)
            cols = len(heatmap[0]) if rows else 0
            if rows <= 0 or cols <= 0:
                return 0.0
            gx = int(clamp(x / max(1.0, self.screen_width) * cols, 0, cols - 1))
            gy = int(clamp(y / max(1.0, self.screen_height) * rows, 0, rows - 1))
            return clamp(float(heatmap[gy][gx]), 0.0, 1.0)
        except Exception:
            return 0.0

    def _window_overlap_score(self, x: float, y: float, inp: SpaceOccupancyInput) -> float:
        avatar = (x, y, x + self.avatar_size, y + self.avatar_size)
        rects: List[Tuple[float, float, float, float]] = []
        for r in (inp.occupied_rects or []):
            rects.append(tuple(map(float, r)))
        for r in [inp.active_window_rect, inp.dragged_window_rect, inp.fullscreen_window_rect]:
            if r is not None:
                rects.append(tuple(map(float, r)))
        if not rects:
            return 0.0
        overlap = sum(self._rect_overlap(avatar, r) for r in rects)
        denom = max(1.0, self.avatar_size * self.avatar_size)
        return clamp(overlap / denom, 0.0, 1.0)

    def _slot_bank_from_screen(self, inp: SpaceOccupancyInput) -> List[Tuple[float, float]]:
        w = max(1.0, inp.screen_width)
        h = max(1.0, inp.screen_height)
        s = self.avatar_size
        safe = max(8.0, inp.safe_margin)
        cx = w * 0.5 - s * 0.5
        cy = h * 0.5 - s * 0.5
        xslots = [safe, w * 0.5 - s * 0.5, max(safe, w - s - safe)]
        yslots = [safe, h * 0.42 - s * 0.5, max(safe, h - s - safe)]
        corners = [(safe, safe), (max(safe, w - s - safe), safe), (safe, max(safe, h - s - safe)), (max(safe, w - s - safe), max(safe, h - s - safe))]
        mids = [(cx, safe), (cx, max(safe, h - s - safe)), (safe, cy), (max(safe, w - s - safe), cy), (cx, cy)]
        front = [(w * 0.5 - s * 0.5, max(safe, h * 0.18 - s * 0.5))]
        rect_slots: List[Tuple[float, float]] = []
        for rect in (inp.occupied_rects or [])[:8]:
            rx, ry = self._rect_center(rect)
            for ox, oy in [(-0.34, -0.28), (0.34, -0.28), (-0.34, 0.28), (0.34, 0.28)]:
                rect_slots.append((rx + ox * w - s * 0.5, ry + oy * h - s * 0.5))
        slots = corners + mids + front + rect_slots + [(x, y) for x in xslots for y in yslots]
        out = []
        for x, y in slots:
            out.append((clamp(x, safe, max(safe, w - s - safe)), clamp(y, safe, max(safe, h - s - safe))))
        # stable dedupe
        dedup: List[Tuple[float, float]] = []
        for slot in out:
            if all(safe_hypot(slot[0] - p[0], slot[1] - p[1]) > 4.0 for p in dedup):
                dedup.append(slot)
        return dedup

    def _visibility_score(self, x: float, y: float, inp: SpaceOccupancyInput) -> float:
        center_x = x + self.avatar_size * 0.5
        center_y = y + self.avatar_size * 0.5
        dx = center_x - inp.mouse_x
        dy = center_y - inp.mouse_y
        dist = safe_hypot(dx, dy)
        mouse_pull = 1.0 / (1.0 + dist / max(1.0, inp.mouse_hot_radius))
        edge_x = min(center_x / max(1.0, self.screen_width), 1.0 - center_x / max(1.0, self.screen_width))
        edge_y = min(center_y / max(1.0, self.screen_height), 1.0 - center_y / max(1.0, self.screen_height))
        edge_bonus = 1.0 - clamp(edge_x + edge_y, 0.0, 1.0)
        heat = self._heat_at(center_x, center_y, inp.heatmap)
        overlap = self._window_overlap_score(x, y, inp)
        cursor_line = abs((center_y - inp.mouse_y) * inp.mouse_dx - (center_x - inp.mouse_x) * inp.mouse_dy)
        glide = 1.0 / (1.0 + cursor_line / 200000.0)
        front_bias = 1.0 - clamp(safe_hypot(center_x - self.screen_width * 0.5, center_y - self.screen_height * 0.42) / max(self.screen_width, self.screen_height), 0.0, 1.0)
        return (
            1.30 * (1.0 - heat)
            + 1.38 * (1.0 - overlap)
            + 0.92 * (1.0 - mouse_pull)
            + 0.22 * edge_bonus
            + 0.12 * inp.workspace_density
            + 0.10 * inp.occupancy_pressure
            + 0.10 * inp.fullscreen_pressure
            + 0.08 * inp.click_pressure
            + 0.08 * inp.silence_pressure
            + 0.06 * glide
            + 0.18 * front_bias * inp.front_bias
        )

    def _compute_features(self, inp: SpaceOccupancyInput) -> torch.Tensor:
        w = max(1.0, inp.screen_width)
        h = max(1.0, inp.screen_height)
        dx = (inp.mouse_x - inp.avatar_x) / w
        dy = (inp.mouse_y - inp.avatar_y) / h
        dist = safe_hypot(inp.mouse_x - inp.avatar_x, inp.mouse_y - inp.avatar_y)
        mouse_pressure = clamp(inp.mouse_pressure + max(0.0, 1.0 - dist / max(120.0, inp.avatar_scale * 0.55)), 0.0, 1.0)
        fullscreen = clamp(inp.fullscreen_pressure, 0.0, 1.0)
        silence = clamp(inp.silence_pressure, 0.0, 1.0)
        workspace = clamp(inp.workspace_density, 0.0, 1.0)
        occupancy = clamp(inp.occupancy_pressure, 0.0, 1.0)
        idle = clamp(inp.idle_seconds / 18.0, 0.0, 1.0)
        time_idle = clamp(inp.time_since_input / 18.0, 0.0, 1.0)
        focus = 1.0 if inp.focus else 0.0
        front_bias = clamp(inp.front_bias, 0.0, 1.0)
        attention = clamp(inp.attention_pressure, 0.0, 1.0)
        drag = 1.0 if inp.drag_active else 0.0
        dragged = 1.0 if inp.dragged_window_rect is not None else 0.0
        return torch.tensor([
            dx, dy, inp.mouse_dx / w, inp.mouse_dy / h, inp.mouse_speed / 1800.0, mouse_pressure,
            fullscreen, silence, workspace, occupancy, idle, focus,
            inp.avatar_scale / max(1.0, min(w, h)), inp.avatar_alpha, inp.float_bias, time_idle,
            front_bias, attention, 1.0 if inp.cursor_locked else 0.0, drag,
            dragged, inp.window_count / 10.0, inp.interaction_intensity, inp.z_order_hint,
        ], device=self.device, dtype=self.dtype)

    def _target_from_score(self, inp: SpaceOccupancyInput, scored: List[Tuple[float, float, float]], decoded: Dict[str, float], active_pressure: float, reason: str) -> MotionTarget:
        if not scored:
            scored = [(0.0, inp.avatar_x, inp.avatar_y)]
        best_score, best_x, best_y = max(scored, key=lambda item: item[0])
        frontness = clamp(decoded['visibility'] + 0.48 * inp.front_bias + 0.25 * (1.0 - active_pressure), 0.0, 1.0)
        if reason in {'idle_front', 'fullscreen_front'}:
            best_x = clamp(self.screen_width * 0.5 - self.avatar_size * 0.5, inp.safe_margin, max(inp.safe_margin, self.screen_width - self.avatar_size - inp.safe_margin))
            best_y = clamp(self.screen_height * 0.18 - self.avatar_size * 0.5, inp.safe_margin, max(inp.safe_margin, self.screen_height - self.avatar_size - inp.safe_margin))
            best_score = max(best_score, 1.0 + 0.3 * frontness)
        return MotionTarget(
            x=best_x,
            y=best_y,
            alpha=decoded['alpha'],
            scale=decoded['scale'],
            z=decoded['z'],
            mode='overlay_minimal' if reason == 'fullscreen_front' else ('float_idle' if reason == 'idle_front' else 'respect_space'),
            slot_id=0,
            click_through=False,
            topmost=reason in {'idle_front', 'fullscreen_front'},
            reason=reason,
        )

    def step(self, inp: SpaceOccupancyInput, dt: float = 0.016) -> SpaceOccupancyState:
        dt = max(1e-4, float(dt))
        features = self._compute_features(inp)
        raw = self.model(features)
        decoded = self.model.decode(raw)

        mouse_pressure = clamp(inp.mouse_pressure + max(0.0, 1.0 - safe_hypot(inp.mouse_x - inp.avatar_x, inp.mouse_y - inp.avatar_y) / max(1.0, inp.avatar_scale * 0.70)), 0.0, 1.0)
        occupancy_pressure = clamp(inp.occupancy_pressure + inp.workspace_density * 0.55, 0.0, 1.0)
        active_pressure = clamp(0.42 * mouse_pressure + 0.24 * inp.click_pressure + 0.18 * occupancy_pressure + 0.18 * inp.fullscreen_pressure + 0.12 * inp.silence_pressure, 0.0, 1.0)
        if not inp.focus:
            active_pressure = min(1.0, active_pressure + 0.12)
        if inp.idle_seconds > 8.0:
            active_pressure = max(0.0, active_pressure - 0.16)

        slots = self._slot_bank_from_screen(inp)
        self._slot_bank = slots
        scored = [(self._visibility_score(x, y, inp), x, y) for x, y in slots]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_x, best_y = scored[0]

        # Expert routing: combines factors into named policy without brittle cascades.
        intent = MotionIntent(
            approach=clamp(0.15 + 0.60 * inp.attention_pressure + 0.20 * inp.talking, 0.0, 1.0),
            avoid=clamp(0.10 + 0.55 * active_pressure + 0.20 * inp.drag_active, 0.0, 1.0),
            focus=clamp(0.18 + 0.50 * inp.focus + 0.22 * (1.0 - inp.silence_pressure), 0.0, 1.0),
            idle=clamp(0.10 + 0.70 * inp.idle_seconds / 18.0, 0.0, 1.0),
            respect=clamp(0.30 + 0.60 * occupancy_pressure, 0.0, 1.0),
            explore=clamp(0.15 + 0.40 * inp.workspace_density, 0.0, 1.0),
            overlay=clamp(0.15 + 0.60 * inp.fullscreen_pressure, 0.0, 1.0),
            hide=clamp(0.10 + 0.55 * mouse_pressure, 0.0, 1.0),
            front=clamp(0.12 + 0.50 * (1.0 - active_pressure) + 0.25 * inp.front_bias, 0.0, 1.0),
            return_home=clamp(0.20 + 0.50 * inp.allow_return_to_last_safe, 0.0, 1.0),
        )
        gate = torch.softmax(torch.tensor([intent.avoid, intent.overlay, intent.front, intent.idle, intent.respect, intent.focus, intent.hide, intent.return_home], device=self.device, dtype=self.dtype), dim=0)
        expert = int(torch.argmax(gate).item())
        expert_name = ['avoid', 'overlay', 'front', 'idle', 'respect', 'focus', 'hide', 'return_home'][expert]

        reason = 'wander'
        if active_pressure > 0.72 and inp.dragged_window_rect is not None:
            reason = 'drag_escape'
        elif inp.fullscreen_pressure > 0.55 and inp.idle_seconds > 2.5:
            reason = 'fullscreen_front'
        elif mouse_pressure > 0.60:
            reason = 'mouse_avoid'
        elif inp.click_pressure > 0.45:
            reason = 'click_escape'
        elif inp.workspace_density > 0.60 and inp.idle_seconds > 5.5:
            reason = 'idle_front'
        elif inp.workspace_density > 0.55:
            reason = 'workspace_compact'
        elif inp.silence_pressure > 0.50 and inp.idle_seconds > 4.0:
            reason = 'quiet_float'
        elif inp.idle_seconds > 12.0:
            reason = 'float'

        if reason in {'mouse_avoid', 'click_escape', 'drag_escape'} and len(scored) > 1:
            for cand_score, cand_x, cand_y in scored:
                if cand_score >= best_score * 0.94:
                    best_x, best_y, best_score = cand_x, cand_y, cand_score
                    break

        # Path planning: curved escape with a stable anchor and fourier idle drift.
        away_x = inp.avatar_x - inp.mouse_x
        away_y = inp.avatar_y - inp.mouse_y
        norm = safe_hypot(away_x, away_y) or 1.0
        perp_x = -away_y / norm
        perp_y = away_x / norm
        curvature = clamp(0.10 + 0.38 * active_pressure + 0.15 * decoded['dodge'], 0.08, 0.56)
        anchor_x = lerp(inp.avatar_x, best_x, 0.48) + perp_x * curvature * 80.0
        anchor_y = lerp(inp.avatar_y, best_y, 0.48) + perp_y * curvature * 80.0
        self._last_target = (
            clamp(anchor_x, inp.safe_margin, max(inp.safe_margin, self.screen_width - self.avatar_size - inp.safe_margin)),
            clamp(anchor_y, inp.safe_margin, max(inp.safe_margin, self.screen_height - self.avatar_size - inp.safe_margin)),
        )

        float_strength = clamp(max(inp.float_bias, (inp.idle_seconds - 3.0) / 10.0 if inp.idle_seconds > 3.0 else 0.0), 0.0, 1.0)
        self.float_phase += dt * (0.65 + 0.60 * float_strength)
        fourier_dx = 16.0 * math.sin(self.float_phase) + 8.0 * math.sin(self.float_phase * 2.17 + 0.3)
        fourier_dy = 10.0 * math.cos(self.float_phase * 0.82) + 6.0 * math.sin(self.float_phase * 1.41 + 1.2)
        if inp.mouse_speed > 700.0 and mouse_pressure > 0.25:
            fourier_dx -= perp_x * 18.0
            fourier_dy -= perp_y * 18.0

        target_x = clamp(best_x + fourier_dx, inp.safe_margin, max(inp.safe_margin, self.screen_width - self.avatar_size - inp.safe_margin))
        target_y = clamp(best_y + fourier_dy, inp.safe_margin, max(inp.safe_margin, self.screen_height - self.avatar_size - inp.safe_margin))

        target_alpha = decoded['alpha']
        target_alpha *= 1.0 - 0.55 * active_pressure
        if reason in {'idle_front', 'fullscreen_front'}:
            target_alpha = max(target_alpha, 0.90)
        elif reason == 'mouse_avoid':
            target_alpha *= 0.74
        elif reason in {'click_escape', 'drag_escape'}:
            target_alpha *= 0.80
        elif reason in {'quiet_float', 'float'}:
            target_alpha *= 0.94
        target_alpha = clamp(target_alpha, 0.04 if active_pressure > 0.65 else 0.16, 1.0)

        target_scale = clamp(decoded['scale'] * (0.92 + 0.12 * (1.0 - active_pressure)) * (0.96 + 0.10 * decoded['visibility']), 0.68, 1.55)
        target_scale += 0.03 * math.sin(self.float_phase * 1.6)
        target_scale = clamp(target_scale, 0.68, 1.55)

        slide_x = clamp((target_x - inp.avatar_x) / max(1.0, inp.screen_width) * 1.85, -0.70, 0.70)
        slide_y = clamp((target_y - inp.avatar_y) / max(1.0, inp.screen_height) * 1.85, -0.70, 0.70)
        if reason in {'mouse_avoid', 'click_escape', 'drag_escape'}:
            slide_x += (away_x / norm) * 0.18 * active_pressure
            slide_y += (away_y / norm) * 0.18 * active_pressure
        elif reason in {'idle_front', 'fullscreen_front'}:
            slide_x *= 0.6
            slide_y *= 0.6

        smooth_time = 0.11 if reason in {'mouse_avoid', 'click_escape', 'drag_escape'} else (0.18 if reason in {'idle_front', 'fullscreen_front'} else 0.22)
        path_tau = 0.14 if reason in {'mouse_avoid', 'click_escape', 'drag_escape'} else 0.28
        self.state.target_x = exp_smooth(self.state.target_x, target_x, dt, smooth_time)
        self.state.target_y = exp_smooth(self.state.target_y, target_y, dt, smooth_time)
        self.state.target_alpha = exp_smooth(self.state.target_alpha, target_alpha, dt, 0.12 if target_alpha < self.state.target_alpha else 0.24)
        self.state.target_scale = exp_smooth(self.state.target_scale, target_scale, dt, 0.16)
        self.state.visible = self.state.target_alpha > 0.12 or reason in {'idle_front', 'fullscreen_front'}
        self.state.topmost = reason in {'idle_front', 'fullscreen_front'}
        self.state.layer_mode = 'overlay_minimal' if reason == 'fullscreen_front' else ('float_idle' if reason == 'idle_front' else 'respect_space')
        self.state.dodge_strength = exp_smooth(self.state.dodge_strength, active_pressure, dt, 0.18)
        self.state.slide_x = exp_smooth(self.state.slide_x, slide_x, dt, 0.20)
        self.state.slide_y = exp_smooth(self.state.slide_y, slide_y, dt, 0.20)
        self.state.velocity_x = exp_smooth(self.state.velocity_x, (self.state.target_x - inp.avatar_x) / max(dt, 1e-3), dt, 0.22)
        self.state.velocity_y = exp_smooth(self.state.velocity_y, (self.state.target_y - inp.avatar_y) / max(dt, 1e-3), dt, 0.22)
        self.state.reason = reason
        self.state.smooth_time = smooth_time
        self.state.hover_bias = exp_smooth(self.state.hover_bias, float(decoded['hover']), dt, 0.28)
        self.state.overlap_score = exp_smooth(self.state.overlap_score, 1.0 - best_score, dt, path_tau)
        self.state.visibility_score = exp_smooth(self.state.visibility_score, best_score, dt, path_tau)
        self.state.motion_curvature = exp_smooth(self.state.motion_curvature, curvature, dt, 0.22)
        self.state.path_curvature = self.state.motion_curvature
        self.state.anchor_x = exp_smooth(self.state.anchor_x, self._last_target[0], dt, 0.20)
        self.state.anchor_y = exp_smooth(self.state.anchor_y, self._last_target[1], dt, 0.20)
        self.state.float_phase = self.float_phase
        self.state.float_amplitude = exp_smooth(self.state.float_amplitude, float_strength, dt, 0.25)
        self.state.pulse = exp_smooth(self.state.pulse, decoded['hover'] * (1.0 - active_pressure), dt, 0.22)
        self.state.slot_stability = exp_smooth(self.state.slot_stability, 1.0 / (1.0 + abs(best_score - (scored[1][0] if len(scored) > 1 else best_score))), dt, 0.35)
        self.state.alpha_floor = 0.04 if active_pressure > 0.72 else 0.12
        self.state.alpha_ceiling = 1.0
        self.state.click_through = bool(inp.cursor_locked) or (active_pressure > 0.92 and reason in {'mouse_avoid', 'click_escape', 'drag_escape'})
        self.state.click_blocking = not self.state.click_through
        self.state.z_policy = 'always_on_top' if self.state.topmost else ('above_overlay' if reason in {'fullscreen_front', 'idle_front'} else 'normal')
        self.state.hold_reason = 'locked' if inp.drag_active else 'free'
        self.state.float_bias = inp.float_bias
        self.state.expert_name = expert_name
        self.last_decision = self.state.as_dict()
        return self.state


# -----------------------------------------------------------------------------
# Animation / body / face / eye / mouth transformers
# -----------------------------------------------------------------------------

class IntentTransformer(nn.Module):
    def __init__(self, input_dim: int = 32, hidden: int = 512, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 10),
        )
        self.register_buffer('last_intent', torch.zeros(10, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> MotionIntent:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        y = torch.softmax(self.net(x).squeeze(0), dim=-1)
        self.last_intent.copy_(y)
        vals = y.tolist()
        return MotionIntent(*vals[:10])


class MotionPlanningTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.router = ExpertRouter(24, 64, device=device, dtype=dtype)
        self.register_buffer('last_target', torch.zeros(4, device=device, dtype=dtype))

    def forward(self, intent: MotionIntent, workspace: SpaceOccupancyState, screen_w: float, screen_h: float) -> MotionTarget:
        intent_vec = torch.tensor([intent.approach, intent.avoid, intent.focus, intent.idle, intent.respect, intent.explore, intent.overlay, intent.hide, intent.front, intent.return_home,
                                   workspace.visibility_score, workspace.overlap_score, workspace.dodge_strength, workspace.float_amplitude, workspace.motion_curvature, workspace.target_alpha,
                                   screen_w / 1920.0, screen_h / 1080.0, workspace.target_x / max(1.0, screen_w), workspace.target_y / max(1.0, screen_h),
                                   workspace.topmost * 1.0, workspace.click_through * 1.0, workspace.path_curvature, workspace.pulse], device=self.last_target.device, dtype=self.last_target.dtype)
        gate = self.router(intent_vec)
        mode_idx = int(torch.argmax(gate).item())
        mode = ['approach', 'avoid', 'focus', 'idle', 'respect', 'overlay', 'hide', 'return_home'][mode_idx]
        alpha = clamp(1.0 - 0.45 * intent.avoid + 0.25 * intent.front - 0.20 * intent.hide + 0.15 * intent.overlay, 0.05, 1.0)
        scale = clamp(0.72 + 0.22 * intent.focus - 0.16 * intent.hide + 0.12 * intent.overlay, 0.65, 1.6)
        click_through = intent.avoid > 0.7 or intent.hide > 0.8
        return MotionTarget(x=workspace.target_x, y=workspace.target_y, alpha=alpha, scale=scale, z=workspace.target_z, mode=mode, slot_id=workspace.current_slot_id, click_through=click_through, topmost=workspace.topmost, reason=workspace.reason)


class TrajectoryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = 0.0
        self.spring_vx = 0.0
        self.spring_vy = 0.0

    def step(self, current: Tuple[float, float], target: Tuple[float, float], dt: float, stiffness: float = 8.0, damping: float = 0.86) -> Tuple[float, float]:
        cx, cy = current
        tx, ty = target
        self.spring_vx += (tx - cx) * stiffness * dt
        self.spring_vy += (ty - cy) * stiffness * dt
        self.spring_vx *= damping
        self.spring_vy *= damping
        nx = cx + self.spring_vx
        ny = cy + self.spring_vy
        return nx, ny

    def eased(self, progress: float) -> float:
        return smoothstep(0.0, 1.0, progress)


class PhysicsTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = torch.zeros(2)

    def step(self, position: Tuple[float, float], target: Tuple[float, float], dt: float, stiffness: float = 6.0, damping: float = 0.85) -> Tuple[float, float]:
        px, py = position
        tx, ty = target
        vx, vy = self.velocity.tolist()
        vx += (tx - px) * stiffness * dt
        vy += (ty - py) * stiffness * dt
        vx *= damping
        vy *= damping
        self.velocity[0] = vx
        self.velocity[1] = vy
        return px + vx, py + vy


class SecondaryMotionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update(self, body_rotation: float, speed: float, dt: float) -> float:
        self.phase += dt * (0.9 + 0.5 * clamp(speed, 0.0, 1.0))
        return body_rotation * 0.62 + math.sin(self.phase) * 1.2 * clamp(speed, 0.0, 1.0)


class BodyRotationTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

    def update(self, intent: MotionIntent, dt: float) -> Tuple[float, float, float]:
        target_yaw = 4.0 * (intent.approach - intent.avoid) + 2.0 * intent.front - 1.2 * intent.hide
        target_pitch = -2.0 * intent.focus + 1.6 * intent.idle
        target_roll = 6.0 * intent.overlay - 4.0 * intent.hide + 2.0 * intent.return_home
        self.yaw = exp_smooth(self.yaw, target_yaw, dt, 0.14)
        self.pitch = exp_smooth(self.pitch, target_pitch, dt, 0.16)
        self.roll = exp_smooth(self.roll, target_roll, dt, 0.18)
        return self.yaw, self.pitch, self.roll


class SphericalEyeRollTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eye_x = 0.0
        self.eye_y = 0.0
        self.left_compress = 1.0
        self.right_compress = 1.0

    def update(self, target_xy: Tuple[float, float], eye_center: Tuple[float, float], radius: float = 11.0, dt: float = 0.016) -> Tuple[float, float, float, float]:
        tx, ty = target_xy
        ex, ey = eye_center
        dx = tx - ex
        dy = ty - ey
        angle = math.atan2(dy, dx)
        distance = safe_hypot(dx, dy)
        d = min(distance * 0.1, radius)
        px = math.cos(angle) * d
        py = math.sin(angle) * d * 0.82
        self.eye_x = exp_smooth(self.eye_x, px, dt, 0.07)
        self.eye_y = exp_smooth(self.eye_y, py, dt, 0.07)
        compress = 1.0 - clamp(distance / 800.0, 0.0, 0.25)
        self.left_compress = exp_smooth(self.left_compress, compress, dt, 0.12)
        self.right_compress = exp_smooth(self.right_compress, compress, dt, 0.12)
        return self.eye_x, self.eye_y, self.left_compress, self.right_compress


class HeadEyeCoordinationTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_x = 0.0
        self.head_y = 0.0
        self.head_roll = 0.0
        self.delay = 0.10

    def update(self, eye_target: Tuple[float, float], head_target: Tuple[float, float, float], dt: float) -> Tuple[float, float, float]:
        ex, ey = eye_target
        hx, hy, hr = head_target
        self.head_x = exp_smooth(self.head_x, hx + ex * 0.15, dt, 0.24)
        self.head_y = exp_smooth(self.head_y, hy + ey * 0.15, dt, 0.24)
        self.head_roll = exp_smooth(self.head_roll, hr, dt, 0.26)
        return self.head_x, self.head_y, self.head_roll


class TorsionalRotationTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.roll = 0.0

    def update(self, uncertainty: float, dt: float) -> float:
        target = clamp(uncertainty, 0.0, 1.0) * 8.0 - 2.0
        self.roll = exp_smooth(self.roll, target, dt, 0.20)
        return self.roll


class ConvergenceTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.convergence = 0.0

    def update(self, mouse_distance_to_face: float, dt: float) -> float:
        target = clamp(1.0 - mouse_distance_to_face / 450.0, 0.0, 1.0)
        self.convergence = exp_smooth(self.convergence, target, dt, 0.08)
        return self.convergence


class GazeSaccadeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = random.random() * math.tau
        self.seed = random.random() * 100.0

    def update(self, base_target: Tuple[float, float], t: float, reading_bias: float = 0.0) -> Tuple[float, float]:
        bx, by = base_target
        self.phase += 0.016 * (2.4 + 1.3 * reading_bias)
        arc = math.sin(self.phase * 1.8) * 7.0
        wobble = math.sin(self.phase * 2.7 + self.seed) * 2.4
        return bx + arc + wobble, by + arc * 0.54 - wobble * 0.35


class EmotionalBlinkTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.timer = 0.0
        self.next_blink = 3.2
        self.amount = 0.0
        self.seed = random.random() * 100.0

    def update(self, tension: float, curiosity: float, fatigue: float, dt: float) -> float:
        self.timer += dt
        interval = clamp(3.5 - 1.8 * tension + 1.2 * curiosity + 2.2 * fatigue, 1.2, 7.0)
        if self.amount <= 0.0 and self.timer >= self.next_blink:
            self.amount = 1.0
            self.timer = 0.0
            self.next_blink = interval
        if self.amount > 0.0:
            decay = 10.0 + 2.0 * fatigue
            self.amount = max(0.0, self.amount - dt * decay)
        return self.amount


class AsymmetricExpressionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = 0.0
        self.right = 0.0

    def update(self, valence: float, tension: float, dt: float) -> Tuple[float, float]:
        target = clamp(valence * 0.7 - tension * 0.35, -1.0, 1.0)
        self.left = exp_smooth(self.left, target * 1.00, dt, 0.16)
        self.right = exp_smooth(self.right, target * 0.82, dt, 0.16)
        return self.left, self.right


class OromandibularKinematicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.jaw = 0.05
        self.rounding = 0.0
        self.spread = 0.1
        self.upper_lip = 0.0
        self.lower_lip = 0.0
        self.tongue = 0.0
        self.cheek = 0.0
        self.saliva = 0.0
        self.lip_bite = 0.0
        self.lips_together = 0.15
        self.lip_corner_up = 0.0
        self.lip_corner_down = 0.0
        self.jaw_clench = 0.0
        self.tongue_out = 0.0
        self.tongue_tip_interdental = 0.0
        self.upper_lip_raise = 0.0
        self.lower_lip_depress = 0.0
        self.cheek_puff = 0.0
        self.cheek_suck = 0.0
        self.mouth_corner_stretch = 0.0
        self.breath_phase = 0.0

    def update(self, viseme: Optional[Dict[str, float]], emotion: Optional[Dict[str, float]], audio_energy: float, dt: float) -> Dict[str, float]:
        vis = viseme or {}
        emo = emotion or {}
        intensity = clamp(float(emo.get('intensity', 0.35)), 0.0, 1.0)
        tension = clamp(float(emo.get('tension', 0.12)), 0.0, 1.0)
        valence = clamp(float(emo.get('valence', 0.0)), -1.0, 1.0)
        curiosity = clamp(float(emo.get('curiosity', 0.35)), 0.0, 1.0)
        voiced = clamp(audio_energy, 0.0, 1.0)
        mouth_open = clamp(float(vis.get('mouth_open', 0.05)), 0.0, 1.0)
        mouth_wide = clamp(float(vis.get('mouth_wide', 0.1)), 0.0, 1.0)
        lip_round = clamp(float(vis.get('lip_round', 0.0)), 0.0, 1.0)
        jaw_drop = clamp(float(vis.get('jaw_drop', 0.03)), 0.0, 1.0)
        teeth = clamp(float(vis.get('teeth', 0.0)), 0.0, 1.0)
        smile = clamp(float(vis.get('smile', 0.0)), -1.0, 1.0)
        tongue_up = clamp(float(vis.get('tongue_up', 0.0)), 0.0, 1.0)
        cheek = clamp(float(vis.get('cheek', 0.0)), 0.0, 1.0)
        lips_together = clamp(float(vis.get('lips_together', 1.0 - mouth_open)), 0.0, 1.0)
        lip_corner_up = clamp(float(vis.get('lip_corner_up', max(0.0, smile))), -1.0, 1.0)
        lip_corner_down = clamp(float(vis.get('lip_corner_down', max(0.0, -smile))), 0.0, 1.0)
        jaw_clench = clamp(float(vis.get('jaw_clench', tension)), 0.0, 1.0)
        tongue_out = clamp(float(vis.get('tongue_out', 0.0)), 0.0, 1.0)
        tongue_tip_interdental = clamp(float(vis.get('tongue_tip_interdental', 0.0)), 0.0, 1.0)
        upper_lip_raise = clamp(float(vis.get('upper_lip_raise', 0.0)), 0.0, 1.0)
        lower_lip_depress = clamp(float(vis.get('lower_lip_depress', 0.0)), 0.0, 1.0)
        cheek_puff = clamp(float(vis.get('cheek_puff', 0.0)), 0.0, 1.0)
        cheek_suck = clamp(float(vis.get('cheek_suck', 0.0)), 0.0, 1.0)
        mouth_corner_stretch = clamp(float(vis.get('mouth_corner_stretch', mouth_wide)), 0.0, 1.0)

        self.breath_phase += dt * (0.9 + 0.5 * intensity)
        breath = 0.03 + 0.02 * math.sin(self.breath_phase * math.tau)
        bite = 0.08 * clamp(tension * 0.7 + max(0.0, -valence) * 0.2 + jaw_clench * 0.45, 0.0, 1.0)
        saliva = 0.04 * clamp(1.0 - voiced + tension * 0.5 + lips_together * 0.15, 0.0, 1.0)

        base_open = mouth_open * 0.46 + jaw_drop * 0.40 + voiced * 0.12 + breath
        base_open *= 1.0 - 0.38 * lips_together
        self.jaw = exp_smooth(self.jaw, clamp(base_open, 0.0, 1.0), dt, 0.05)
        self.rounding = exp_smooth(self.rounding, clamp(lip_round + 0.12 * mouth_wide + 0.08 * tongue_out, 0.0, 1.0), dt, 0.08)
        self.spread = exp_smooth(self.spread, clamp(0.42 * mouth_wide + 0.20 * max(0.0, smile) + 0.18 * teeth + 0.10 * mouth_corner_stretch, 0.0, 1.0), dt, 0.08)
        self.upper_lip = exp_smooth(self.upper_lip, clamp(0.18 * smile + 0.16 * tension + 0.18 * upper_lip_raise, -1.0, 1.0), dt, 0.10)
        self.lower_lip = exp_smooth(self.lower_lip, clamp(0.20 * smile + 0.14 * jaw_drop + 0.16 * lower_lip_depress, -1.0, 1.0), dt, 0.10)
        self.tongue = exp_smooth(self.tongue, clamp(0.42 * tongue_up + 0.24 * voiced + 0.18 * tongue_out, 0.0, 1.0), dt, 0.09)
        self.cheek = exp_smooth(self.cheek, clamp(cheek + 0.12 * max(0.0, smile), 0.0, 1.0), dt, 0.12)
        self.saliva = exp_smooth(self.saliva, saliva, dt, 0.18)
        self.lip_bite = exp_smooth(self.lip_bite, bite, dt, 0.18)
        self.lips_together = exp_smooth(self.lips_together, lips_together, dt, 0.10)
        self.lip_corner_up = exp_smooth(self.lip_corner_up, clamp(lip_corner_up + 0.08 * valence, -1.0, 1.0), dt, 0.12)
        self.lip_corner_down = exp_smooth(self.lip_corner_down, clamp(lip_corner_down + 0.10 * tension, 0.0, 1.0), dt, 0.12)
        self.jaw_clench = exp_smooth(self.jaw_clench, clamp(jaw_clench + bite, 0.0, 1.0), dt, 0.12)
        self.tongue_out = exp_smooth(self.tongue_out, clamp(tongue_out + 0.22 * max(0.0, voiced - mouth_open), 0.0, 1.0), dt, 0.11)
        self.tongue_tip_interdental = exp_smooth(self.tongue_tip_interdental, clamp(tongue_tip_interdental + 0.16 * max(0.0, curiosity - 0.35), 0.0, 1.0), dt, 0.11)
        self.upper_lip_raise = exp_smooth(self.upper_lip_raise, clamp(upper_lip_raise + 0.14 * max(0.0, smile), 0.0, 1.0), dt, 0.11)
        self.lower_lip_depress = exp_smooth(self.lower_lip_depress, clamp(lower_lip_depress + 0.14 * jaw_drop, 0.0, 1.0), dt, 0.11)
        self.cheek_puff = exp_smooth(self.cheek_puff, clamp(cheek_puff + 0.12 * lip_round + 0.08 * voiced, 0.0, 1.0), dt, 0.11)
        self.cheek_suck = exp_smooth(self.cheek_suck, clamp(cheek_suck + 0.12 * max(0.0, smile) + 0.05 * mouth_wide, 0.0, 1.0), dt, 0.11)
        self.mouth_corner_stretch = exp_smooth(self.mouth_corner_stretch, clamp(mouth_corner_stretch, 0.0, 1.0), dt, 0.12)

        return {
            'jaw_open': self.jaw,
            'lip_round': self.rounding,
            'lip_spread': self.spread,
            'upper_lip': self.upper_lip,
            'lower_lip': self.lower_lip,
            'tongue': self.tongue,
            'cheek': self.cheek,
            'saliva': self.saliva,
            'lip_bite': self.lip_bite,
            'lips_together': self.lips_together,
            'lip_corner_up': self.lip_corner_up,
            'lip_corner_down': self.lip_corner_down,
            'jaw_clench': self.jaw_clench,
            'tongue_out': self.tongue_out,
            'tongue_tip_interdental': self.tongue_tip_interdental,
            'upper_lip_raise': self.upper_lip_raise,
            'lower_lip_depress': self.lower_lip_depress,
            'cheek_puff': self.cheek_puff,
            'cheek_suck': self.cheek_suck,
            'mouth_corner_stretch': self.mouth_corner_stretch,
        }


class AdvancedAnimationOrchestrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.intent = IntentTransformer()
        self.plan = MotionPlanningTransformer()
        self.trajectory = TrajectoryTransformer()
        self.physics = PhysicsTransformer()
        self.secondary = SecondaryMotionTransformer()
        self.body_rot = BodyRotationTransformer()
        self.eye_roll = SphericalEyeRollTransformer()
        self.head_coord = HeadEyeCoordinationTransformer()
        self.torsion = TorsionalRotationTransformer()
        self.convergence = ConvergenceTransformer()
        self.blink = EmotionalBlinkTransformer()
        self.smirk = AsymmetricExpressionTransformer()
        self.mouth = OromandibularKinematicTransformer()

    def build_workspace_vector(self, perception: Dict[str, float]) -> torch.Tensor:
        keys = [
            'valence', 'arousal', 'dominance', 'attention', 'curiosity', 'confidence', 'tension', 'fatigue',
            'energy', 'speech', 'audio_energy', 'mouse_speed', 'mouse_distance', 'idle_seconds',
            'fullscreen_pressure', 'workspace_density', 'x', 'y', 'screen_w', 'screen_h',
            'click_pressure', 'drag_pressure', 'cursor_locked', 'hover_bias',
            'overlap_score', 'visibility_score', 'target_alpha', 'float_bias', 'front_bias',
            'interaction_intensity', 'window_velocity_x', 'window_velocity_y'
        ]
        vec = [float(perception.get(k, 0.0)) for k in keys]
        return torch.tensor(vec[:32], dtype=DTYPE)

    def update(self, perception: Dict[str, float], viseme: Optional[Dict[str, float]] = None, dt: float = 0.016) -> Dict[str, Any]:
        intent = self.intent(self.build_workspace_vector(perception))
        workspace = perception.get('space_state')
        if not isinstance(workspace, SpaceOccupancyState):
            workspace = SpaceOccupancyState(target_x=float(perception.get('x', 0.0)), target_y=float(perception.get('y', 0.0)))
        target = self.plan(intent, workspace, float(perception.get('screen_w', 1920.0)), float(perception.get('screen_h', 1080.0)))
        pos = self.trajectory.step((float(perception.get('x', 0.0)), float(perception.get('y', 0.0))), (target.x, target.y), dt)
        physics_position = self.physics.step((float(perception.get('x', 0.0)), float(perception.get('y', 0.0))), (target.x, target.y), dt)
        body_rot = self.body_rot.update(intent, dt)
        eye = self.eye_roll.update((float(perception.get('mouse_x', 0.0)), float(perception.get('mouse_y', 0.0))), (float(perception.get('eye_x', 0.0)), float(perception.get('eye_y', 0.0))), dt=dt)
        head = self.head_coord.update((eye[0], eye[1]), body_rot, dt)
        torsion = self.torsion.update(float(perception.get('uncertainty', 0.0)), dt)
        convergence = self.convergence.update(float(perception.get('mouse_distance', 999.0)), dt)
        blink = self.blink.update(float(perception.get('tension', 0.12)), float(perception.get('curiosity', 0.3)), float(perception.get('fatigue', 0.1)), dt)
        smirk = self.smirk.update(float(perception.get('valence', 0.0)), float(perception.get('tension', 0.0)), dt)
        mouth = self.mouth.update(viseme, perception, float(perception.get('audio_energy', 0.0)), dt)
        return {
            'position': pos,
            'physics_position': physics_position,
            'body_rotation': body_rot,
            'head_rotation': head,
            'torsion': torsion,
            'convergence': convergence,
            'blink': blink,
            'smirk': smirk,
            'mouth': mouth,
            'intent': intent,
            'motion_target': target,
            'workspace_gate': self.plan.router.last_gate.detach().cpu().tolist(),
        }


# -----------------------------------------------------------------------------
# Integration helpers
# -----------------------------------------------------------------------------

def build_advanced_transformers() -> AdvancedAnimationOrchestrator:
    return AdvancedAnimationOrchestrator()


def apply_motion_stack(motion: Dict[str, Any], stack: Optional[AdvancedAnimationOrchestrator] = None, perception: Optional[Dict[str, float]] = None, viseme: Optional[Dict[str, float]] = None, dt: float = 0.016) -> Dict[str, Any]:
    stack = stack or build_advanced_transformers()
    perception = perception or {}
    advanced = stack.update(perception, viseme=viseme, dt=dt)
    out = dict(motion or {})
    out.update(advanced)
    return out


__all__ = [
    'SpaceOccupancyInput', 'SpaceOccupancyState', 'MotionIntent', 'MotionTarget',
    'OccupancyVisibilityController', 'OccupancyVisibilityTransformer', 'SpaceOccupancyTransformer', 'ExpertRouter',
    'IntentTransformer', 'MotionPlanningTransformer', 'TrajectoryTransformer', 'PhysicsTransformer',
    'SecondaryMotionTransformer', 'BodyRotationTransformer', 'SphericalEyeRollTransformer',
    'HeadEyeCoordinationTransformer', 'TorsionalRotationTransformer', 'ConvergenceTransformer',
    'GazeSaccadeTransformer', 'EmotionalBlinkTransformer', 'AsymmetricExpressionTransformer',
    'OromandibularKinematicTransformer', 'AdvancedAnimationOrchestrator', 'build_advanced_transformers',
    'apply_motion_stack', 'clamp', 'lerp', 'smoothstep', 'exp_smooth'
]


# =============================================================================
# V2 / ultra-robust orchestration layer
# =============================================================================

@dataclass
class MotionEvidence:
    source: str = 'unknown'
    score: float = 0.0
    timestamp: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)


class VectorMemoryBank(nn.Module):
    def __init__(self, dimension: int = 32, capacity: int = 96, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.dimension = int(dimension)
        self.capacity = int(capacity)
        self.device = device
        self.dtype = dtype
        self.register_buffer('vectors', torch.zeros(self.capacity, self.dimension, device=device, dtype=dtype))
        self.register_buffer('importance', torch.zeros(self.capacity, device=device, dtype=dtype))
        self.register_buffer('age', torch.zeros(self.capacity, device=device, dtype=dtype))
        self._cursor = 0

    def reset(self) -> None:
        self.vectors.zero_()
        self.importance.zero_()
        self.age.zero_()
        self._cursor = 0

    def write(self, vector: torch.Tensor, importance: float = 0.5, dt: float = 0.016) -> None:
        vec = vector.detach().to(device=self.vectors.device, dtype=self.vectors.dtype).flatten()
        if vec.numel() < self.dimension:
            pad = torch.zeros(self.dimension - vec.numel(), device=self.vectors.device, dtype=self.vectors.dtype)
            vec = torch.cat([vec, pad], dim=0)
        elif vec.numel() > self.dimension:
            vec = vec[: self.dimension]
        self.vectors[self._cursor].copy_(vec)
        self.importance[self._cursor] = clamp(float(importance), 0.0, 1.0)
        self.age[self._cursor] = 0.0
        self._cursor = (self._cursor + 1) % self.capacity
        self.age.add_(float(dt))
        self.importance.mul_(torch.exp(torch.tensor(-0.004 * float(dt), device=self.importance.device, dtype=self.importance.dtype)))

    def retrieve(self, query: torch.Tensor, topk: int = 6) -> torch.Tensor:
        q = query.detach().to(device=self.vectors.device, dtype=self.vectors.dtype).flatten()
        if q.numel() < self.dimension:
            q = torch.cat([q, torch.zeros(self.dimension - q.numel(), device=q.device, dtype=q.dtype)], dim=0)
        elif q.numel() > self.dimension:
            q = q[: self.dimension]
        if not torch.any(self.vectors):
            return torch.zeros(self.dimension, device=self.vectors.device, dtype=self.vectors.dtype)
        sims = torch.nn.functional.cosine_similarity(self.vectors, q.unsqueeze(0), dim=-1)
        recency = torch.exp(-0.015 * self.age)
        score = sims * (0.45 + 0.55 * self.importance) * recency
        k = min(int(topk), score.numel())
        if k <= 0:
            return torch.zeros(self.dimension, device=self.vectors.device, dtype=self.vectors.dtype)
        idx = torch.topk(score, k=k, largest=True).indices
        w = torch.softmax(score[idx], dim=0).unsqueeze(-1)
        return torch.sum(self.vectors[idx] * w, dim=0)

    def summary(self) -> torch.Tensor:
        if not torch.any(self.vectors):
            return torch.zeros(self.dimension * 3, device=self.vectors.device, dtype=self.vectors.dtype)
        return torch.cat([
            self.vectors.mean(dim=0),
            self.vectors.std(dim=0, unbiased=False),
            self.retrieve(self.vectors[self._cursor - 1 if self._cursor > 0 else 0], topk=4),
        ], dim=0)


class FourierMotionScheduler(nn.Module):
    def __init__(self, harmonics: int = 6, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.harmonics = int(harmonics)
        self.device = device
        self.dtype = dtype
        self.register_buffer('phase', torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer('secondary_phase', torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer('breath_phase', torch.tensor(0.0, device=device, dtype=dtype))

    def update(self, dt: float, intensity: float, tension: float, attention: float) -> Dict[str, float]:
        dtt = float(dt)
        self.phase.add_(dtt * (0.7 + 1.2 * float(intensity)))
        self.secondary_phase.add_(dtt * (1.4 + 0.8 * float(attention)))
        self.breath_phase.add_(dtt * (0.5 + 0.4 * float(intensity)))
        harmonic = 0.0
        for i in range(1, self.harmonics + 1):
            harmonic += math.sin(float(self.phase) * (0.9 + 0.17 * i) + i * 0.61) / i
        secondary = 0.0
        for i in range(1, self.harmonics + 1):
            secondary += math.sin(float(self.secondary_phase) * (1.1 + 0.11 * i) + i * 0.37) / i
        breath = 0.012 + 0.018 * (0.5 + 0.5 * math.sin(float(self.breath_phase) * math.tau))
        squash = clamp(0.04 + 0.10 * float(tension) + 0.03 * abs(harmonic), 0.0, 0.20)
        stretch = clamp(0.03 + 0.08 * float(intensity) + 0.03 * abs(secondary), 0.0, 0.18)
        return {
            'float_phase': float(self.phase.item()),
            'secondary_phase': float(self.secondary_phase.item()),
            'breath': breath,
            'harmonic': clamp(harmonic, -1.0, 1.0),
            'secondary': clamp(secondary, -1.0, 1.0),
            'squash': squash,
            'stretch': stretch,
            'pulse': clamp(0.5 + 0.35 * harmonic + 0.15 * secondary, 0.0, 1.0),
        }


class ClickSafetyTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.router = ExpertRouter(input_dim=24, hidden=64, device=device, dtype=dtype)
        self.register_buffer('last_gate', torch.zeros(8, device=device, dtype=dtype))

    def update(self, inp: SpaceOccupancyInput, state: SpaceOccupancyState, dt: float) -> Dict[str, Any]:
        w = max(1.0, float(inp.screen_width))
        h = max(1.0, float(inp.screen_height))
        ax = float(inp.avatar_x)
        ay = float(inp.avatar_y)
        mx = float(inp.mouse_x)
        my = float(inp.mouse_y)
        hx = clamp((ax + 0.5 * float(inp.avatar_scale)) / w, 0.0, 1.0)
        hy = clamp((ay + 0.5 * float(inp.avatar_scale)) / h, 0.0, 1.0)
        cursor_dx = mx - ax
        cursor_dy = my - ay
        cursor_dist = math.hypot(cursor_dx, cursor_dy)
        hot = clamp(1.0 - cursor_dist / max(1.0, float(inp.mouse_hot_radius)), 0.0, 1.0)
        drag_pressure = 1.0 if (inp.drag_active or inp.dragged_window_rect is not None) else 0.0
        full = 1.0 if inp.fullscreen_window_rect is not None or inp.fullscreen_pressure > 0.5 else 0.0
        workspace = clamp(float(inp.workspace_density), 0.0, 1.0)
        click_pressure = clamp(float(inp.click_pressure), 0.0, 1.0)
        attention = clamp(float(inp.attention_pressure), 0.0, 1.0)
        vector = torch.tensor([
            hx, hy, cursor_dx / w, cursor_dy / h, hot, drag_pressure, full, workspace,
            click_pressure, attention, float(inp.mouse_speed) / 1200.0, float(inp.mouse_pressure),
            float(inp.silence_pressure), float(inp.occupancy_pressure), float(inp.float_bias), float(inp.front_bias),
            float(inp.interaction_intensity), float(inp.window_velocity_x) / 1200.0, float(inp.window_velocity_y) / 1200.0,
            float(state.visibility_score), float(state.overlap_score), float(state.dodge_strength), float(state.float_amplitude), float(state.target_alpha),
        ], device=self.device, dtype=self.dtype)
        gate = self.router(vector)
        self.last_gate.copy_(gate)
        safe_click = bool((hot < 0.20 and drag_pressure < 0.5 and full < 0.5) or click_pressure > 0.75)
        click_through = bool((hot > 0.42 or workspace > 0.55 or full > 0.5) and not safe_click)
        topmost = bool(full > 0.5 or attention > 0.72 or state.visibility_score < 0.36)
        alpha_floor = clamp(0.08 + 0.12 * workspace + 0.18 * hot + 0.10 * full, 0.04, 0.65)
        layer_mode = 'overlay_minimal' if full > 0.5 else ('respect_space' if hot < 0.35 else 'layered_interaction')
        z_policy = 'front' if topmost else 'normal'
        return {
            'safe_click': safe_click,
            'click_through': click_through,
            'transparentcolor_safe': bool(not click_through or safe_click),
            'topmost': topmost,
            'alpha_floor': alpha_floor,
            'layer_mode': layer_mode,
            'z_policy': z_policy,
            'hot': hot,
            'drag_pressure': drag_pressure,
            'full_screen': bool(full > 0.5),
            'safety_gate': gate.detach().cpu().tolist(),
        }


class WindowTopologyTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.router = ExpertRouter(input_dim=24, hidden=80, device=device, dtype=dtype)
        self.register_buffer('last_slot', torch.zeros(6, device=device, dtype=dtype))

    def _score_slot(self, x: float, y: float, inp: SpaceOccupancyInput) -> float:
        score = 0.0
        for rect in inp.occupied_rects or []:
            x1, y1, x2, y2 = rect
            inside = 1.0 if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0
            dx = min(abs(x - x1), abs(x - x2))
            dy = min(abs(y - y1), abs(y - y2))
            score += inside * 2.0 + 0.18 / max(1.0, dx + dy)
        if inp.heatmap:
            gy = min(int(max(0.0, y / max(1.0, inp.screen_height)) * (len(inp.heatmap) - 1)), len(inp.heatmap) - 1)
            gx = min(int(max(0.0, x / max(1.0, inp.screen_width)) * (len(inp.heatmap[gy]) - 1)), len(inp.heatmap[gy]) - 1)
            score += float(inp.heatmap[gy][gx]) * 2.0
        center_dx = x - inp.screen_width * 0.5
        center_dy = y - inp.screen_height * 0.5
        score -= 0.000003 * (center_dx * center_dx + center_dy * center_dy)
        return score

    def pick_slot(self, inp: SpaceOccupancyInput, state: SpaceOccupancyState) -> Tuple[float, float, int, float]:
        margin = max(24.0, float(inp.safe_margin))
        w = max(1.0, float(inp.screen_width))
        h = max(1.0, float(inp.screen_height))
        candidates = [
            (margin + 0.16 * w, margin + 0.18 * h),
            (w - margin - 0.18 * w, margin + 0.16 * h),
            (margin + 0.16 * w, h - margin - 0.18 * h),
            (w - margin - 0.18 * w, h - margin - 0.18 * h),
            (0.50 * w, 0.15 * h),
            (0.50 * w, 0.83 * h),
            (0.18 * w, 0.50 * h),
            (0.82 * w, 0.50 * h),
            (0.50 * w, 0.50 * h),
        ]
        scored = []
        for i, (x, y) in enumerate(candidates):
            score = self._score_slot(x, y, inp)
            score += 0.25 * float(inp.front_bias) if i == state.home_slot_id else 0.0
            score -= 0.10 * abs(i - max(0, state.locked_slot_id))
            scored.append((score, x, y, i))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0]
        slot_score = float(best[0])
        return float(best[1]), float(best[2]), int(best[3]), slot_score

    def forward(self, inp: SpaceOccupancyInput, state: SpaceOccupancyState, dt: float) -> SpaceOccupancyState:
        x, y, slot_id, slot_score = self.pick_slot(inp, state)
        mouse_dx = float(inp.mouse_x) - float(inp.avatar_x)
        mouse_dy = float(inp.mouse_y) - float(inp.avatar_y)
        move_dist = math.hypot(mouse_dx, mouse_dy)
        drag_pressure = 1.0 if (inp.drag_active or inp.dragged_window_rect is not None) else 0.0
        full = 1.0 if inp.fullscreen_window_rect is not None or inp.fullscreen_pressure > 0.5 else 0.0
        overlap = clamp(float(inp.occupancy_pressure) + 0.40 * float(inp.workspace_density) + 0.35 * drag_pressure + 0.40 * full, 0.0, 1.0)
        visibility = clamp(1.0 - overlap * 0.78 + 0.18 * float(inp.focus), 0.0, 1.0)
        dodge = clamp(0.25 + 0.45 * drag_pressure + 0.25 * full + 0.16 * clamp(1.0 - move_dist / max(1.0, float(inp.mouse_hot_radius)), 0.0, 1.0), 0.0, 1.0)
        alpha = clamp(visibility * (0.70 + 0.25 * float(inp.front_bias)) , state.alpha_floor, state.alpha_ceiling)
        scale = clamp(0.78 + 0.16 * visibility + 0.10 * float(inp.float_bias), 0.56, 1.45)
        lock_time = max(0.22, float(inp.hold_lock_seconds))
        locked_until = state.locked_until
        if inp.allow_return_to_last_safe and slot_id == state.last_safe_slot_id:
            locked_until = max(locked_until, float(getattr(state, 'locked_until', 0.0)))
        if drag_pressure > 0.5 or full > 0.5 or move_dist < float(inp.mouse_hot_radius) * 0.36:
            locked_until = max(locked_until, time.time() + lock_time)
        if time.time() < state.locked_until and state.current_slot_id >= 0:
            x, y, slot_id = state.target_x, state.target_y, state.current_slot_id
        spring = 1.0 - math.exp(-max(1e-4, float(dt)) / max(0.02, state.smooth_time))
        state.target_x = lerp(state.target_x, x, spring)
        state.target_y = lerp(state.target_y, y, spring)
        state.target_alpha = lerp(state.target_alpha, alpha, spring)
        state.target_scale = lerp(state.target_scale, scale, spring)
        state.target_z = lerp(state.target_z, 0.92 if full > 0.5 else (0.72 if dodge > 0.55 else 0.52), spring)
        state.visible = state.target_alpha > 0.05
        state.topmost = bool(full > 0.5 or inp.talking or float(inp.interaction_intensity) > 0.55)
        state.layer_mode = 'overlay_minimal' if full > 0.5 else ('layered_interaction' if dodge > 0.42 else 'respect_space')
        state.anchor_x = lerp(state.anchor_x, float(inp.avatar_x), spring)
        state.anchor_y = lerp(state.anchor_y, float(inp.avatar_y), spring)
        state.current_slot_id = slot_id
        state.locked_slot_id = slot_id if slot_score >= 0.0 else state.locked_slot_id
        state.home_slot_id = state.home_slot_id if state.home_slot_id >= 0 else slot_id
        state.last_safe_slot_id = slot_id if visibility > 0.28 else state.last_safe_slot_id
        state.last_safe_x = state.target_x
        state.last_safe_y = state.target_y
        state.locked_until = max(state.locked_until, locked_until)
        state.dodge_strength = lerp(state.dodge_strength, dodge, spring)
        state.slide_x = lerp(state.slide_x, (x - state.anchor_x) * 0.42, spring)
        state.slide_y = lerp(state.slide_y, (y - state.anchor_y) * 0.42, spring)
        state.velocity_x = lerp(state.velocity_x, (state.target_x - inp.avatar_x) / max(1e-3, float(dt)), spring)
        state.velocity_y = lerp(state.velocity_y, (state.target_y - inp.avatar_y) / max(1e-3, float(dt)), spring)
        state.spring_x = lerp(state.spring_x, state.target_x, spring)
        state.spring_y = lerp(state.spring_y, state.target_y, spring)
        state.spring_vx = lerp(state.spring_vx, state.velocity_x, spring)
        state.spring_vy = lerp(state.spring_vy, state.velocity_y, spring)
        state.hover_bias = lerp(state.hover_bias, clamp(1.0 - move_dist / max(1.0, float(inp.mouse_hot_radius)), 0.0, 1.0), spring)
        state.overlap_score = overlap
        state.visibility_score = visibility
        state.escape_score = dodge
        state.float_amplitude = lerp(state.float_amplitude, 0.02 + 0.05 * float(inp.workspace_density) + 0.04 * float(inp.attention_pressure), spring)
        state.motion_curvature = lerp(state.motion_curvature, 0.08 + 0.18 * drag_pressure + 0.12 * full + 0.08 * state.hover_bias, spring)
        state.path_curvature = lerp(state.path_curvature, 0.10 + 0.22 * state.dodge_strength, spring)
        state.transition_alpha = state.target_alpha
        state.click_through = bool((state.target_alpha < 0.32 and visibility < 0.36 and not full > 0.5) or (drag_pressure < 0.1 and state.hover_bias < 0.18 and visibility < 0.28))
        state.click_blocking = not state.click_through
        state.z_policy = 'front' if state.topmost else ('overlay' if state.layer_mode != 'respect_space' else 'normal')
        state.pulse = clamp(0.5 + 0.18 * math.sin(time.time() * 1.2 + state.float_phase), 0.0, 1.0)
        state.reason = 'fullscreen' if full > 0.5 else ('drag_escape' if drag_pressure > 0.5 else ('cursor_avoid' if state.hover_bias > 0.35 else 'idle'))
        state.hold_reason = 'hold' if time.time() < state.locked_until else 'free'
        state.expert_name = 'window_topology_v2'
        self.last_slot.copy_(torch.tensor([state.target_x, state.target_y, state.target_alpha, state.target_scale, state.target_z, float(slot_id)], device=self.last_slot.device, dtype=self.last_slot.dtype))
        return state


class OralDynamicsFusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = OromandibularKinematicTransformer()
        self.bite_phase = 0.0
        self.swallow_phase = 0.0
        self.lick_phase = 0.0
        self.last = {
            'jaw_open': 0.0,
            'lip_round': 0.0,
            'lip_spread': 0.0,
            'upper_lip': 0.0,
            'lower_lip': 0.0,
            'tongue': 0.0,
            'cheek': 0.0,
            'saliva': 0.0,
            'lip_bite': 0.0,
            'mouth_compression': 0.0,
            'mouth_asymmetry': 0.0,
            'breath': 0.0,
        }

    def update(self, viseme: Optional[Dict[str, float]], emotion: Optional[Dict[str, float]], audio_energy: float, dt: float) -> Dict[str, float]:
        core = self.base.update(viseme, emotion, audio_energy, dt)
        emo = emotion or {}
        tension = clamp(float(emo.get('tension', 0.12)), 0.0, 1.0)
        curiosity = clamp(float(emo.get('curiosity', 0.35)), 0.0, 1.0)
        intensity = clamp(float(emo.get('intensity', 0.35)), 0.0, 1.0)
        valence = clamp(float(emo.get('valence', 0.0)), -1.0, 1.0)
        self.bite_phase += float(dt) * (0.8 + 0.9 * tension)
        self.swallow_phase += float(dt) * (0.3 + 0.5 * intensity)
        self.lick_phase += float(dt) * (0.2 + 0.3 * curiosity)
        bite = 0.08 * max(0.0, math.sin(self.bite_phase * math.tau)) * (0.35 + 0.65 * tension)
        swallow = 0.04 * max(0.0, math.sin(self.swallow_phase * math.tau)) * (0.25 + 0.75 * intensity)
        lip_asym = 0.06 * math.sin(self.lick_phase * math.tau + 0.7) * (0.25 + 0.75 * max(0.0, valence))
        mouth_compression = clamp(0.12 * tension + 0.08 * max(0.0, -valence) + swallow, 0.0, 1.0)
        out = dict(core)
        out['lip_bite'] = clamp(core.get('lip_bite', 0.0) + bite, 0.0, 1.0)
        out['mouth_compression'] = mouth_compression
        out['mouth_asymmetry'] = lip_asym
        out['swallow'] = swallow
        out['breath'] = clamp(core.get('saliva', 0.0) * 0.2 + 0.02 + 0.02 * intensity, 0.0, 1.0)
        return out


class UltraAnimationOrchestratorV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.intent = IntentTransformer(input_dim=20, hidden=80)
        self.plan = MotionPlanningTransformer()
        self.trajectory = TrajectoryTransformer()
        self.physics = PhysicsTransformer()
        self.secondary = SecondaryMotionTransformer()
        self.body_rot = BodyRotationTransformer()
        self.eye_roll = SphericalEyeRollTransformer()
        self.head_coord = HeadEyeCoordinationTransformer()
        self.torsion = TorsionalRotationTransformer()
        self.convergence = ConvergenceTransformer()
        self.blink = EmotionalBlinkTransformer()
        self.smirk = AsymmetricExpressionTransformer()
        self.mouth = OralDynamicsFusionTransformer()
        self.memory = VectorMemoryBank(dimension=20, capacity=96)
        self.scheduler = FourierMotionScheduler(harmonics=7)
        self.click_safety = ClickSafetyTransformer()
        self.window_topology = WindowTopologyTransformer()
        self.register_buffer('last_position', torch.zeros(2, dtype=DTYPE))

    def build_workspace_vector(self, perception: Dict[str, float]) -> torch.Tensor:
        keys = [
            'valence', 'arousal', 'dominance', 'attention', 'curiosity', 'confidence', 'tension', 'fatigue',
            'energy', 'speech', 'audio_energy', 'mouse_speed', 'mouse_distance', 'idle_seconds',
            'fullscreen_pressure', 'workspace_density', 'x', 'y', 'screen_w', 'screen_h',
            'click_pressure', 'drag_pressure', 'cursor_locked', 'hover_bias',
            'overlap_score', 'visibility_score', 'target_alpha', 'float_bias', 'front_bias',
            'interaction_intensity', 'window_velocity_x', 'window_velocity_y'
        ]
        vec = [float(perception.get(k, 0.0)) for k in keys]
        return torch.tensor(vec[:32], dtype=DTYPE)

    def update(self, perception: Dict[str, float], viseme: Optional[Dict[str, float]] = None, dt: float = 0.016) -> Dict[str, Any]:
        p = dict(perception or {})
        workspace = p.get('space_state')
        if not isinstance(workspace, SpaceOccupancyState):
            workspace = SpaceOccupancyState(target_x=float(p.get('x', 0.0)), target_y=float(p.get('y', 0.0)))
        vec = self.build_workspace_vector(p)
        self.memory.write(vec, importance=0.45 + 0.35 * float(p.get('attention', 0.5)) + 0.20 * float(p.get('tension', 0.1)), dt=dt)
        memory_context = self.memory.retrieve(vec, topk=5)
        memory_summary = self.memory.summary()
        intent_vec = torch.clamp(vec + 0.15 * memory_context[:20], -3.0, 3.0)
        intent = self.intent(intent_vec)
        target = self.plan(intent, workspace, float(p.get('screen_w', 1920.0)), float(p.get('screen_h', 1080.0)))
        safety = self.click_safety.update(
            SpaceOccupancyInput(
                avatar_x=float(p.get('x', 0.0)),
                avatar_y=float(p.get('y', 0.0)),
                avatar_scale=float(p.get('scale', 1.0)),
                avatar_alpha=float(p.get('alpha', 1.0)),
                mouse_x=float(p.get('mouse_x', 0.0)),
                mouse_y=float(p.get('mouse_y', 0.0)),
                mouse_dx=float(p.get('mouse_dx', 0.0)),
                mouse_dy=float(p.get('mouse_dy', 0.0)),
                mouse_speed=float(p.get('mouse_speed', 0.0)),
                mouse_pressure=clamp(float(p.get('mouse_speed', 0.0)) / 1200.0, 0.0, 1.0),
                focus=bool(p.get('focus', True)),
                click_pressure=clamp(float(p.get('click_pressure', 0.0)), 0.0, 1.0),
                fullscreen_pressure=clamp(float(p.get('fullscreen_pressure', 0.0)), 0.0, 1.0),
                silence_pressure=clamp(float(p.get('idle_seconds', 0.0)) / 12.0, 0.0, 1.0),
                workspace_density=clamp(float(p.get('workspace_density', 0.0)), 0.0, 1.0),
                occupancy_pressure=clamp(float(p.get('occupancy_pressure', 0.0)), 0.0, 1.0),
                idle_seconds=float(p.get('idle_seconds', 0.0)),
                time_since_input=float(p.get('idle_seconds', 0.0)),
                screen_width=float(p.get('screen_w', 1920.0)),
                screen_height=float(p.get('screen_h', 1080.0)),
                safe_margin=24.0,
                float_bias=clamp(float(p.get('attention', 0.5)), 0.0, 1.0),
                attention_pressure=clamp(float(p.get('attention', 0.5)), 0.0, 1.0),
                front_bias=0.66 if bool(p.get('focus', True)) else 0.82,
                mouse_hot_radius=260.0,
                cursor_locked=False,
                heatmap=p.get('heatmap'),
                occupied_rects=p.get('occupied_rects'),
                active_window_rect=p.get('active_window_rect'),
                dragged_window_rect=p.get('dragged_window_rect'),
                fullscreen_window_rect=p.get('fullscreen_window_rect'),
                z_order_hint=0.55 if p.get('fullscreen_window_rect') is not None else 0.2,
                interaction_intensity=clamp(float(p.get('audio_energy', 0.0)) * 2.6 + float(p.get('mouse_speed', 0.0)) / 1600.0, 0.0, 1.0),
                window_velocity_x=float(p.get('mouse_dx', 0.0)),
                window_velocity_y=float(p.get('mouse_dy', 0.0)),
                talking=bool(p.get('speech', 0.0) > 0.5),
                window_count=int(p.get('window_count', 1)),
                drag_active=bool(p.get('drag_active', False)),
                allow_return_to_last_safe=True,
                hold_lock_seconds=0.85,
                preferred_slot_index=-1,
                return_bias=0.35,
                occupied_rect_bias=0.60,
                workspace_focus_bias=0.25,
            ),
            workspace,
            dt,
        )
        topology = self.window_topology.pick_slot(
            SpaceOccupancyInput(avatar_x=float(p.get('x', 0.0)), avatar_y=float(p.get('y', 0.0)), screen_width=float(p.get('screen_w', 1920.0)), screen_height=float(p.get('screen_h', 1080.0)), occupied_rects=p.get('occupied_rects'), heatmap=p.get('heatmap')), workspace
        )
        pos = self.trajectory.step((float(p.get('x', 0.0)), float(p.get('y', 0.0))), (target.x, target.y), dt)
        body_rot = self.body_rot.update(intent, dt)
        eye = self.eye_roll.update((float(p.get('mouse_x', 0.0)), float(p.get('mouse_y', 0.0))), (float(p.get('eye_x', 0.0)), float(p.get('eye_y', 0.0))), dt=dt)
        head = self.head_coord.update((eye[0], eye[1]), body_rot, dt)
        torsion = self.torsion.update(float(p.get('uncertainty', 0.0)), dt)
        convergence = self.convergence.update(float(p.get('mouse_distance', 999.0)), dt)
        blink = self.blink.update(float(p.get('tension', 0.12)), float(p.get('curiosity', 0.3)), float(p.get('fatigue', 0.1)), dt)
        smirk = self.smirk.update(float(p.get('valence', 0.0)), float(p.get('tension', 0.0)), dt)
        mouth = self.mouth.update(viseme, p, float(p.get('audio_energy', 0.0)), dt)
        schedule = self.scheduler.update(dt, float(p.get('intensity', p.get('attention', 0.5))), float(p.get('tension', 0.1)), float(p.get('attention', 0.5)))
        physics_position = self.physics.step((float(p.get('x', 0.0)), float(p.get('y', 0.0))), (target.x, target.y), dt)
        self.last_position.copy_(torch.tensor(pos, dtype=self.last_position.dtype, device=self.last_position.device))
        body_rotation = (
            float(body_rot[0]) + 0.18 * schedule['harmonic'],
            float(body_rot[1]) + 0.12 * schedule['secondary'],
            float(body_rot[2]) + 0.08 * schedule['harmonic'] + 0.5 * float(torsion),
        )
        head_rotation = (
            float(head[0]),
            float(head[1]),
            float(head[2]) + 0.12 * float(smirk[0] + smirk[1]),
        )
        return {
            'position': pos,
            'physics_position': physics_position,
            'body_rotation': body_rotation,
            'head_rotation': head_rotation,
            'torsion': torsion,
            'convergence': convergence,
            'blink': blink,
            'smirk': smirk,
            'mouth': mouth,
            'intent': intent,
            'motion_target': target,
            'workspace_gate': self.plan.router.last_gate.detach().cpu().tolist(),
            'memory_vector': memory_context.detach().cpu().tolist(),
            'memory_summary': memory_summary.detach().cpu().tolist(),
            'schedule': schedule,
            'click_through': safety['click_through'],
            'safe_click': safety['safe_click'],
            'transparentcolor_safe': safety['transparentcolor_safe'],
            'topmost': safety['topmost'],
            'alpha_floor': safety['alpha_floor'],
            'layer_mode': safety['layer_mode'],
            'z_policy': safety['z_policy'],
            'safety_gate': safety['safety_gate'],
            'window_slot': {'x': topology[0], 'y': topology[1], 'slot_id': topology[2], 'score': topology[3]},
            'body_scale': clamp(1.0 + schedule['stretch'] - schedule['squash'], 0.70, 1.35),
            'motion_curvature': workspace.motion_curvature,
            'path_curvature': workspace.path_curvature,
            'space_state': workspace.as_dict(),
            'secondary_motion': {
                'breath': schedule['breath'],
                'squash': schedule['squash'],
                'stretch': schedule['stretch'],
                'pulse': schedule['pulse'],
            },
        }


# Re-export the V2 orchestrator as the default builder.
AdvancedAnimationOrchestrator = UltraAnimationOrchestratorV2


def build_advanced_transformers() -> UltraAnimationOrchestratorV2:
    return UltraAnimationOrchestratorV2()


def apply_motion_stack(motion: Dict[str, Any], stack: Optional[UltraAnimationOrchestratorV2] = None, perception: Optional[Dict[str, float]] = None, viseme: Optional[Dict[str, float]] = None, dt: float = 0.016) -> Dict[str, Any]:
    stack = stack or build_advanced_transformers()
    perception = perception or {}
    advanced = stack.update(perception, viseme=viseme, dt=dt)
    out = dict(motion or {})
    out.update(advanced)
    return out


try:
    __all__ = list(dict.fromkeys(list(__all__) + [
        'MotionEvidence', 'VectorMemoryBank', 'FourierMotionScheduler', 'ClickSafetyTransformer',
        'WindowTopologyTransformer', 'OralDynamicsFusionTransformer', 'UltraAnimationOrchestratorV2',
        'AdvancedAnimationOrchestrator',
    ]))
except Exception:
    __all__ = [
        'SpaceOccupancyInput', 'SpaceOccupancyState', 'MotionIntent', 'MotionTarget',
        'OccupancyVisibilityController', 'OccupancyVisibilityTransformer', 'SpaceOccupancyTransformer', 'ExpertRouter',
        'IntentTransformer', 'MotionPlanningTransformer', 'TrajectoryTransformer', 'PhysicsTransformer',
        'SecondaryMotionTransformer', 'BodyRotationTransformer', 'SphericalEyeRollTransformer',
        'HeadEyeCoordinationTransformer', 'TorsionalRotationTransformer', 'ConvergenceTransformer',
        'GazeSaccadeTransformer', 'EmotionalBlinkTransformer', 'AsymmetricExpressionTransformer',
        'OromandibularKinematicTransformer', 'AdvancedAnimationOrchestrator', 'UltraAnimationOrchestratorV2',
        'build_advanced_transformers', 'apply_motion_stack', 'clamp', 'lerp', 'smoothstep', 'exp_smooth'
    ]

try:
    from avatar_orchestrator_experts import patch_all as _patch_all_experts
    _patch_all_experts(globals())
except Exception:
    pass
