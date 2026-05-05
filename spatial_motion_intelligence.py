
from __future__ import annotations

import ctypes
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

DTYPE = torch.float32
DEVICE = torch.device("cpu")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return float(target)
    k = 1.0 - math.exp(-max(1e-6, float(dt)) / tau)
    return lerp(current, target, k)


def safe_norm2(x: float, y: float, eps: float = 1e-8) -> float:
    return max(eps, math.hypot(float(x), float(y)))


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def rect_center(rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = rect
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def rect_area(rect: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = rect
    return max(1.0, (x2 - x1) * (y2 - y1))


def rect_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / rect_area(a)


def point_to_rect_distance(x: float, y: float, rect: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = rect
    cx = min(max(x, x1), x2)
    cy = min(max(y, y1), y2)
    return math.hypot(x - cx, y - cy)


@dataclass
class ScreenWindow:
    left: float
    top: float
    width: float
    height: float
    title: str = ""
    z_index: int = 0
    active: bool = True
    fullscreen: bool = False
    moving: bool = False

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return rect_center(self.rect)


@dataclass
class ScreenContext:
    screen_width: float = 1920.0
    screen_height: float = 1080.0
    avatar_x: float = 0.0
    avatar_y: float = 0.0
    avatar_scale: float = 1.0
    avatar_alpha: float = 1.0
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    focus: bool = True
    click_pressure: float = 0.0
    fullscreen_pressure: float = 0.0
    silence_pressure: float = 0.0
    occupancy_pressure: float = 0.0
    workspace_density: float = 0.0
    idle_seconds: float = 0.0
    safe_margin: float = 24.0
    float_bias: float = 0.0
    drag_active: bool = False
    cursor_click_intent: bool = False
    active_window: Optional[ScreenWindow] = None
    occupied_rects: Optional[Sequence[Tuple[float, float, float, float]]] = None
    heatmap: Optional[Sequence[Sequence[float]]] = None

    def avatar_rect(self, size: Optional[float] = None) -> Tuple[float, float, float, float]:
        size = float(size or max(128.0, self.avatar_scale * 0.95))
        return (self.avatar_x, self.avatar_y, self.avatar_x + size, self.avatar_y + size)

    def as_features(self) -> torch.Tensor:
        w = max(1.0, float(self.screen_width))
        h = max(1.0, float(self.screen_height))
        ax = self.avatar_x / w
        ay = self.avatar_y / h
        mx = self.mouse_x / w
        my = self.mouse_y / h
        dx = self.mouse_dx / max(1.0, w)
        dy = self.mouse_dy / max(1.0, h)
        cursor_dist = math.hypot(self.mouse_x - self.avatar_x, self.mouse_y - self.avatar_y)
        cursor_pressure = clamp(1.0 - cursor_dist / max(90.0, self.avatar_scale * 0.55), 0.0, 1.0)
        full = clamp(self.fullscreen_pressure, 0.0, 1.0)
        silence = clamp(self.silence_pressure, 0.0, 1.0)
        occupancy = clamp(self.occupancy_pressure, 0.0, 1.0)
        density = clamp(self.workspace_density, 0.0, 1.0)
        idle = clamp(self.idle_seconds / 20.0, 0.0, 1.0)
        drag = 1.0 if self.drag_active else 0.0
        focus = 1.0 if self.focus else 0.0
        click_intent = 1.0 if self.cursor_click_intent else 0.0
        return torch.tensor([
            ax, ay, mx, my, dx, dy,
            self.mouse_speed / 1800.0,
            cursor_pressure, full, silence,
            occupancy, density, idle, drag, focus, click_intent,
        ], device=DEVICE, dtype=DTYPE)


@dataclass
class MotionIntent:
    approach: float = 0.1
    avoid: float = 0.1
    focus: float = 0.5
    respect: float = 0.5
    floatiness: float = 0.1
    visibility: float = 0.6
    click_safety: float = 0.9
    body_intensity: float = 0.35
    move_x: float = 0.0
    move_y: float = 0.0
    rotation: float = 0.0
    head_latency: float = 0.12
    z_bias: float = 0.5
    target_alpha: float = 1.0
    target_scale: float = 1.0


class IntentTransformer(nn.Module):
    """Gera intenção contínua a partir do estado espacial e comportamental."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),
        )
        self.register_buffer("ema_intent", torch.zeros(9, device=device, dtype=dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] < 16:
            pad = torch.zeros(*features.shape[:-1], 16 - features.shape[-1], device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > 16:
            features = features[..., :16]
        out = self.net(features).squeeze(0)
        intent = torch.tanh(out)
        self.ema_intent = self.ema_intent * 0.86 + intent.detach() * 0.14
        return self.ema_intent if self.training else intent

    def decode(self, out: torch.Tensor) -> MotionIntent:
        if out.ndim > 1:
            out = out.squeeze(0)
        vals = [float(v.item()) for v in out]
        return MotionIntent(
            approach=sigmoid(vals[0]),
            avoid=sigmoid(vals[1]),
            focus=sigmoid(vals[2]),
            respect=sigmoid(vals[3]),
            floatiness=sigmoid(vals[4]),
            visibility=sigmoid(vals[5]),
            click_safety=sigmoid(vals[6]),
            body_intensity=sigmoid(vals[7]),
            move_x=math.tanh(vals[0] - vals[1]) * 0.5,
            move_y=math.tanh(vals[2] - vals[3]) * 0.5,
            rotation=math.tanh(vals[8]) * 8.0,
            head_latency=0.08 + 0.16 * sigmoid(vals[4]),
            z_bias=clamp(0.30 + 0.40 * sigmoid(vals[5]), 0.0, 1.0),
            target_alpha=clamp(0.15 + 0.85 * sigmoid(vals[5]), 0.0, 1.0),
            target_scale=clamp(0.72 + 0.78 * sigmoid(vals[7]), 0.7, 1.5),
        )


class MotionPlanningTransformer:
    """Escolhe slots fixos, evita janelas e mantém a Denise fora do caminho do utilizador."""

    def __init__(self, screen_width: float = 1920.0, screen_height: float = 1080.0, avatar_size: float = 900.0):
        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)
        self.avatar_size = float(avatar_size)
        self.hold_seconds = 1.2
        self.hold_timer = 0.0
        self.last_target: Tuple[float, float] = (screen_width * 0.62, screen_height * 0.56)
        self.fixed_slots = self._build_slots()
        self.slot_bias: Dict[int, float] = {i: 0.0 for i in range(len(self.fixed_slots))}
        self.preferred_slot = 0
        self.safe_slot_lock = 0.0
        self.last_reason = "idle"

    def _build_slots(self) -> List[Tuple[float, float]]:
        w, h, s = self.screen_width, self.screen_height, self.avatar_size
        margin = 24.0
        slots = [
            (w - s - margin, h - s - margin),
            (w - s - margin, margin),
            (margin, h - s - margin),
            (margin, margin),
            (w * 0.50 - s * 0.50, h - s - margin),
            (w * 0.50 - s * 0.50, margin),
            (w - s - margin, h * 0.50 - s * 0.50),
            (margin, h * 0.50 - s * 0.50),
            (w * 0.50 - s * 0.50, h * 0.50 - s * 0.50),
        ]
        return [(clamp(x, margin, max(margin, w - s - margin)), clamp(y, margin, max(margin, h - s - margin))) for x, y in slots]

    def _heat_at(self, x: float, y: float, heatmap: Optional[Sequence[Sequence[float]]]) -> float:
        if not heatmap:
            return 0.0
        rows = len(heatmap)
        cols = len(heatmap[0]) if rows and heatmap[0] is not None else 0
        if rows <= 0 or cols <= 0:
            return 0.0
        gx = int(clamp(x / max(1.0, self.screen_width) * cols, 0, cols - 1))
        gy = int(clamp(y / max(1.0, self.screen_height) * rows, 0, rows - 1))
        try:
            return float(heatmap[gy][gx])
        except Exception:
            return 0.0

    def _overlap_cost(self, rect: Tuple[float, float, float, float], occupied_rects: Optional[Sequence[Tuple[float, float, float, float]]]) -> float:
        if not occupied_rects:
            return 0.0
        total = 0.0
        for other in occupied_rects:
            total += rect_overlap(rect, other)
        return clamp(total, 0.0, 1.0)

    def _score_slot(self, slot_xy: Tuple[float, float], ctx: ScreenContext, intent: MotionIntent) -> float:
        x, y = slot_xy
        avatar_rect = (x, y, x + self.avatar_size, y + self.avatar_size)
        cx, cy = rect_center(avatar_rect)
        dist_mouse = math.hypot(cx - ctx.mouse_x, cy - ctx.mouse_y)
        dist_avatar = math.hypot(x - ctx.avatar_x, y - ctx.avatar_y)
        heat = self._heat_at(cx, cy, ctx.heatmap)
        overlap = self._overlap_cost(avatar_rect, ctx.occupied_rects)
        edge = min(cx / max(1.0, self.screen_width), 1.0 - cx / max(1.0, self.screen_width)) + min(cy / max(1.0, self.screen_height), 1.0 - cy / max(1.0, self.screen_height))
        edge_bonus = 1.0 - clamp(edge, 0.0, 1.0)
        center_bias = 1.0 - clamp(abs(cx - self.screen_width * 0.5) / max(1.0, self.screen_width * 0.5), 0.0, 1.0) * 0.6
        cursor_line = abs((cx - ctx.mouse_x) * ctx.mouse_dy - (cy - ctx.mouse_y) * ctx.mouse_dx)
        glide = 1.0 / (1.0 + cursor_line / 250000.0)
        safety = clamp(1.0 - math.exp(-dist_mouse / 180.0), 0.0, 1.0)
        return (
            1.40 * heat +
            1.55 * overlap +
            1.10 * (1.0 - safety) +
            0.30 * edge_bonus +
            0.24 * center_bias * intent.focus +
            0.18 * ctx.workspace_density +
            0.12 * ctx.occupancy_pressure +
            0.10 * ctx.fullscreen_pressure +
            0.08 * ctx.silence_pressure +
            0.08 * glide +
            0.04 * dist_avatar / max(self.screen_width, self.screen_height)
        )

    def choose(self, ctx: ScreenContext, intent: MotionIntent) -> Tuple[float, float, str]:
        if ctx.active_window and getattr(ctx.active_window, "fullscreen", False):
            self.safe_slot_lock = max(self.safe_slot_lock, 0.65)
        self.safe_slot_lock = max(0.0, self.safe_slot_lock - 0.016)
        scored: List[Tuple[float, float, float, int]] = []
        for idx, slot in enumerate(self.fixed_slots):
            score = self._score_slot(slot, ctx, intent) + self.slot_bias.get(idx, 0.0)
            scored.append((score, slot[0], slot[1], idx))
        scored.sort(key=lambda item: item[0])
        if ctx.drag_active or ctx.cursor_click_intent or ctx.fullscreen_pressure > 0.66:
            reason = "respect_space"
        elif ctx.mouse_speed > 850.0 or ctx.click_pressure > 0.45:
            reason = "mouse_avoid"
        elif ctx.workspace_density > 0.60:
            reason = "workspace_compact"
        elif ctx.idle_seconds > 10.0:
            reason = "float"
        elif ctx.silence_pressure > 0.50:
            reason = "quiet_float"
        else:
            reason = "wander"

        best = scored[0]
        if reason in {"mouse_avoid", "respect_space"} and len(scored) > 1:
            # mantém o slot fixo por um tempo para não ficar indo e voltando
            if self.safe_slot_lock > 0.0:
                best = scored[min(1, len(scored) - 1)]
            else:
                best = scored[0]

        bx, by, idx = best[1], best[2], best[3]
        if reason in {"float", "quiet_float"}:
            self.hold_timer += 0.016
            if self.hold_timer > self.hold_seconds:
                self.preferred_slot = (self.preferred_slot + 1) % len(self.fixed_slots)
                bx, by = self.fixed_slots[self.preferred_slot]
                idx = self.preferred_slot
                self.hold_timer = 0.0
        else:
            self.hold_timer = 0.0

        self.last_target = (bx, by)
        self.last_reason = reason
        self.slot_bias[idx] = exp_smooth(self.slot_bias.get(idx, 0.0), 0.0, 0.016, 0.35)
        return bx, by, reason


class TrajectoryTransformer:
    """Move a Denise com curva, aceleração e desaceleração sem tremor."""

    def __init__(self):
        self.position_x = 0.0
        self.position_y = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.anchor_x = 0.0
        self.anchor_y = 0.0
        self.curve_seed = random.random() * 50.0
        self.phase = random.random() * math.tau
        self.last_target: Tuple[float, float] = (0.0, 0.0)

    def reset(self, x: float, y: float) -> None:
        self.position_x = float(x)
        self.position_y = float(y)
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.anchor_x = float(x)
        self.anchor_y = float(y)
        self.last_target = (float(x), float(y))

    def step(self, target_x: float, target_y: float, dt: float, spring: float = 13.0, damping: float = 0.84, overshoot: float = 0.12, return_curve: float = 0.0) -> Tuple[float, float, float, float]:
        dt = max(1e-4, float(dt))
        target_x = float(target_x)
        target_y = float(target_y)
        if self.position_x == 0.0 and self.position_y == 0.0 and self.anchor_x == 0.0 and self.anchor_y == 0.0:
            self.reset(target_x, target_y)

        dx = target_x - self.position_x
        dy = target_y - self.position_y
        dist = math.hypot(dx, dy)

        # curva suave: desloca a trajetória num arco leve e previsível
        self.phase += dt * (1.3 + 0.7 * clamp(dist / 320.0, 0.0, 1.0))
        curve = math.sin(self.phase + self.curve_seed) * overshoot * clamp(dist / 260.0, 0.0, 1.0)
        normal_x = -dy / max(1e-6, dist) if dist > 1e-6 else 0.0
        normal_y = dx / max(1e-6, dist) if dist > 1e-6 else 0.0
        curved_target_x = target_x + normal_x * curve * 22.0
        curved_target_y = target_y + normal_y * curve * 22.0

        self.velocity_x += (curved_target_x - self.position_x) * spring * dt
        self.velocity_y += (curved_target_y - self.position_y) * spring * dt
        self.velocity_x *= damping
        self.velocity_y *= damping

        # amortecimento adicional quando já está perto do alvo
        close = clamp(1.0 - dist / 42.0, 0.0, 1.0)
        if close > 0.0:
            self.velocity_x *= 1.0 - 0.25 * close
            self.velocity_y *= 1.0 - 0.25 * close

        self.position_x += self.velocity_x
        self.position_y += self.velocity_y

        # evita tremor ao chegar
        if dist < 1.3 and abs(self.velocity_x) < 0.08 and abs(self.velocity_y) < 0.08:
            self.position_x = target_x
            self.position_y = target_y
            self.velocity_x = 0.0
            self.velocity_y = 0.0

        return self.position_x, self.position_y, self.velocity_x, self.velocity_y


class PhysicsTransformer:
    """Integra massa, inércia e desaceleração para corpo e rotação."""

    def __init__(self):
        self.rotation = 0.0
        self.rotation_velocity = 0.0
        self.scale = 1.0
        self.scale_velocity = 0.0
        self.head_delay = 0.0

    def step(self, target_rotation: float, target_scale: float, dt: float, stiffness: float = 10.0, damping: float = 0.84) -> Tuple[float, float]:
        dt = max(1e-4, float(dt))
        self.rotation_velocity += (float(target_rotation) - self.rotation) * stiffness * dt
        self.rotation_velocity *= damping
        self.rotation += self.rotation_velocity
        self.scale_velocity += (float(target_scale) - self.scale) * (stiffness * 0.7) * dt
        self.scale_velocity *= damping
        self.scale += self.scale_velocity
        if abs(self.rotation_velocity) < 0.01:
            self.rotation_velocity *= 0.92
        if abs(self.scale_velocity) < 0.01:
            self.scale_velocity *= 0.92
        return self.rotation, self.scale


class BodyRotationTransformer(nn.Module):
    """Gera rotação axial do corpo, atraso da cabeça e assimetria suave."""

    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 8),
        )
        self.register_buffer("ema", torch.zeros(8, device=device, dtype=dtype))
        self.physics = PhysicsTransformer()

    def forward(self, emotion_intensity: float, energy: float, attention: float, tension: float, curiosity: float, motion_speed: float, body_scale: float, mouse_pressure: float) -> Dict[str, float]:
        x = torch.tensor([
            float(emotion_intensity), float(energy), float(attention), float(tension),
            float(curiosity), float(motion_speed), float(body_scale), float(mouse_pressure)
        ], device=self.device, dtype=self.dtype)
        raw = self.net(x)
        smooth = torch.tanh(raw)
        self.ema = self.ema * 0.88 + smooth.detach() * 0.12
        vals = self.ema if self.training else smooth
        lean_x = math.tanh(float(vals[0].item())) * 0.14
        lean_y = math.tanh(float(vals[1].item())) * 0.14
        rotation = math.tanh(float(vals[2].item())) * 9.0 + tension * 2.0 - curiosity * 1.5
        scale_x = 1.0 + math.tanh(float(vals[3].item())) * 0.045
        scale_y = 1.0 + math.tanh(float(vals[4].item())) * 0.055
        head_x = math.tanh(float(vals[5].item())) * 0.10
        head_y = math.tanh(float(vals[6].item())) * 0.10
        torsion = math.tanh(float(vals[7].item())) * 6.0
        self.physics.rotation, self.physics.scale = self.physics.step(rotation, 1.0 + 0.02 * emotion_intensity, 0.016)
        return {
            "body_lean_x": lean_x,
            "body_lean_y": lean_y,
            "rotation": self.physics.rotation + torsion * 0.06,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "head_x": head_x,
            "head_y": head_y,
            "head_tilt": torsion,
        }


class ZOrderVisibilityTransformer:
    """Define alpha, z-order e click-through com base na atenção e ocupação."""

    def __init__(self):
        self.alpha = 1.0
        self.z_order = 0.5
        self.click_through = False
        self.always_on_top = True
        self.mode = "overlay"

    def step(self, ctx: ScreenContext, intent: MotionIntent, motion_intent: Optional[MotionIntent] = None) -> Dict[str, float | bool | str]:
        full = clamp(ctx.fullscreen_pressure, 0.0, 1.0)
        busy = clamp(ctx.workspace_density * 0.6 + ctx.occupancy_pressure * 0.5 + ctx.click_pressure * 0.3 + ctx.mouse_speed / 2200.0, 0.0, 1.0)
        idle = clamp(ctx.idle_seconds / 10.0, 0.0, 1.0)
        cursor_hot = clamp(1.0 - math.hypot(ctx.mouse_x - ctx.avatar_x, ctx.mouse_y - ctx.avatar_y) / max(80.0, ctx.avatar_scale * 0.42), 0.0, 1.0)
        overlay = full > 0.50 or busy > 0.72
        if full > 0.72:
            self.mode = "fullscreen_minimal"
            target_alpha = 0.32 + 0.18 * (1.0 - busy)
            self.always_on_top = True
        elif ctx.drag_active:
            self.mode = "respect_space"
            target_alpha = 0.28
            self.always_on_top = True
        elif busy > 0.65:
            self.mode = "compact_overlay"
            target_alpha = 0.48
            self.always_on_top = True
        elif idle > 0.50:
            self.mode = "float_visible"
            target_alpha = 0.82
            self.always_on_top = True
        else:
            self.mode = "standard"
            target_alpha = 0.92
            self.always_on_top = True

        # So permite click-through quando isso evita atrapalhar o usuário
        should_click_through = overlay or cursor_hot > 0.18 or ctx.cursor_click_intent or busy > 0.58
        if ctx.drag_active:
            should_click_through = False

        # durante interação com a Denise, ela pode ficar mais visível mas ainda não bloquear clicks
        if ctx.cursor_click_intent and not ctx.drag_active:
            target_alpha = max(target_alpha, 0.38)
            should_click_through = True

        self.alpha = exp_smooth(self.alpha, clamp(target_alpha, 0.04, 1.0), 0.016, 0.18)
        self.z_order = exp_smooth(self.z_order, 0.80 if (overlay or idle > 0.45) else 0.62, 0.016, 0.22)
        self.click_through = bool(should_click_through)
        return {
            "alpha": float(self.alpha),
            "z_order": float(self.z_order),
            "click_through": self.click_through,
            "always_on_top": self.always_on_top,
            "mode": self.mode,
            "busy": busy,
            "cursor_hot": cursor_hot,
        }


class ClickThroughManager:
    """Ativa WS_EX_LAYERED / WS_EX_TRANSPARENT sem conflitar com transparentcolor."""

    def __init__(self):
        self.enabled = False
        self.hwnd = None

    def attach(self, root: Any) -> None:
        try:
            root.update_idletasks()
            self.hwnd = int(root.winfo_id())
        except Exception:
            self.hwnd = None

    def _set_windows_style(self, enabled: bool) -> None:
        if os.name != "nt" or not self.hwnd:
            return
        try:
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080
            user32 = ctypes.windll.user32
            hwnd = int(self.hwnd)
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW
            if enabled:
                style |= WS_EX_TRANSPARENT
            else:
                style &= ~WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            # força refresh do style
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0020)
        except Exception:
            pass

    def update(self, root: Any, enabled: bool) -> None:
        if self.hwnd is None:
            self.attach(root)
        enabled = bool(enabled)
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self._set_windows_style(enabled)


class MotionOrchestrator:
    """Une intenção, plano espacial, trajetória e visibilidade num único controlador."""

    def __init__(self, screen_width: float = 1920.0, screen_height: float = 1080.0, avatar_size: float = 900.0):
        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)
        self.avatar_size = float(avatar_size)
        self.intent = IntentTransformer()
        self.planner = MotionPlanningTransformer(screen_width, screen_height, avatar_size)
        self.trajectory = TrajectoryTransformer()
        self.body_rotation = BodyRotationTransformer()
        self.visibility = ZOrderVisibilityTransformer()
        self.physics = PhysicsTransformer()
        self.last_space: Dict[str, Any] = {
            "target_x": screen_width * 0.62,
            "target_y": screen_height * 0.56,
            "reason": "idle",
        }
        self.initialized = False

    def step(self, ctx: ScreenContext, emotion_intensity: float = 0.25, energy: float = 0.35, attention: float = 0.45, tension: float = 0.10, curiosity: float = 0.20, motion_speed: float = 0.35) -> Dict[str, Any]:
        features = ctx.as_features()
        raw = self.intent(features)
        intent = self.intent.decode(raw)
        intent.body_intensity = clamp((emotion_intensity + energy + motion_speed) / 3.0, 0.0, 1.0)
        intent.click_safety = clamp(1.0 - ctx.mouse_speed / 2500.0, 0.0, 1.0)
        intent.visibility = clamp(0.35 + 0.65 * (1.0 - ctx.workspace_density * 0.35), 0.0, 1.0)
        intent.floatiness = clamp(0.15 + 0.85 * (ctx.idle_seconds / 12.0), 0.0, 1.0)
        intent.focus = clamp(attention, 0.0, 1.0)

        target_x, target_y, reason = self.planner.choose(ctx, intent)
        if not self.initialized:
            self.trajectory.reset(target_x, target_y)
            self.initialized = True

        # trajetória suave e inercial
        pos_x, pos_y, vel_x, vel_y = self.trajectory.step(
            target_x, target_y, dt=0.016,
            spring=12.0 + 3.0 * intent.approach,
            damping=0.84,
            overshoot=0.08 + 0.10 * intent.floatiness,
        )

        body = self.body_rotation(
            emotion_intensity=emotion_intensity,
            energy=energy,
            attention=attention,
            tension=tension,
            curiosity=curiosity,
            motion_speed=motion_speed,
            body_scale=1.0,
            mouse_pressure=clamp(1.0 - math.hypot(ctx.mouse_x - ctx.avatar_x, ctx.mouse_y - ctx.avatar_y) / 240.0, 0.0, 1.0),
        )
        visibility = self.visibility.step(ctx, intent)

        # ponte entre intenção e física
        rot, scale = self.physics.step(body["rotation"], body["scale_x"], 0.016, stiffness=9.0, damping=0.86)
        body["rotation"] = rot
        body["scale_x"] = scale
        body["scale_y"] = exp_smooth(body["scale_y"], body["scale_y"], 0.016, 0.18)

        self.last_space = {
            "target_x": float(pos_x),
            "target_y": float(pos_y),
            "velocity_x": float(vel_x),
            "velocity_y": float(vel_y),
            "reason": reason,
            "intent": {
                "approach": intent.approach,
                "avoid": intent.avoid,
                "focus": intent.focus,
                "respect": intent.respect,
                "floatiness": intent.floatiness,
                "visibility": intent.visibility,
                "click_safety": intent.click_safety,
                "body_intensity": intent.body_intensity,
                "z_bias": intent.z_bias,
            },
            "visibility": visibility,
            "body": body,
        }
        return self.last_space


__all__ = [
    "ScreenWindow",
    "ScreenContext",
    "MotionIntent",
    "IntentTransformer",
    "MotionPlanningTransformer",
    "TrajectoryTransformer",
    "PhysicsTransformer",
    "BodyRotationTransformer",
    "ZOrderVisibilityTransformer",
    "ClickThroughManager",
    "MotionOrchestrator",
    "clamp",
    "lerp",
    "smoothstep",
    "exp_smooth",
]
