from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random

import torch
import torch.nn as nn

DTYPE = torch.float32
DEVICE = torch.device("cpu")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-dt / tau)
    return lerp(current, target, k)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def smooth_damp(current: float, target: float, velocity: float, smooth_time: float, dt: float) -> Tuple[float, float]:
    smooth_time = max(1e-3, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * dt
    exp_term = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    temp = (velocity + omega * change) * dt
    new_velocity = (velocity - omega * temp) * exp_term
    new_value = target + (change + temp) * exp_term
    if (target - current > 0.0) == (new_value > target):
        new_value = target
        new_velocity = 0.0
    return new_value, new_velocity


@dataclass
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


@dataclass
class SpaceOccupancyState:
    target_x: float = 0.0
    target_y: float = 0.0
    target_alpha: float = 1.0
    target_scale: float = 1.0
    visible: bool = True

    target_z: float = 0.5
    topmost: bool = False
    layer_mode: str = "respect_space"
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

    reason: str = "idle"
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
    alpha_floor: float = 0.12
    alpha_ceiling: float = 1.0

    def as_dict(self) -> Dict[str, float | str | bool | int]:
        return {
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_alpha": self.target_alpha,
            "target_scale": self.target_scale,
            "visible": self.visible,
            "target_z": self.target_z,
            "topmost": self.topmost,
            "layer_mode": self.layer_mode,
            "anchor_x": self.anchor_x,
            "anchor_y": self.anchor_y,
            "current_slot_id": self.current_slot_id,
            "locked_slot_id": self.locked_slot_id,
            "home_slot_id": self.home_slot_id,
            "last_safe_slot_id": self.last_safe_slot_id,
            "last_safe_x": self.last_safe_x,
            "last_safe_y": self.last_safe_y,
            "locked_until": self.locked_until,
            "dodge_strength": self.dodge_strength,
            "slide_x": self.slide_x,
            "slide_y": self.slide_y,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "spring_x": self.spring_x,
            "spring_y": self.spring_y,
            "spring_vx": self.spring_vx,
            "spring_vy": self.spring_vy,
            "reason": self.reason,
            "smooth_time": self.smooth_time,
            "hover_bias": self.hover_bias,
            "overlap_score": self.overlap_score,
            "visibility_score": self.visibility_score,
            "escape_score": self.escape_score,
            "pulse": self.pulse,
            "float_phase": self.float_phase,
            "float_amplitude": self.float_amplitude,
            "slot_stability": self.slot_stability,
            "motion_curvature": self.motion_curvature,
            "alpha_floor": self.alpha_floor,
            "alpha_ceiling": self.alpha_ceiling,
        }


class OccupancyVisibilityTransformer(nn.Module):
    def __init__(self, input_dim: int = 24, hidden_dim: int = 96, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 8),
        )
        self.register_buffer("last_alpha", torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer("last_scale", torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer("last_x", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("last_y", torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer("last_z", torch.tensor(0.5, device=device, dtype=dtype))
        self.register_buffer("last_visibility", torch.tensor(1.0, device=device, dtype=dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] < 24:
            pad = torch.zeros(*features.shape[:-1], 24 - features.shape[-1], device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > 24:
            features = features[..., :24]
        return self.net(features).squeeze(0)

    def decode(self, out: torch.Tensor) -> Dict[str, float]:
        if out.ndim > 1:
            out = out.squeeze(0)
        return {
            "target_x": float(torch.tanh(out[0]).item()),
            "target_y": float(torch.tanh(out[1]).item()),
            "alpha": float(torch.sigmoid(out[2]).item()),
            "scale": float(0.72 + 0.78 * torch.sigmoid(out[3]).item()),
            "dodge": float(torch.sigmoid(out[4]).item()),
            "hover": float(torch.sigmoid(out[5]).item()),
            "z": float(torch.sigmoid(out[6]).item()),
            "visibility": float(torch.sigmoid(out[7]).item()),
        }


class OccupancyVisibilityController(nn.Module):
    def __init__(self, screen_width: float = 1920.0, screen_height: float = 1080.0, avatar_size: float = 900.0, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)
        self.avatar_size = float(avatar_size)
        self.device = device
        self.dtype = dtype
        self.transformer = OccupancyVisibilityTransformer(device=device, dtype=dtype)
        self.state = SpaceOccupancyState(target_x=self.screen_width * 0.60, target_y=self.screen_height * 0.55)
        self.idle_timer = 0.0
        self.float_phase = random.random() * math.tau
        self.last_mouse = (self.screen_width * 0.5, self.screen_height * 0.5)
        self.last_decision: Dict[str, float | str | bool | int] = self.state.as_dict()

    def reset(self) -> None:
        self.state = SpaceOccupancyState(target_x=self.screen_width * 0.60, target_y=self.screen_height * 0.55)
        self.idle_timer = 0.0
        self.float_phase = random.random() * math.tau
        self.last_decision = self.state.as_dict()

    def _rect_area(self, rect: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = rect
        return max(1.0, (x2 - x1) * (y2 - y1))

    def _rect_overlap(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return ((ix2 - ix1) * (iy2 - iy1)) / self._rect_area(a)

    def _distance_to_rect(self, x: float, y: float, rect: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = rect
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        return math.hypot(dx, dy)

    def _predict_rect(self, rect: Tuple[float, float, float, float], vx: float, vy: float, horizon: float = 0.45) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = rect
        shift_x = vx * horizon
        shift_y = vy * horizon
        return (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)

    def _heat_at(self, x: float, y: float, heatmap: Optional[Sequence[Sequence[float]]]) -> float:
        if heatmap is None:
            return 0.0
        rows = len(heatmap)
        if rows == 0:
            return 0.0
        cols = len(heatmap[0]) if heatmap[0] is not None else 0
        if cols == 0:
            return 0.0
        gx = int(clamp(x / max(1.0, self.screen_width) * cols, 0, cols - 1))
        gy = int(clamp(y / max(1.0, self.screen_height) * rows, 0, rows - 1))
        return float(heatmap[gy][gx])

    def _slot_catalog(self, inp: SpaceOccupancyInput) -> List[Tuple[int, str, float, float]]:
        w = self.screen_width
        h = self.screen_height
        s = self.avatar_size
        m = max(8.0, inp.safe_margin)
        mid_x = (w - s) * 0.5
        mid_y = (h - s) * 0.5
        slots = [
            (0, "top_left", m, m),
            (1, "top_right", w - s - m, m),
            (2, "bottom_left", m, h - s - m),
            (3, "bottom_right", w - s - m, h - s - m),
            (4, "top_center", mid_x, m),
            (5, "bottom_center", mid_x, h - s - m),
            (6, "left_center", m, mid_y),
            (7, "right_center", w - s - m, mid_y),
            (8, "center", mid_x, mid_y),
            (9, "upper_left_band", w * 0.20 - s * 0.5, h * 0.16),
            (10, "upper_right_band", w * 0.80 - s * 0.5, h * 0.16),
            (11, "lower_center_band", w * 0.50 - s * 0.5, h * 0.74),
        ]
        out = []
        for sid, name, x, y in slots:
            out.append((sid, name, clamp(x, m, max(m, w - s - m)), clamp(y, m, max(m, h - s - m))))
        return out

    def _slot_rect(self, x: float, y: float) -> Tuple[float, float, float, float]:
        return (x, y, x + self.avatar_size, y + self.avatar_size)

    def _slot_cost(self, slot_id: int, x: float, y: float, inp: SpaceOccupancyInput, decoded: Dict[str, float]) -> float:
        cx = x + self.avatar_size * 0.5
        cy = y + self.avatar_size * 0.5
        rect = self._slot_rect(x, y)
        mouse_dist = math.hypot(cx - inp.mouse_x, cy - inp.mouse_y)
        mouse_pull = 1.0 / (1.0 + mouse_dist / max(120.0, inp.mouse_hot_radius))
        edge_x = min(cx / max(1.0, self.screen_width), 1.0 - cx / max(1.0, self.screen_width))
        edge_y = min(cy / max(1.0, self.screen_height), 1.0 - cy / max(1.0, self.screen_height))
        edge_penalty = 1.0 - clamp(edge_x + edge_y, 0.0, 1.0)
        heat = self._heat_at(cx, cy, inp.heatmap)
        overlap = 0.0
        future_overlap = 0.0
        for candidate in (inp.active_window_rect, inp.dragged_window_rect, inp.fullscreen_window_rect):
            if candidate is None:
                continue
            overlap += self._rect_overlap(rect, candidate)
            future_overlap += self._rect_overlap(rect, self._predict_rect(candidate, inp.window_velocity_x, inp.window_velocity_y))
        if inp.occupied_rects:
            for r in inp.occupied_rects:
                overlap += inp.occupied_rect_bias * self._rect_overlap(rect, r)
        clearance = 0.0
        if inp.active_window_rect:
            clearance += 1.0 / (1.0 + self._distance_to_rect(cx, cy, inp.active_window_rect) / 180.0)
        if inp.dragged_window_rect:
            clearance += 1.0 / (1.0 + self._distance_to_rect(cx, cy, inp.dragged_window_rect) / 180.0)
        if inp.fullscreen_window_rect:
            clearance += 1.0 / (1.0 + self._distance_to_rect(cx, cy, inp.fullscreen_window_rect) / 220.0)
        path_line = abs((cy - inp.mouse_y) * inp.mouse_dx - (cx - inp.mouse_x) * inp.mouse_dy)
        glide = 1.0 / (1.0 + path_line / 220000.0)
        corner_preference = 1.0 if slot_id in {0, 1, 2, 3} else 0.0
        center_preference = 1.0 if slot_id == 8 else 0.0
        band_preference = 1.0 if slot_id in {9, 10, 11} else 0.0
        lock_penalty = 0.0 if slot_id == self.state.locked_slot_id else 0.09
        history_penalty = 0.0 if slot_id != self.state.current_slot_id else -0.11
        idle_bonus = -0.12 * clamp(inp.idle_seconds / 10.0, 0.0, 1.0) * (1.0 if slot_id == 8 else 0.0)
        fullscreen_bonus = -0.15 * inp.fullscreen_pressure * (1.0 if slot_id in {0, 1, 2, 3, 11} else 0.0)
        return (
            2.30 * heat
            + 2.85 * overlap
            + 1.90 * future_overlap
            + 0.70 * inp.occupancy_pressure
            + 0.55 * inp.workspace_density
            + 0.80 * inp.fullscreen_pressure
            + 0.60 * inp.click_pressure
            + 0.42 * inp.silence_pressure
            + 1.00 * mouse_pull
            + 0.36 * edge_penalty
            + 0.20 * clearance
            + 0.14 * glide
            + 0.20 * band_preference
            - 0.24 * corner_preference
            - 0.18 * center_preference * clamp(1.0 - inp.workspace_density, 0.0, 1.0)
            + lock_penalty
            + history_penalty
            + idle_bonus
            + fullscreen_bonus
            + 0.10 * abs(decoded["dodge"] - 0.5)
            - 0.10 * inp.front_bias
            - 0.08 * inp.return_bias
        )

    def _compute_features(self, inp: SpaceOccupancyInput) -> torch.Tensor:
        w = max(1.0, inp.screen_width)
        h = max(1.0, inp.screen_height)
        dx = (inp.mouse_x - inp.avatar_x) / w
        dy = (inp.mouse_y - inp.avatar_y) / h
        dist = math.hypot(inp.mouse_x - inp.avatar_x, inp.mouse_y - inp.avatar_y)
        mouse_pressure = clamp(inp.mouse_pressure + max(0.0, 1.0 - dist / max(120.0, inp.avatar_scale * 0.55)), 0.0, 1.0)
        fullscreen = clamp(inp.fullscreen_pressure, 0.0, 1.0)
        silence = clamp(inp.silence_pressure, 0.0, 1.0)
        workspace = clamp(inp.workspace_density, 0.0, 1.0)
        occupancy = clamp(inp.occupancy_pressure, 0.0, 1.0)
        idle = clamp(inp.idle_seconds / 18.0, 0.0, 1.0)
        focus = 1.0 if inp.focus else 0.0
        aw = (inp.active_window_rect[2] - inp.active_window_rect[0]) / w if inp.active_window_rect else 0.0
        ah = (inp.active_window_rect[3] - inp.active_window_rect[1]) / h if inp.active_window_rect else 0.0
        drag = 1.0 if inp.drag_active or inp.dragged_window_rect is not None else 0.0
        window_count = clamp(inp.window_count / 10.0, 0.0, 1.0)
        return torch.tensor([
            dx, dy,
            inp.mouse_dx / w, inp.mouse_dy / h,
            inp.mouse_speed / 1800.0, mouse_pressure,
            fullscreen, silence, workspace, occupancy, idle, focus,
            inp.avatar_scale / max(1.0, min(w, h)), inp.avatar_alpha, inp.float_bias,
            inp.interaction_intensity, inp.window_velocity_x / w, inp.window_velocity_y / h,
            inp.z_order_hint, 1.0 if inp.talking else 0.0, aw, ah,
            self._heat_at(inp.avatar_x, inp.avatar_y, inp.heatmap),
            self._heat_at(inp.mouse_x, inp.mouse_y, inp.heatmap),
        ], device=self.device, dtype=self.dtype)

    def _choose_slot(self, inp: SpaceOccupancyInput, decoded: Dict[str, float], now: float) -> Tuple[int, float, float, float]:
        slots = self._slot_catalog(inp)
        scored = []
        for sid, _name, x, y in slots:
            scored.append((self._slot_cost(sid, x, y, inp, decoded), sid, x, y))
        scored.sort(key=lambda item: item[0])
        best_cost, best_sid, best_x, best_y = scored[0]

        current_rect = self._slot_rect(self.state.target_x, self.state.target_y)
        overlap_now = 0.0
        for candidate in (inp.active_window_rect, inp.dragged_window_rect, inp.fullscreen_window_rect):
            if candidate is not None:
                overlap_now = max(overlap_now, self._rect_overlap(current_rect, candidate))

        can_keep = (
            now < self.state.locked_until
            and self.state.locked_slot_id != 0
            and self.state.current_slot_id == self.state.locked_slot_id
            and overlap_now < 0.04
            and inp.workspace_density < 0.82
        )
        if can_keep:
            for sid, _name, x, y in slots:
                if sid == self.state.locked_slot_id:
                    return sid, x, y, best_cost

        if inp.allow_return_to_last_safe and inp.idle_seconds > 6.0 and inp.fullscreen_pressure < 0.35 and inp.occupancy_pressure < 0.30:
            for sid, _name, x, y in slots:
                if sid == self.state.last_safe_slot_id:
                    best_sid, best_x, best_y = sid, x, y
                    break

        if inp.fullscreen_pressure >= 0.55:
            best_full = None
            best_full_cost = float("inf")
            for sid, _name, x, y in slots:
                c = self._slot_cost(sid, x, y, inp, decoded)
                if c < best_full_cost:
                    best_full_cost = c
                    best_full = (sid, x, y)
            if best_full is not None:
                best_sid, best_x, best_y = best_full

        return best_sid, best_x, best_y, best_cost

    def step(self, inp: SpaceOccupancyInput, dt: float = 0.016) -> SpaceOccupancyState:
        dt = max(1e-4, float(dt))
        now = float(self.idle_timer + dt)
        self.idle_timer = now
        raw = self.transformer(self._compute_features(inp))
        decoded = self.transformer.decode(raw)

        mouse_dist = math.hypot(inp.mouse_x - inp.avatar_x, inp.mouse_y - inp.avatar_y)
        mouse_pressure = clamp(inp.mouse_pressure + max(0.0, 1.0 - mouse_dist / max(1.0, inp.avatar_scale * 0.70)), 0.0, 1.0)
        occupancy_pressure = clamp(inp.occupancy_pressure + inp.workspace_density * 0.55, 0.0, 1.0)
        active_pressure = clamp(
            0.36 * mouse_pressure + 0.20 * inp.click_pressure + 0.18 * occupancy_pressure + 0.18 * inp.fullscreen_pressure + 0.12 * inp.silence_pressure + 0.10 * inp.interaction_intensity,
            0.0,
            1.0,
        )
        if not inp.focus:
            active_pressure = min(1.0, active_pressure + 0.12)
        if inp.idle_seconds > 8.0:
            active_pressure = max(0.0, active_pressure - 0.16)

        slot_id, slot_x, slot_y, best_score = self._choose_slot(inp, decoded, now)
        slot_rect = self._slot_rect(slot_x, slot_y)
        overlap_current = 0.0
        for candidate in (inp.active_window_rect, inp.dragged_window_rect, inp.fullscreen_window_rect):
            if candidate is not None:
                overlap_current = max(overlap_current, self._rect_overlap(slot_rect, candidate))

        if overlap_current > 0.02 or inp.drag_active or (inp.window_velocity_x * inp.window_velocity_x + inp.window_velocity_y * inp.window_velocity_y) > 1e-6:
            self.state.locked_slot_id = slot_id
            self.state.locked_until = now + max(0.55, inp.hold_lock_seconds)
            self.state.last_safe_slot_id = slot_id
            self.state.last_safe_x = slot_x
            self.state.last_safe_y = slot_y

        reason = "wander"
        if inp.fullscreen_pressure > 0.60:
            reason = "fullscreen_overlay"
        elif overlap_current > 0.04 or inp.drag_active or inp.dragged_window_rect is not None:
            reason = "window_escape"
        elif mouse_pressure > 0.58 or inp.click_pressure > 0.45:
            reason = "mouse_avoid"
        elif inp.workspace_density > 0.60:
            reason = "workspace_compact"
        elif inp.idle_seconds > 10.0:
            reason = "float_front"
        elif inp.silence_pressure > 0.50:
            reason = "quiet_float"

        if reason == "window_escape" and len(self._slot_catalog(inp)) > 1:
            # escolhe o slot menos exposto entre os 3 melhores
            slots = self._slot_catalog(inp)
            scored = sorted([(self._slot_cost(sid, x, y, inp, decoded), sid, x, y) for sid, _name, x, y in slots], key=lambda item: item[0])
            best_sid, slot_x, slot_y = scored[min(2, len(scored) - 1)][1:]
            slot_id = best_sid
            best_score = scored[min(2, len(scored) - 1)][0]

        if reason == "fullscreen_overlay":
            # Em ecrã cheio, mantém-se visível sem cobrir o centro crítico
            best_score = min(best_score, 0.5)

        self.float_phase += dt * (0.40 + 0.90 * clamp(inp.idle_seconds / 8.0, 0.0, 1.0) + 0.30 * inp.front_bias)
        if inp.mouse_speed > 700.0 and mouse_pressure > 0.25:
            self.float_phase += dt * 1.6

        float_strength = 0.0
        if inp.idle_seconds > 3.0 and inp.focus:
            float_strength = clamp((inp.idle_seconds - 3.0) / 12.0, 0.0, 1.0)
        orbit_x = math.sin(self.float_phase * 0.75) * (4.0 + 8.0 * float_strength)
        orbit_y = math.cos(self.float_phase * 0.62) * (3.0 + 6.0 * float_strength)

        target_x = slot_x + orbit_x
        target_y = slot_y + orbit_y
        target_scale = clamp(decoded["scale"] * (0.94 + 0.10 * (1.0 - active_pressure) + 0.04 * float_strength), 0.70, 1.60)
        target_scale += 0.03 * math.sin(self.float_phase * 1.4)
        target_scale = clamp(target_scale, 0.70, 1.60)

        # alpha suave por sigmóide
        visibility_curve = sigmoid((1.0 - active_pressure - 0.35) * 5.0)
        target_alpha = decoded["alpha"] * (0.38 + 0.62 * visibility_curve)
        target_alpha += 0.10 * clamp(inp.front_bias, 0.0, 1.0) * (1.0 - active_pressure)
        target_alpha += 0.06 * clamp(inp.attention_pressure, 0.0, 1.0)
        if reason == "fullscreen_overlay":
            target_alpha = min(target_alpha, 0.52)
            if inp.talking or inp.interaction_intensity > 0.25:
                pulse = 0.5 + 0.5 * math.sin(self.float_phase * 2.2)
                target_alpha = max(target_alpha, 0.72 + 0.08 * pulse)
        if inp.cursor_locked:
            target_alpha = max(target_alpha, 0.82)
        elif reason in {"mouse_avoid", "window_escape"}:
            target_alpha *= 0.74
        elif reason == "workspace_compact":
            target_alpha *= 0.84
        elif reason in {"quiet_float", "float_front"}:
            target_alpha *= 0.94
        target_alpha = clamp(target_alpha, 0.08 if active_pressure > 0.65 else 0.18, 1.0)

        # Z-order / camada
        if reason == "fullscreen_overlay":
            target_z = 0.88
            topmost = True
            layer_mode = "layered_interaction"
        elif reason in {"mouse_avoid", "window_escape", "workspace_compact"}:
            target_z = 0.18
            topmost = False
            layer_mode = "ghost_dodge"
        elif reason in {"quiet_float", "float_front"}:
            target_z = 0.70
            topmost = True
            layer_mode = "float_front"
        else:
            target_z = 0.52
            topmost = bool(inp.focus)
            layer_mode = "respect_space"

        if inp.fullscreen_pressure > 0.65 and (inp.talking or inp.interaction_intensity > 0.20):
            target_alpha = max(target_alpha, 0.78)
            topmost = True
            layer_mode = "layered_interaction"
            target_z = max(target_z, 0.90)

        # Movimento em mola: rápido ao sair e suave ao estacionar
        smooth_time = 0.15 if reason in {"mouse_avoid", "window_escape"} else 0.35
        self.state.spring_x, self.state.spring_vx = smooth_damp(self.state.spring_x, target_x, self.state.spring_vx, smooth_time, dt)
        self.state.spring_y, self.state.spring_vy = smooth_damp(self.state.spring_y, target_y, self.state.spring_vy, smooth_time, dt)
        target_x = self.state.spring_x + orbit_x * 0.12
        target_y = self.state.spring_y + orbit_y * 0.12

        slide_x = clamp((target_x - inp.avatar_x) / max(1.0, inp.screen_width) * 1.95, -0.68, 0.68)
        slide_y = clamp((target_y - inp.avatar_y) / max(1.0, inp.screen_height) * 1.95, -0.68, 0.68)
        if reason in {"mouse_avoid", "window_escape"}:
            away_x = inp.avatar_x - inp.mouse_x
            away_y = inp.avatar_y - inp.mouse_y
            norm = math.hypot(away_x, away_y) or 1.0
            slide_x += (away_x / norm) * 0.18 * active_pressure
            slide_y += (away_y / norm) * 0.18 * active_pressure

        self.state.target_x = exp_smooth(self.state.target_x, target_x, dt, smooth_time)
        self.state.target_y = exp_smooth(self.state.target_y, target_y, dt, smooth_time)
        self.state.anchor_x = exp_smooth(self.state.anchor_x, target_x + 0.5 * self.avatar_size, dt, smooth_time)
        self.state.anchor_y = exp_smooth(self.state.anchor_y, target_y + 0.5 * self.avatar_size, dt, smooth_time)
        self.state.target_alpha = exp_smooth(self.state.target_alpha, target_alpha, dt, 0.14 if target_alpha < self.state.target_alpha else 0.20)
        self.state.target_scale = exp_smooth(self.state.target_scale, target_scale, dt, 0.18)
        self.state.visible = self.state.target_alpha > 0.10
        self.state.topmost = topmost
        self.state.target_z = exp_smooth(self.state.target_z, target_z, dt, 0.20)
        self.state.layer_mode = layer_mode
        self.state.current_slot_id = slot_id
        self.state.dodge_strength = exp_smooth(self.state.dodge_strength, active_pressure, dt, 0.18)
        self.state.slide_x = exp_smooth(self.state.slide_x, slide_x, dt, 0.18)
        self.state.slide_y = exp_smooth(self.state.slide_y, slide_y, dt, 0.18)
        self.state.velocity_x = exp_smooth(self.state.velocity_x, (self.state.target_x - inp.avatar_x) / max(dt, 1e-3), dt, 0.25)
        self.state.velocity_y = exp_smooth(self.state.velocity_y, (self.state.target_y - inp.avatar_y) / max(dt, 1e-3), dt, 0.25)
        self.state.reason = reason
        self.state.smooth_time = smooth_time
        self.state.hover_bias = exp_smooth(self.state.hover_bias, float(decoded["hover"]), dt, 0.30)
        self.state.float_amplitude = exp_smooth(self.state.float_amplitude, float_strength, dt, 0.22)
        self.state.pulse = exp_smooth(self.state.pulse, 0.5 + 0.5 * inp.interaction_intensity, dt, 0.14)
        self.state.overlap_score = float(best_score)
        self.state.visibility_score = clamp(1.0 - self.state.overlap_score, 0.0, 1.0)
        self.state.escape_score = clamp(active_pressure + (1.0 - self.state.visibility_score), 0.0, 1.0)
        self.state.slot_stability = exp_smooth(self.state.slot_stability, 1.0 - min(1.0, abs(self.state.target_x - target_x) / max(1.0, self.avatar_size)), dt, 0.30)
        self.state.motion_curvature = exp_smooth(self.state.motion_curvature, abs(orbit_x) + abs(orbit_y), dt, 0.30)
        self.state.alpha_floor = 0.08 if reason == "fullscreen_overlay" else 0.12
        self.state.alpha_ceiling = 1.0

        if self.state.reason in {"float_front", "fullscreen_overlay"} or (inp.idle_seconds > 6.0 and inp.focus and inp.workspace_density < 0.50):
            self.state.last_safe_slot_id = self.state.current_slot_id
            self.state.last_safe_x = self.state.target_x
            self.state.last_safe_y = self.state.target_y

        self.last_decision = self.state.as_dict()
        self.last_mouse = (inp.mouse_x, inp.mouse_y)
        self.transformer.last_alpha.copy_(torch.tensor(self.state.target_alpha, device=self.device, dtype=self.dtype))
        self.transformer.last_scale.copy_(torch.tensor(self.state.target_scale, device=self.device, dtype=self.dtype))
        self.transformer.last_x.copy_(torch.tensor(self.state.target_x, device=self.device, dtype=self.dtype))
        self.transformer.last_y.copy_(torch.tensor(self.state.target_y, device=self.device, dtype=self.dtype))
        self.transformer.last_z.copy_(torch.tensor(self.state.target_z, device=self.device, dtype=self.dtype))
        self.transformer.last_visibility.copy_(torch.tensor(self.state.visibility_score, device=self.device, dtype=self.dtype))
        return self.state


__all__ = ["SpaceOccupancyInput", "SpaceOccupancyState", "OccupancyVisibilityTransformer", "OccupancyVisibilityController"]

try:
    from avatar_orchestrator_experts import patch_all as _patch_all_experts
    _patch_all_experts(globals())
except Exception:
    pass
