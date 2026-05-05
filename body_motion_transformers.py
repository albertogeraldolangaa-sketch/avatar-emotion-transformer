from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


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
    k = 1.0 - math.exp(-dt / tau)
    return lerp(current, target, k)


def soft_noise(seed: float, t: float, freq: float = 1.0) -> float:
    return (
        math.sin(t * freq + seed) * 0.52
        + math.sin(t * freq * 1.71 + seed * 1.37) * 0.31
        + math.sin(t * freq * 2.63 + seed * 0.73) * 0.17
    )


@dataclass
class BodyMotionConfig:
    screen_w: int = 1024
    screen_h: int = 1024
    slot_margin: int = 46
    slot_lock_s: float = 0.72
    slot_release_s: float = 0.20
    drift_tau: float = 0.20
    spring_stiffness: float = 9.0
    spring_damping: float = 0.84
    repel_radius: float = 220.0
    click_safe_radius: float = 170.0
    fullscreen_alpha: float = 0.42
    normal_alpha: float = 0.98
    min_alpha: float = 0.10
    max_alpha: float = 1.00
    max_scale: float = 1.42
    min_scale: float = 0.72
    max_rotation_deg: float = 12.0
    max_head_rotation_deg: float = 18.0
    intent_smooth_tau: float = 0.18
    trajectory_tau: float = 0.20
    focus_hold_threshold: float = 0.012


@dataclass
class BodyMotionContext:
    t: float = 0.0
    dt: float = 0.016
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    focus: bool = True
    silence_s: float = 0.0
    speaking: bool = False
    audio_energy: float = 0.0
    emotion_intensity: float = 0.2
    emotion_energy: float = 0.4
    curiosity: float = 0.2
    tension: float = 0.1
    confidence: float = 0.5
    attention: float = 0.5
    workspace_density: float = 0.0
    fullscreen_pressure: float = 0.0
    click_pressure: float = 0.0
    double_click_pressure: float = 0.0
    interaction_intensity: float = 0.0
    occupied_rects: List[Tuple[float, float, float, float]] = field(default_factory=list)
    preferred_slots: List[Tuple[float, float]] = field(default_factory=list)
    current_pos: Tuple[float, float] = (0.0, 0.0)
    last_safe_pos: Tuple[float, float] = (0.0, 0.0)
    window_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1024.0, 1024.0)
    target_hint: Optional[Tuple[float, float]] = None
    intent_hint: str = "idle"


@dataclass
class BodyMotionState:
    target_pos: Tuple[float, float] = (0.0, 0.0)
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    alpha: float = 1.0
    scale: float = 1.0
    rotation_z: float = 0.0
    head_rotation: float = 0.0
    head_latency: float = 0.12
    squash: float = 0.0
    stretch: float = 0.0
    secondary_motion: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    click_safe: float = 1.0
    z_order_hint: float = 0.5
    motion_blend: float = 0.0
    lock_timer: float = 0.0
    last_intent: str = "idle"


class IntentTransformer:
    """Converte contexto em impulso de intenção estável."""

    def __init__(self, config: Optional[BodyMotionConfig] = None):
        self.config = config or BodyMotionConfig()
        self.intent = {"approach": 0.15, "avoid": 0.15, "focus": 0.25, "idle": 0.45}

    def forward(self, ctx: BodyMotionContext) -> Dict[str, float]:
        mouse_close = clamp(1.0 - ctx.mouse_speed / 1400.0, 0.0, 1.0)
        overlap_pressure = clamp(ctx.workspace_density + 0.6 * ctx.click_pressure + 0.35 * ctx.double_click_pressure + 0.25 * ctx.fullscreen_pressure, 0.0, 1.0)
        avoid = clamp(0.18 + 0.38 * mouse_close + 0.28 * overlap_pressure + 0.18 * ctx.tension - 0.12 * ctx.confidence, 0.0, 1.0)
        approach = clamp(0.18 + 0.28 * ctx.curiosity + 0.24 * ctx.attention + 0.12 * ctx.speaking, 0.0, 1.0)
        focus = clamp(0.28 + 0.32 * ctx.attention + 0.18 * ctx.audio_energy + 0.10 * ctx.emotion_intensity, 0.0, 1.0)
        idle = clamp(0.42 + 0.16 * ctx.silence_s / 18.0 - 0.18 * ctx.speaking - 0.10 * ctx.interaction_intensity, 0.0, 1.0)
        total = max(1e-6, avoid + approach + focus + idle)
        target = {"approach": approach / total, "avoid": avoid / total, "focus": focus / total, "idle": idle / total}
        for k, v in target.items():
            self.intent[k] = exp_smooth(self.intent[k], v, ctx.dt, self.config.intent_smooth_tau)
        return dict(self.intent)


class MotionPlanningTransformer:
    """Escolhe um destino estável em slots seguros do ecrã."""

    def __init__(self, config: Optional[BodyMotionConfig] = None):
        self.config = config or BodyMotionConfig()
        self.locked_slot: Optional[Tuple[float, float]] = None
        self.lock_timer = 0.0
        self.last_target = (0.0, 0.0)
        self._slots = self._build_slots()

    def _build_slots(self) -> List[Tuple[float, float]]:
        w, h = float(self.config.screen_w), float(self.config.screen_h)
        m = float(self.config.slot_margin)
        return [
            (m, m), (w * 0.5, m), (w - m, m),
            (m, h * 0.5), (w * 0.5, h * 0.5), (w - m, h * 0.5),
            (m, h - m), (w * 0.5, h - m), (w - m, h - m),
            (w * 0.20, h * 0.18), (w * 0.80, h * 0.18), (w * 0.20, h * 0.82), (w * 0.80, h * 0.82),
        ]

    def _rect_distance(self, x: float, y: float, rect: Tuple[float, float, float, float]) -> float:
        rx, ry, rw, rh = rect
        cx = clamp(x, rx, rx + rw)
        cy = clamp(y, ry, ry + rh)
        return math.hypot(x - cx, y - cy)

    def _score_slot(self, slot: Tuple[float, float], ctx: BodyMotionContext, intent: Dict[str, float]) -> float:
        x, y = slot
        dx = x - ctx.mouse_x
        dy = y - ctx.mouse_y
        dist_mouse = math.hypot(dx, dy)
        center_x = (self.config.screen_w * 0.5)
        center_y = (self.config.screen_h * 0.5)
        dist_center = math.hypot(x - center_x, y - center_y)
        overlap_cost = 0.0
        for rect in ctx.occupied_rects:
            overlap_cost += max(0.0, 1.0 - self._rect_distance(x, y, rect) / 260.0)
        if ctx.fullscreen_pressure > 0.5:
            # prefer corners when the screen is visually full
            corner_bonus = 1.0 - min(abs(x - center_x) / center_x, 1.0) * min(abs(y - center_y) / center_y, 1.0)
        else:
            corner_bonus = 0.0
        focus_bias = 1.0 if ctx.focus else 0.85
        return (
            1.45 * dist_mouse / max(self.config.screen_w, self.config.screen_h)
            + 0.42 * dist_center / max(self.config.screen_w, self.config.screen_h)
            - 0.55 * intent.get("focus", 0.0)
            - 0.38 * intent.get("approach", 0.0)
            + 0.72 * intent.get("avoid", 0.0)
            + 0.58 * overlap_cost
            - 0.20 * corner_bonus
            + (0.0 if ctx.focus else 0.12)
            + (0.06 if focus_bias < 1.0 else 0.0)
        )

    def choose_target(self, ctx: BodyMotionContext, intent: Dict[str, float]) -> Tuple[float, float]:
        if self.lock_timer > 0.0 and self.locked_slot is not None:
            self.lock_timer = max(0.0, self.lock_timer - ctx.dt)
            self.last_target = self.locked_slot
            return self.last_target

        ranked = sorted(self._slots, key=lambda slot: self._score_slot(slot, ctx, intent), reverse=True)
        best = ranked[0]
        if self.locked_slot is None or math.hypot(best[0] - self.locked_slot[0], best[1] - self.locked_slot[1]) > 20.0:
            self.locked_slot = best
            self.lock_timer = self.config.slot_lock_s
        self.last_target = best
        return best


class TrajectoryTransformer:
    """Gera trajetória suave, com easing e micro-overshoot controlado."""

    def __init__(self, config: Optional[BodyMotionConfig] = None):
        self.config = config or BodyMotionConfig()
        self._phase = 0.0

    def step(self, current: Tuple[float, float], target: Tuple[float, float], velocity: Tuple[float, float], dt: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        cx, cy = current
        tx, ty = target
        vx, vy = velocity
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)
        if dist < self.config.focus_hold_threshold:
            return (target[0], target[1]), (0.0, 0.0)
        self._phase += dt * 2.2
        progress = smoothstep(0.0, 1.0, clamp(dist / 240.0, 0.0, 1.0))
        overshoot = 0.035 * math.sin(self._phase * math.tau)
        acc_x = dx * self.config.spring_stiffness * (0.50 + 0.50 * progress)
        acc_y = dy * self.config.spring_stiffness * (0.50 + 0.50 * progress)
        vx = (vx + acc_x * dt) * self.config.spring_damping
        vy = (vy + acc_y * dt) * self.config.spring_damping
        nx = cx + vx * dt * (1.0 + overshoot)
        ny = cy + vy * dt * (1.0 - overshoot * 0.4)
        return (nx, ny), (vx, vy)


class PhysicsTransformer:
    """Aplica massa, inércia e amortecimento sem tremor."""

    def __init__(self, config: Optional[BodyMotionConfig] = None):
        self.config = config or BodyMotionConfig()
        self._body_rotation = 0.0
        self._head_rotation = 0.0
        self._scale = 1.0
        self._alpha = 1.0
        self._secondary = 0.0

    def update(self, ctx: BodyMotionContext, intent: Dict[str, float], position: Tuple[float, float], target: Tuple[float, float], velocity: Tuple[float, float]) -> BodyMotionState:
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        speed = math.hypot(velocity[0], velocity[1])
        accel = math.hypot(dx, dy)
        is_fullscreen = clamp(ctx.fullscreen_pressure, 0.0, 1.0)
        mouse_panic = clamp(1.0 - ctx.mouse_speed / 1300.0, 0.0, 1.0)
        overlap = clamp(ctx.workspace_density + 0.55 * ctx.click_pressure + 0.25 * ctx.double_click_pressure + 0.25 * is_fullscreen, 0.0, 1.0)
        click_safe = clamp(1.0 - (0.42 * overlap + 0.22 * mouse_panic), 0.0, 1.0)

        target_scale = clamp(
            self.config.min_scale
            + (self.config.max_scale - self.config.min_scale)
            * clamp(0.24 + 0.42 * ctx.emotion_intensity + 0.18 * ctx.curiosity + 0.14 * ctx.audio_energy - 0.18 * overlap, 0.0, 1.0),
            self.config.min_scale,
            self.config.max_scale,
        )
        if ctx.speaking:
            target_scale = min(self.config.max_scale, target_scale + 0.03)
        if intent.get("avoid", 0.0) > intent.get("approach", 0.0):
            target_scale = max(self.config.min_scale, target_scale - 0.02)

        target_alpha = clamp(
            self.config.normal_alpha
            - 0.42 * overlap
            - 0.18 * intent.get("avoid", 0.0)
            - 0.08 * clamp(ctx.silence_s / 18.0, 0.0, 1.0)
            + 0.20 * intent.get("focus", 0.0)
            + (0.10 if ctx.speaking else 0.0),
            self.config.min_alpha,
            self.config.max_alpha,
        )
        if is_fullscreen > 0.65:
            target_alpha = min(target_alpha, self.config.fullscreen_alpha)

        # Rotation: eyes arrive first, head follows with latency
        eye_drive = clamp(0.18 + 0.42 * ctx.attention + 0.18 * ctx.curiosity + 0.16 * intent.get("focus", 0.0), 0.0, 1.0)
        body_rot_target = clamp(
            6.0 * math.tanh(dx / 220.0) + 2.2 * math.tanh(velocity[0] / 140.0) + 1.2 * math.sin(ctx.t * 0.25),
            -self.config.max_rotation_deg,
            self.config.max_rotation_deg,
        )
        if intent.get("avoid", 0.0) > 0.55:
            body_rot_target += 2.5 * math.copysign(1.0, dx if abs(dx) > 1e-6 else 1.0)
        head_rot_target = clamp(
            body_rot_target * 0.62 + 1.5 * math.tanh(dy / 240.0) + 0.8 * math.sin(ctx.t * 0.35),
            -self.config.max_head_rotation_deg,
            self.config.max_head_rotation_deg,
        )
        secondary_target = clamp(
            0.18 * speed / 320.0 + 0.10 * ctx.emotion_energy + 0.08 * ctx.tension,
            0.0,
            1.0,
        )
        squash_target = clamp(max(0.0, -dy / 220.0) * 0.45 + 0.10 * speed / 260.0, 0.0, 1.0)
        stretch_target = clamp(max(0.0, dy / 220.0) * 0.42 + 0.08 * accel / 220.0, 0.0, 1.0)

        self._scale = exp_smooth(self._scale, target_scale, ctx.dt, 0.16)
        self._alpha = exp_smooth(self._alpha, target_alpha, ctx.dt, 0.18)
        self._body_rotation = exp_smooth(self._body_rotation, body_rot_target, ctx.dt, 0.17)
        self._head_rotation = exp_smooth(self._head_rotation, head_rot_target, ctx.dt, 0.20)
        self._secondary = exp_smooth(self._secondary, secondary_target, ctx.dt, 0.22)

        offset_x = clamp(dx / 220.0, -1.0, 1.0) * 0.10 * intent.get("avoid", 0.0)
        offset_y = clamp(dy / 220.0, -1.0, 1.0) * 0.10 * intent.get("avoid", 0.0)
        z_hint = clamp(0.20 + 0.35 * intent.get("focus", 0.0) + 0.18 * intent.get("approach", 0.0) + 0.15 * (1.0 - overlap), 0.0, 1.0)
        motion_blend = clamp(0.26 + 0.42 * eye_drive + 0.12 * speed / 200.0, 0.0, 1.0)

        return BodyMotionState(
            target_pos=target,
            position=position,
            velocity=velocity,
            alpha=self._alpha,
            scale=self._scale,
            rotation_z=self._body_rotation,
            head_rotation=self._head_rotation,
            head_latency=0.10 + 0.08 * (1.0 - eye_drive),
            squash=squash_target,
            stretch=stretch_target,
            secondary_motion=self._secondary,
            offset_x=offset_x,
            offset_y=offset_y,
            click_safe=click_safe,
            z_order_hint=z_hint,
            motion_blend=motion_blend,
            lock_timer=0.0,
            last_intent=max(intent, key=intent.get),
        )


class BodyRotationTransformer:
    """Composição de intenção, planejamento e física para rotação corporal suave."""

    def __init__(self, config: Optional[BodyMotionConfig] = None):
        self.config = config or BodyMotionConfig()
        self.intent = IntentTransformer(self.config)
        self.planner = MotionPlanningTransformer(self.config)
        self.trajectory = TrajectoryTransformer(self.config)
        self.physics = PhysicsTransformer(self.config)
        self.position = (self.config.screen_w * 0.50, self.config.screen_h * 0.54)
        self.velocity = (0.0, 0.0)
        self.last_state = BodyMotionState(position=self.position)

    @staticmethod
    def _occupied_from_context(ctx: BodyMotionContext) -> List[Tuple[float, float, float, float]]:
        return list(ctx.occupied_rects or [])

    def update(self, ctx: BodyMotionContext) -> BodyMotionState:
        ctx = BodyMotionContext(**{**ctx.__dict__})
        ctx.occupied_rects = self._occupied_from_context(ctx)
        intents = self.intent.forward(ctx)
        # Prefer a stable target; allow explicit hints from the caller.
        if ctx.target_hint is not None:
            target = ctx.target_hint
        else:
            target = self.planner.choose_target(ctx, intents)
        self.position, self.velocity = self.trajectory.step(self.position, target, self.velocity, ctx.dt)
        state = self.physics.update(ctx, intents, self.position, target, self.velocity)
        # Keep a memory of a safe position to return to after distraction.
        if state.click_safe > 0.65 and state.alpha > 0.72:
            self.last_state = state
            self.last_state.position = self.position
        return state


# Optional OS helper: make a Tk window click-through on Windows.
def set_tk_click_through(root: Any, enable: bool = True) -> bool:
    try:
        if sys.platform != "win32":
            return False
        import ctypes
        from ctypes import wintypes

        hwnd = wintypes.HWND(root.winfo_id())
        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_LAYERED = 0x00080000
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enable:
            style |= (WS_EX_TRANSPARENT | WS_EX_LAYERED)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            return True
        style &= ~WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        return True
    except Exception:
        return False


__all__ = [
    "BodyMotionConfig",
    "BodyMotionContext",
    "BodyMotionState",
    "BodyRotationTransformer",
    "IntentTransformer",
    "MotionPlanningTransformer",
    "TrajectoryTransformer",
    "PhysicsTransformer",
    "set_tk_click_through",
]
