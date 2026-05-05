
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def lerp(a: float, b: float, t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-8:
        return target
    k = 1.0 - math.exp(-max(0.0, dt) / tau)
    return current + (target - current) * k


def critically_damped_step(current: float, target: float, velocity: float, dt: float, omega: float = 10.0) -> Tuple[float, float]:
    dt = max(1e-5, float(dt))
    omega = max(1e-3, float(omega))
    f = 1.0 + 2.0 * dt * omega
    oo = omega * omega
    hoo = dt * oo
    hhoo = dt * hoo
    det_inv = 1.0 / (f + hhoo)
    new_current = (f * current + dt * velocity + hhoo * target) * det_inv
    new_velocity = (velocity + hoo * (target - current)) * det_inv
    return new_current, new_velocity


def vec_len(x: float, y: float) -> float:
    return math.hypot(x, y)


def normalize(x: float, y: float) -> Tuple[float, float]:
    n = math.hypot(x, y)
    if n < 1e-9:
        return 0.0, 0.0
    return x / n, y / n


def ease_curve(t: float, mode: str = "smooth") -> float:
    t = clamp(t, 0.0, 1.0)
    if mode == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if mode == "smoother":
        return smootherstep(0.0, 1.0, t)
    if mode == "in_out":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    return t


@dataclass
class AnimationContext:
    t: float = 0.0
    dt: float = 0.016
    state_name: str = "idle"
    emotion: str = "neutral"
    intensity: float = 0.25
    arousal: float = 0.25
    energy: float = 0.25
    tension: float = 0.10
    curiosity: float = 0.20
    confidence: float = 0.50
    attention: float = 0.50
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    mouse_distance: float = 9999.0
    focus: bool = True
    speaking: bool = False
    silence_s: float = 0.0
    audio_energy: float = 0.0
    workspace_density: float = 0.0
    fullscreen_pressure: float = 0.0
    drag_pressure: float = 0.0
    interaction_intensity: float = 0.0
    occupied_rects: list = field(default_factory=list)
    active_window_rect: Optional[Tuple[float, float, float, float]] = None
    dragged_window_rect: Optional[Tuple[float, float, float, float]] = None
    fullscreen_window_rect: Optional[Tuple[float, float, float, float]] = None
    screen_width: float = 1280.0
    screen_height: float = 720.0
    avatar_x: float = 0.0
    avatar_y: float = 0.0
    avatar_scale: float = 1.0
    avatar_alpha: float = 1.0
    preferred_slot_index: int = -1
    allow_return_to_last_safe: bool = True
    hold_lock_seconds: float = 0.85
    return_bias: float = 0.35
    occupied_rect_bias: float = 0.60
    workspace_focus_bias: float = 0.25
    front_bias: float = 0.45
    target_rect: Optional[Tuple[float, float, float, float]] = None
    force_folder_enter: bool = False
    folder_rect: Optional[Tuple[float, float, float, float]] = None
    folder_progress: float = 0.0


@dataclass
class AnimationOutput:
    body_scale_x: float = 1.0
    body_scale_y: float = 1.0
    lean_x: float = 0.0
    lean_y: float = 0.0
    rotation: float = 0.0
    head_x: float = 0.0
    head_y: float = 0.0
    head_rot: float = 0.0
    alpha: float = 1.0
    z_order: float = 0.0
    secondary_x: float = 0.0
    secondary_y: float = 0.0
    squash: float = 0.0
    stretch: float = 0.0
    anticipation: float = 0.0
    recoil: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    slot_index: int = 0
    slot_lock: bool = False
    folder_progress: float = 0.0
    folder_mode: bool = False
    motion_quality: float = 1.0
    float_bias: float = 0.0
    no_tremor: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["animation_meta"] = data.pop("meta", {})
        return data


class FixedSlotNavigator:
    def __init__(self):
        self.locked_slot = -1
        self.lock_until = 0.0
        self.last_safe_slot = -1
        self.last_target = (0.0, 0.0)

    @staticmethod
    def _slot_positions(screen_w: float, screen_h: float) -> List[Tuple[float, float]]:
        w = max(640.0, float(screen_w))
        h = max(360.0, float(screen_h))
        return [
            (w * 0.16, h * 0.80),
            (w * 0.84, h * 0.80),
            (w * 0.16, h * 0.18),
            (w * 0.84, h * 0.18),
            (w * 0.50, h * 0.16),
            (w * 0.50, h * 0.84),
            (w * 0.12, h * 0.50),
            (w * 0.88, h * 0.50),
            (w * 0.50, h * 0.50),
        ]

    @staticmethod
    def _rect_area(rect: Tuple[float, float, float, float]) -> float:
        x, y, w, h = rect
        return max(0.0, w) * max(0.0, h)

    @staticmethod
    def _overlap_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def choose(self, ctx: AnimationContext, avatar_size: float) -> Tuple[float, float, int, float]:
        now = ctx.t
        slots = self._slot_positions(ctx.screen_width, ctx.screen_height)
        avatar_size = max(1.0, float(avatar_size))
        best_i = self.locked_slot if (self.locked_slot >= 0 and now < self.lock_until) else -1
        best_score = 1e18
        if best_i < 0:
            for i, (sx, sy) in enumerate(slots):
                score = 0.0
                # favor stable positions and visible corners when workspace is dense
                if i == self.last_safe_slot:
                    score -= 0.35 * ctx.return_bias
                if i in (0, 1, 2, 3):
                    score -= 0.08 * (1.0 - ctx.workspace_density)
                # avoid the center when windows dominate, prefer the center when workspace is idle
                dist_center = vec_len(sx - ctx.screen_width * 0.5, sy - ctx.screen_height * 0.5) / max(ctx.screen_width, ctx.screen_height)
                score += dist_center * (0.4 + 0.6 * ctx.workspace_density)
                # keep away from mouse and active windows
                score += vec_len(sx - ctx.mouse_x, sy - ctx.mouse_y) / max(ctx.screen_width, ctx.screen_height) * 0.8
                if ctx.active_window_rect:
                    overlap = self._overlap_area((sx - avatar_size * 0.5, sy - avatar_size * 0.5, avatar_size, avatar_size), ctx.active_window_rect)
                    score += overlap / max(1.0, self._rect_area(ctx.active_window_rect)) * 1.5
                if ctx.dragged_window_rect:
                    overlap = self._overlap_area((sx - avatar_size * 0.5, sy - avatar_size * 0.5, avatar_size, avatar_size), ctx.dragged_window_rect)
                    score += overlap / max(1.0, self._rect_area(ctx.dragged_window_rect)) * 2.0
                if ctx.fullscreen_window_rect:
                    # when fullscreen, move to corner/edge with minimal visual density
                    fx, fy, fw, fh = ctx.fullscreen_window_rect
                    inside = self._overlap_area((sx - avatar_size * 0.5, sy - avatar_size * 0.5, avatar_size, avatar_size), ctx.fullscreen_window_rect)
                    score += inside / max(1.0, fw * fh) * 2.4
                # prefer upper corners slightly when user is actively using the screen
                score += (0.12 if i in (2, 3) and ctx.interaction_intensity > 0.45 else 0.0)
                if score < best_score:
                    best_score = score
                    best_i = i
            self.locked_slot = best_i
            self.lock_until = now + max(0.35, ctx.hold_lock_seconds)
            self.last_safe_slot = best_i
        x, y = slots[best_i]
        self.last_target = (x, y)
        return x, y, best_i, clamp(1.0 - best_score * 0.25, 0.0, 1.0)


class SecondaryMotionTransformer:
    def __init__(self):
        self.secondary_x = 0.0
        self.secondary_y = 0.0
        self.secondary_rot = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_rot = 0.0

    def step(self, base_dx: float, base_dy: float, base_rot: float, dt: float, tension: float, energy: float) -> Tuple[float, float, float]:
        # deliberately smooth and low-frequency; no tremor
        follower = 5.5 + 2.0 * clamp(energy, 0.0, 1.0)
        damp = 0.84 + 0.08 * (1.0 - clamp(tension, 0.0, 1.0))
        tx = base_dx * 0.42
        ty = base_dy * 0.42
        tr = base_rot * 0.28
        self.secondary_x = exp_smooth(self.secondary_x, tx, dt, 1.0 / follower)
        self.secondary_y = exp_smooth(self.secondary_y, ty, dt, 1.0 / follower)
        self.secondary_rot = exp_smooth(self.secondary_rot, tr, dt, 1.0 / follower)
        self.secondary_x *= damp
        self.secondary_y *= damp
        self.secondary_rot *= damp
        return self.secondary_x, self.secondary_y, self.secondary_rot


class MicroLifeTransformer:
    def __init__(self):
        self.phase = random.random() * math.tau
        self.phase2 = random.random() * math.tau
        self.phase3 = random.random() * math.tau
        self.smoothed = 0.0

    def step(self, t: float, dt: float, tension: float, energy: float, speaking: bool) -> float:
        # low amplitude, smooth, non-shaky life signal
        tension = clamp(tension, 0.0, 1.0)
        energy = clamp(energy, 0.0, 1.0)
        speed = 0.22 + 0.34 * energy + 0.22 * tension
        if speaking:
            speed += 0.10
        wave = (
            0.65 * math.sin(t * (0.62 + speed) + self.phase) +
            0.25 * math.sin(t * (0.31 + speed * 0.62) + self.phase2) +
            0.10 * math.sin(t * (0.17 + speed * 0.33) + self.phase3)
        )
        target = wave * (0.004 + 0.008 * tension + 0.004 * energy)
        self.smoothed = exp_smooth(self.smoothed, target, dt, 0.18)
        return self.smoothed


class SquashStretchTransformer:
    def __init__(self):
        self.squash_x = 1.0
        self.squash_y = 1.0

    def step(self, vx: float, vy: float, speed: float, dt: float) -> Tuple[float, float]:
        # fast downward motion -> squash; upward/rapid exit -> stretch
        accel = clamp(speed / 240.0, 0.0, 1.5)
        vertical = clamp(vy / 180.0, -1.0, 1.0)
        target_x = 1.0 + 0.05 * accel - 0.03 * abs(vertical)
        target_y = 1.0 - 0.06 * accel + 0.05 * abs(vertical)
        if vy < -40.0:
            target_y += 0.05
            target_x -= 0.02
        elif vy > 40.0:
            target_y -= 0.04
            target_x += 0.01
        self.squash_x = exp_smooth(self.squash_x, clamp(target_x, 0.88, 1.18), dt, 0.10)
        self.squash_y = exp_smooth(self.squash_y, clamp(target_y, 0.82, 1.16), dt, 0.10)
        return self.squash_x, self.squash_y


class AnticipationRecoilTransformer:
    def __init__(self):
        self.state = 0.0
        self.target = 0.0
        self.phase = 0.0
        self.recoil = 0.0

    def trigger(self, strength: float = 1.0) -> None:
        self.target = clamp(strength, 0.0, 1.5)
        self.state = max(self.state, self.target)

    def step(self, dt: float, active: bool, target_changed: bool, arrival: float = 0.0) -> Tuple[float, float]:
        if target_changed:
            self.trigger(1.0)
            self.phase = 0.0
        self.phase += dt
        envelope = math.exp(-self.phase * 7.5)
        if active:
            anticipation = clamp((1.0 - smoothstep(0.0, 0.18, self.phase)) * self.state, 0.0, 1.0)
        else:
            anticipation = 0.0
        recoil_target = clamp(arrival * 0.85, 0.0, 1.0)
        self.recoil = exp_smooth(self.recoil, recoil_target * envelope, dt, 0.14)
        if not active and envelope < 0.02:
            self.state = 0.0
        return anticipation, self.recoil


class FolderTransitionTransformer:
    def __init__(self):
        self.active = False
        self.progress = 0.0
        self.duration = 0.95
        self.source = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.rotation_turns = 2.0
        self.alpha_min = 0.08
        self.scale_min = 0.12
        self.hold = 0.0

    def start(self, source: Tuple[float, float], target: Tuple[float, float], duration: float = 0.95, rotation_turns: float = 2.0) -> None:
        self.active = True
        self.progress = 0.0
        self.duration = max(0.18, float(duration))
        self.source = (float(source[0]), float(source[1]))
        self.target = (float(target[0]), float(target[1]))
        self.rotation_turns = float(rotation_turns)

    def step(self, dt: float) -> AnimationOutput:
        if not self.active:
            return AnimationOutput(folder_mode=False, folder_progress=0.0, alpha=1.0, motion_quality=1.0)
        self.progress = min(1.0, self.progress + max(0.0, dt) / self.duration)
        t = ease_curve(self.progress, "smoother")
        x = lerp(self.source[0], self.target[0], t)
        y = lerp(self.source[1], self.target[1], t)
        scale = lerp(1.0, self.scale_min, t)
        rotation = 360.0 * self.rotation_turns * t
        alpha = lerp(1.0, self.alpha_min, t)
        if self.progress >= 1.0:
            self.active = False
        return AnimationOutput(
            body_scale_x=scale,
            body_scale_y=scale,
            lean_x=0.0,
            lean_y=0.0,
            rotation=rotation,
            head_x=0.0,
            head_y=0.0,
            head_rot=rotation * 0.25,
            alpha=alpha,
            z_order=1.0,
            target_x=x,
            target_y=y,
            folder_progress=t,
            folder_mode=True,
            motion_quality=clamp(1.0 - 0.35 * t, 0.0, 1.0),
            float_bias=0.4,
            no_tremor=1.0,
            meta={"mode": "folder_enter", "source": self.source, "target": self.target}
        )


class AdvancedAnimationController:
    def __init__(self):
        self.screen_w = 1280.0
        self.screen_h = 720.0
        self.avatar_size = 320.0
        self.slot_navigator = FixedSlotNavigator()
        self.secondary = SecondaryMotionTransformer()
        self.micro = MicroLifeTransformer()
        self.squash = SquashStretchTransformer()
        self.anticipation = AnticipationRecoilTransformer()
        self.folder = FolderTransitionTransformer()
        self.last_target = (0.0, 0.0)
        self.last_mode = "idle"
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_alpha = 1.0
        self.current_scale = 1.0
        self.current_rotation = 0.0
        self.current_lean_x = 0.0
        self.current_lean_y = 0.0
        self.current_head_x = 0.0
        self.current_head_y = 0.0
        self.current_head_rot = 0.0
        self.current_slot = 0
        self.slot_lock_until = 0.0
        self.last_safe_pos = (0.0, 0.0)
        self.last_retarget_t = 0.0
        self.safe_mode = True
        self.last_idle_front_t = 0.0
        self.follow_target = (0.0, 0.0)
        self.vx = 0.0
        self.vy = 0.0
        self.requested_folder = None

    def set_workspace_metrics(self, screen_width: float, screen_height: float, avatar_size: float) -> None:
        self.screen_w = max(640.0, float(screen_width))
        self.screen_h = max(360.0, float(screen_height))
        self.avatar_size = max(64.0, float(avatar_size))

    def request_folder_enter(self, folder_rect: Tuple[float, float, float, float], source: Optional[Tuple[float, float]] = None, duration: float = 0.95) -> None:
        cx = folder_rect[0] + folder_rect[2] * 0.5
        cy = folder_rect[1] + folder_rect[3] * 0.5
        if source is None:
            source = self.last_target
        self.folder.start(source, (cx, cy), duration=duration, rotation_turns=2.0)
        self.requested_folder = folder_rect

    @staticmethod
    def _is_fullscreen(rect: Optional[Tuple[float, float, float, float]], w: float, h: float) -> bool:
        if not rect:
            return False
        _, _, rw, rh = rect
        return rw >= w * 0.95 and rh >= h * 0.95

    def _build_context(self, perception: Any, emotion: Any, behavior: Any, attention: Any, plan: Any, runtime_meta: Optional[Dict[str, Any]] = None) -> AnimationContext:
        state_name = getattr(behavior, "current_state", None) or (runtime_meta or {}).get("state_name", "idle")
        mouse_x = float(getattr(perception, "mouse_x", self.screen_w * 0.5))
        mouse_y = float(getattr(perception, "mouse_y", self.screen_h * 0.5))
        mouse_dx = float(getattr(perception, "mouse_dx", 0.0))
        mouse_dy = float(getattr(perception, "mouse_dy", 0.0))
        mouse_speed = float(getattr(perception, "mouse_speed", 0.0))
        mouse_distance = float(getattr(perception, "mouse_distance_to_face", 9999.0))
        focus = bool(getattr(perception, "focus", True))
        silence = float(getattr(perception, "silence_s", 0.0))
        emotion_name = getattr(emotion, "emotion", "neutral")
        intensity = float(getattr(emotion, "intensity", 0.25))
        arousal = float(getattr(emotion, "arousal", 0.25))
        energy = float(getattr(emotion, "energy", 0.25))
        tension = float(getattr(emotion, "tension", 0.10))
        curiosity = float(getattr(emotion, "curiosity", 0.20))
        confidence = float(getattr(emotion, "confidence", 0.50))
        attention_v = float(getattr(emotion, "attention", 0.50))
        speaking = bool(getattr(perception, "speaking", False))
        active_window_rect = getattr(perception, "active_window_rect", None)
        dragged_window_rect = getattr(perception, "dragged_window_rect", None)
        fullscreen_window_rect = getattr(perception, "fullscreen_window_rect", None)
        workspace_density = float(getattr(perception, "workspace_density", 0.0))
        fullscreen_pressure = float(getattr(perception, "fullscreen_pressure", 1.0 if self._is_fullscreen(fullscreen_window_rect, self.screen_w, self.screen_h) else 0.0))
        drag_pressure = float(getattr(perception, "drag_pressure", min(1.0, mouse_speed / 1500.0)))
        interaction_intensity = float(getattr(perception, "interaction_intensity", 0.0))
        return AnimationContext(
            t=float(getattr(perception, "t", time.time())),
            dt=float(getattr(perception, "dt", 0.016)),
            state_name=state_name,
            emotion=emotion_name,
            intensity=intensity,
            arousal=arousal,
            energy=energy,
            tension=tension,
            curiosity=curiosity,
            confidence=confidence,
            attention=attention_v,
            mouse_x=mouse_x,
            mouse_y=mouse_y,
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            mouse_speed=mouse_speed,
            mouse_distance=mouse_distance,
            focus=focus,
            speaking=speaking,
            silence_s=silence,
            audio_energy=float(getattr(perception, "audio_rms", 0.0)),
            workspace_density=clamp(workspace_density, 0.0, 1.0),
            fullscreen_pressure=clamp(fullscreen_pressure, 0.0, 1.0),
            drag_pressure=clamp(drag_pressure, 0.0, 1.0),
            interaction_intensity=clamp(interaction_intensity, 0.0, 1.0),
            occupied_rects=list(getattr(perception, "occupied_rects", []) or []),
            active_window_rect=active_window_rect,
            dragged_window_rect=dragged_window_rect,
            fullscreen_window_rect=fullscreen_window_rect,
            screen_width=self.screen_w,
            screen_height=self.screen_h,
            avatar_x=float(getattr(perception, "avatar_x", self.current_x)),
            avatar_y=float(getattr(perception, "avatar_y", self.current_y)),
            avatar_scale=float(getattr(perception, "avatar_scale", self.current_scale)),
            avatar_alpha=float(getattr(perception, "avatar_alpha", self.current_alpha)),
            preferred_slot_index=int(getattr(perception, "preferred_slot_index", -1)),
            allow_return_to_last_safe=bool(getattr(perception, "allow_return_to_last_safe", True)),
            hold_lock_seconds=float(getattr(perception, "hold_lock_seconds", 0.85)),
            return_bias=float(getattr(perception, "return_bias", 0.35)),
            occupied_rect_bias=float(getattr(perception, "occupied_rect_bias", 0.60)),
            workspace_focus_bias=float(getattr(perception, "workspace_focus_bias", 0.25)),
            front_bias=float(getattr(perception, "front_bias", 0.45)),
            target_rect=active_window_rect,
            force_folder_enter=bool(getattr(perception, "force_folder_enter", False)),
            folder_rect=getattr(perception, "folder_rect", None),
            folder_progress=float(getattr(perception, "folder_progress", 0.0)),
        )

    def _window_pressure(self, ctx: AnimationContext) -> float:
        pressure = 0.10 + 0.35 * ctx.workspace_density + 0.18 * ctx.drag_pressure + 0.15 * ctx.fullscreen_pressure
        if ctx.mouse_speed > 600.0:
            pressure += 0.10
        if not ctx.focus:
            pressure += 0.15
        if ctx.silence_s > 10.0:
            pressure -= 0.06
        return clamp(pressure, 0.0, 1.0)

    def _should_float_front(self, ctx: AnimationContext) -> bool:
        idle = ctx.silence_s > 8.0 and ctx.mouse_speed < 35.0 and not ctx.speaking and ctx.focus
        full = ctx.fullscreen_pressure > 0.85 and ctx.interaction_intensity < 0.35
        return idle or full

    def step(self, ctx: AnimationContext) -> AnimationOutput:
        if ctx.force_folder_enter and ctx.folder_rect:
            if not self.folder.active:
                self.request_folder_enter(ctx.folder_rect, source=self.follow_target, duration=max(0.72, 0.95 - 0.18 * ctx.intensity))
        # slot selection
        avatar_size = max(96.0, self.avatar_size * (0.20 + 0.10 * ctx.avatar_scale))
        target_x, target_y, slot_idx, stability = self.slot_navigator.choose(ctx, avatar_size)

        # front/visibility decisions
        pressure = self._window_pressure(ctx)
        float_front = self._should_float_front(ctx)
        z_order = 0.65 + 0.30 * pressure + (0.08 if float_front else 0.0)
        alpha_goal = 1.0
        if ctx.fullscreen_pressure > 0.8:
            alpha_goal = 0.36 + 0.10 * ctx.front_bias
        elif pressure > 0.55:
            alpha_goal = 0.56 - 0.12 * pressure
        else:
            alpha_goal = 0.86 + 0.10 * ctx.front_bias
        if ctx.drag_pressure > 0.55 or ctx.mouse_speed > 450.0:
            alpha_goal = min(alpha_goal, 0.70)
        if ctx.interaction_intensity > 0.6:
            alpha_goal = max(alpha_goal, 0.80)

        # if user is idle, bubble slightly to front with safe visibility
        if float_front:
            alpha_goal = max(alpha_goal, 0.86)
            z_order = max(z_order, 0.90)

        # focus on a stable slot, not moving back and forth every frame
        if vec_len(target_x - self.current_x, target_y - self.current_y) > 1.0:
            self.follow_target = (target_x, target_y)

        # spring dynamics toward the target
        tx = target_x
        ty = target_y
        self.current_x, self.vx = critically_damped_step(self.current_x, tx, self.vx, ctx.dt, omega=7.5)
        self.current_y, self.vy = critically_damped_step(self.current_y, ty, self.vy, ctx.dt, omega=7.5)
        speed = vec_len(self.vx, self.vy)

        # anticipation and recoil envelope
        target_changed = vec_len(tx - self.last_target[0], ty - self.last_target[1]) > max(24.0, self.avatar_size * 0.04)
        anticipation, recoil = self.anticipation.step(ctx.dt, active=(speed > 0.6), target_changed=target_changed, arrival=float(float_front))
        self.last_target = (tx, ty)

        # the visible motion should be smooth and non-jittery
        secondary_x, secondary_y, secondary_rot = self.secondary.step(self.vx, self.vy, self.current_rotation if hasattr(self, "current_rotation") else 0.0, ctx.dt, ctx.tension, ctx.energy)
        squash_x, squash_y = self.squash.step(self.vx, self.vy, speed, ctx.dt)
        micro = self.micro.step(ctx.t, ctx.dt, ctx.tension, ctx.energy, ctx.speaking)

        # folder transition has priority, but remains smooth
        folder_out = self.folder.step(ctx.dt)
        if folder_out.folder_mode:
            alpha_goal = min(alpha_goal, folder_out.alpha)
            target_x, target_y = folder_out.target_x, folder_out.target_y
            tx = target_x
            ty = target_y
            self.current_x, self.vx = critically_damped_step(self.current_x, tx, self.vx, ctx.dt, omega=9.5)
            self.current_y, self.vy = critically_damped_step(self.current_y, ty, self.vy, ctx.dt, omega=9.5)
            speed = vec_len(self.vx, self.vy)
            self.current_scale = exp_smooth(self.current_scale, folder_out.body_scale_x, ctx.dt, 0.12)
            self.current_rotation = exp_smooth(self.current_rotation, folder_out.rotation, ctx.dt, 0.14)
            return AnimationOutput(
                body_scale_x=self.current_scale,
                body_scale_y=self.current_scale,
                lean_x=0.0,
                lean_y=0.0,
                rotation=self.current_rotation,
                head_x=0.0,
                head_y=0.0,
                head_rot=self.current_rotation * 0.25,
                alpha=alpha_goal,
                z_order=z_order,
                secondary_x=0.0,
                secondary_y=0.0,
                squash=1.0,
                stretch=1.0,
                anticipation=anticipation,
                recoil=recoil,
                target_x=self.current_x,
                target_y=self.current_y,
                slot_index=slot_idx,
                slot_lock=True,
                folder_progress=folder_out.folder_progress,
                folder_mode=True,
                motion_quality=folder_out.motion_quality,
                float_bias=0.5 if float_front else 0.0,
                no_tremor=1.0,
                meta=folder_out.meta,
            )

        # base body transform: stable, quiet and physically plausible
        scale_goal = 1.0
        if ctx.focus is False:
            scale_goal *= 0.96
        if ctx.fullscreen_pressure > 0.5:
            scale_goal *= 0.92 + 0.05 * ctx.front_bias
        if ctx.state_name in {"shy", "hide", "retreat"}:
            scale_goal *= 0.90
        if ctx.state_name in {"curious", "lean_in"}:
            scale_goal *= 1.05
        if ctx.speaking:
            scale_goal *= 1.02
        scale_goal *= squash_x
        scale_goal *= squash_y
        scale_goal = clamp(scale_goal, 0.72, 1.48)

        # posture/rotation lean toward target, but softened by spring
        dx = (tx - ctx.screen_width * 0.5) / max(1.0, ctx.screen_width * 0.5)
        dy = (ty - ctx.screen_height * 0.5) / max(1.0, ctx.screen_height * 0.5)
        lean_x = clamp(dx * 5.0 + self.current_x * 0.0015, -18.0, 18.0)
        lean_y = clamp(dy * 4.2 + self.current_y * 0.0015, -18.0, 18.0)
        rotation = clamp(dx * 2.2 + secondary_rot, -14.0, 14.0)
        head_x = clamp(lean_x * 0.18 + secondary_x * 0.35 + micro * 26.0, -8.0, 8.0)
        head_y = clamp(lean_y * 0.18 + secondary_y * 0.35 + micro * 18.0, -8.0, 8.0)
        head_rot = clamp(rotation * 0.22 + secondary_rot * 0.28, -10.0, 10.0)

        body_scale_x = exp_smooth(self.current_scale, scale_goal, ctx.dt, 0.11)
        body_scale_y = exp_smooth(self.current_scale, scale_goal * (1.0 + 0.01 * abs(self.vy) / 160.0), ctx.dt, 0.11)
        alpha = exp_smooth(self.current_alpha, clamp(alpha_goal, 0.08, 1.0), ctx.dt, 0.16)
        self.current_alpha = alpha
        self.current_scale = body_scale_x

        # if user is interacting or moving a window, keep Denise out of the way and stable
        if ctx.drag_pressure > 0.25 or ctx.mouse_speed > 180.0:
            alpha = min(alpha, 0.72 + 0.10 * ctx.front_bias)
        if ctx.interaction_intensity > 0.50:
            alpha = max(alpha, 0.84)
        if ctx.fullscreen_pressure > 0.88 and ctx.speaking:
            alpha = max(alpha, 0.82)
        if ctx.fullscreen_pressure > 0.95 and not ctx.speaking:
            alpha = min(alpha, 0.48 + 0.16 * ctx.front_bias)

        # apply a gentle, non-shaky movement profile
        motion_quality = clamp(1.0 - 0.18 * ctx.tension + 0.12 * ctx.energy - 0.08 * pressure, 0.0, 1.0)
        no_tremor = 1.0 - clamp(0.60 * abs(micro) + 0.18 * ctx.tension, 0.0, 0.70)

        return AnimationOutput(
            body_scale_x=body_scale_x,
            body_scale_y=body_scale_y,
            lean_x=lean_x,
            lean_y=lean_y,
            rotation=rotation,
            head_x=head_x,
            head_y=head_y,
            head_rot=head_rot,
            alpha=alpha,
            z_order=z_order,
            secondary_x=secondary_x,
            secondary_y=secondary_y,
            squash=squash_x,
            stretch=squash_y,
            anticipation=anticipation,
            recoil=recoil,
            target_x=self.current_x,
            target_y=self.current_y,
            slot_index=slot_idx,
            slot_lock=True,
            folder_progress=0.0,
            folder_mode=False,
            motion_quality=motion_quality,
            float_bias=1.0 if float_front else 0.0,
            no_tremor=no_tremor,
            meta={
                "mode": "float_front" if float_front else "respect_space",
                "slot_index": slot_idx,
                "slot_lock_until": self.slot_navigator.lock_until,
                "pressure": pressure,
                "workspace_density": ctx.workspace_density,
                "fullscreen_pressure": ctx.fullscreen_pressure,
                "drag_pressure": ctx.drag_pressure,
                "secondary_motion": {"x": secondary_x, "y": secondary_y, "rot": secondary_rot},
            },
        )


def _apply_runtime_motion_smoothing(engine: Any, motion_data: Dict[str, Any], ctx: AnimationContext, anim: AnimationOutput) -> Dict[str, Any]:
    # prevent visible jitter by gently filtering body/face values after the original engine update
    body = getattr(engine, "body", None)
    face = getattr(engine, "face", None)
    if body is not None:
        try:
            body.scale_x = exp_smooth(float(getattr(body, "scale_x", 1.0)), anim.body_scale_x, ctx.dt, 0.08)
            body.scale_y = exp_smooth(float(getattr(body, "scale_y", 1.0)), anim.body_scale_y, ctx.dt, 0.08)
            body.lean_x = exp_smooth(float(getattr(body, "lean_x", 0.0)), anim.lean_x, ctx.dt, 0.08)
            body.lean_y = exp_smooth(float(getattr(body, "lean_y", 0.0)), anim.lean_y, ctx.dt, 0.08)
            body.rotation = exp_smooth(float(getattr(body, "rotation", 0.0)), anim.rotation, ctx.dt, 0.10)
            body.head_x = exp_smooth(float(getattr(body, "head_x", 0.0)), anim.head_x, ctx.dt, 0.08)
            body.head_y = exp_smooth(float(getattr(body, "head_y", 0.0)), anim.head_y, ctx.dt, 0.08)
            body.head_rot = exp_smooth(float(getattr(body, "head_rot", 0.0)), anim.head_rot, ctx.dt, 0.08)
        except Exception:
            pass
    if face is not None:
        try:
            # keep mouth and eye-related features calm, without micro tremor
            face.face_energy = exp_smooth(float(getattr(face, "face_energy", 0.5)), clamp(0.55 + 0.20 * ctx.energy + 0.10 * ctx.intensity, 0.0, 1.0), ctx.dt, 0.16)
            face.micro_smile = exp_smooth(float(getattr(face, "micro_smile", 0.0)), clamp(0.10 * ctx.intensity + 0.06 * ctx.curiosity, -0.2, 0.4), ctx.dt, 0.20)
        except Exception:
            pass

    motion_data = dict(motion_data or {})
    motion_data.setdefault("motion_meta", {})
    motion_data["animation_meta"] = anim.as_dict()
    motion_data["animation_meta"]["animation_meta"] = motion_data["animation_meta"].get("animation_meta", {})
    motion_data["motion_meta"] = {**motion_data.get("motion_meta", {}), **anim.as_dict()}
    motion_data["space_alpha"] = clamp(min(float(motion_data.get("space_alpha", 1.0)), float(anim.alpha)), 0.0, 1.0)
    motion_data["space_target_z"] = float(motion_data.get("space_target_z", anim.z_order))
    motion_data["space_allow_return_to_last_safe"] = bool(motion_data.get("space_allow_return_to_last_safe", True))
    motion_data["space_hold_lock_seconds"] = float(motion_data.get("space_hold_lock_seconds", 0.85))
    motion_data["space_preferred_slot_index"] = int(anim.slot_index)
    motion_data["space_front_bias"] = float(anim.float_bias)
    motion_data["space_no_tremor"] = float(anim.no_tremor)
    return motion_data


def _extract_runtime_context(args: tuple, kwargs: dict) -> Tuple[Any, Any, Any, Any, Any]:
    # runtime: update(self, perception, emotion, behavior, attention, plan)
    perception = kwargs.get("perception", args[1] if len(args) > 1 else None)
    emotion = kwargs.get("emotion", args[2] if len(args) > 2 else None)
    behavior = kwargs.get("behavior", args[3] if len(args) > 3 else None)
    attention = kwargs.get("attention", args[4] if len(args) > 4 else None)
    plan = kwargs.get("plan", args[5] if len(args) > 5 else None)
    return perception, emotion, behavior, attention, plan


def _extract_core_context(args: tuple, kwargs: dict) -> Tuple[Any, Any, Any, Any, Any]:
    # core: forward(self, body, emotion, attention, dt, mode, spatial_features=None, workspace_features=None)
    body = kwargs.get("body", args[1] if len(args) > 1 else None)
    emotion = kwargs.get("emotion", args[2] if len(args) > 2 else None)
    attention = kwargs.get("attention", args[3] if len(args) > 3 else None)
    dt = kwargs.get("dt", args[4] if len(args) > 4 else None)
    mode = kwargs.get("mode", args[5] if len(args) > 5 else None)
    return body, emotion, attention, dt, mode


def install_advanced_animation_transformers(module_globals: Optional[Dict[str, Any]] = None) -> None:
    if not module_globals:
        return

    MotionEngine = module_globals.get("MotionEngine")
    AvatarApp = module_globals.get("AvatarApp")
    AvatarAppV2 = module_globals.get("AvatarAppV2")
    AvatarRenderer = module_globals.get("AvatarRenderer")

    def _install_motion_engine(cls: Any, *, core_style: bool = False) -> None:
        if cls is None or getattr(cls, "_advanced_animation_installed", False):
            return

        orig_init = getattr(cls, "__init__", None)
        orig_update = getattr(cls, "update", None)
        orig_forward = getattr(cls, "forward", None)

        if orig_init is not None:
            def __init__(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                self.advanced_animation = AdvancedAnimationController()
                # reasoned default; can be overwritten by AvatarApp after init
                screen_w = float(module_globals.get("DEFAULT_SCREEN_WIDTH", 1280.0))
                screen_h = float(module_globals.get("DEFAULT_SCREEN_HEIGHT", 720.0))
                avatar_size = float(module_globals.get("DEFAULT_AVATAR_SIZE", 320.0))
                try:
                    self.advanced_animation.set_workspace_metrics(screen_w, screen_h, avatar_size)
                except Exception:
                    pass
                try:
                    # some engines store config objects or spatial cfg
                    if hasattr(self, "config") and hasattr(self.config, "screen_width"):
                        self.advanced_animation.set_workspace_metrics(float(self.config.screen_width), float(self.config.screen_height), float(getattr(self.config, "avatar_size", avatar_size)))
                except Exception:
                    pass
            cls.__init__ = __init__

        if orig_update is not None:
            def update(self, *args, **kwargs):
                result = orig_update(self, *args, **kwargs)
                try:
                    perception, emotion, behavior, attention, plan = _extract_runtime_context(args, kwargs)
                    if perception is None or emotion is None:
                        return result
                    anim = getattr(self, "advanced_animation", None)
                    if anim is None:
                        anim = AdvancedAnimationController()
                        self.advanced_animation = anim
                    # workspace metrics: prefer values from perception if present, else default
                    sw = float(getattr(perception, "screen_width", anim.screen_w))
                    sh = float(getattr(perception, "screen_height", anim.screen_h))
                    avatar_size = float(getattr(perception, "avatar_scale", 1.0) * getattr(perception, "avatar_size", anim.avatar_size))
                    anim.set_workspace_metrics(sw, sh, max(128.0, avatar_size if avatar_size > 2.0 else anim.avatar_size))
                    ctx = anim._build_context(perception, emotion, behavior, attention, plan, runtime_meta=result.get("motion_meta", {}) if isinstance(result, dict) else None)
                    # folder enter trigger if an explicit signal is present
                    runtime_meta = result.get("motion_meta", {}) if isinstance(result, dict) else {}
                    folder_rect = runtime_meta.get("space_folder_rect") or runtime_meta.get("space_dragged_window_rect") or getattr(perception, "folder_rect", None)
                    folder_enter = bool(runtime_meta.get("folder_enter", False) or runtime_meta.get("space_force_folder_enter", False))
                    if folder_enter and folder_rect:
                        anim.request_folder_enter(folder_rect, source=(anim.current_x, anim.current_y), duration=float(runtime_meta.get("folder_enter_duration", 0.95)))
                    anim_out = anim.step(ctx)
                    if isinstance(result, dict):
                        result = _apply_runtime_motion_smoothing(self, result, ctx, anim_out)
                    return result
                except Exception:
                    return result

            cls.update = update

        if orig_forward is not None:
            def forward(self, *args, **kwargs):
                result = orig_forward(self, *args, **kwargs)
                # core style smoothing over tensor outputs
                try:
                    anim = getattr(self, "advanced_animation", None)
                    if anim is None:
                        anim = AdvancedAnimationController()
                        self.advanced_animation = anim
                    body, emotion, attention, dt, mode = _extract_core_context(args, kwargs)
                    if body is None or emotion is None or attention is None or dt is None or mode is None:
                        return result
                    # build a lightweight context from tensors
                    try:
                        emo = emotion.detach().cpu().tolist() if hasattr(emotion, "detach") else list(emotion)
                    except Exception:
                        emo = [0.0] * 6
                    try:
                        att = attention.detach().cpu().tolist() if hasattr(attention, "detach") else list(attention)
                    except Exception:
                        att = [0.0] * 4
                    ctx = AnimationContext(
                        t=float(kwargs.get("t", time.time())),
                        dt=float(dt.item() if hasattr(dt, "item") else float(dt)),
                        state_name=str(mode.value if hasattr(mode, "value") else mode),
                        emotion="neutral",
                        intensity=float(max(0.0, min(1.0, abs(float(emo[0])) if emo else 0.25))),
                        arousal=float(max(0.0, min(1.0, abs(float(emo[1])) if len(emo) > 1 else 0.25))),
                        energy=float(max(0.0, min(1.0, abs(float(emo[5])) if len(emo) > 5 else 0.25))),
                        tension=float(max(0.0, min(1.0, abs(float(emo[4])) if len(emo) > 4 else 0.10))),
                        curiosity=float(max(0.0, min(1.0, abs(float(emo[2])) if len(emo) > 2 else 0.20))),
                        confidence=float(max(0.0, min(1.0, abs(float(emo[3])) if len(emo) > 3 else 0.50))),
                        attention=float(max(0.0, min(1.0, float(att[0]) if att else 0.50))),
                        mouse_x=anim.current_x,
                        mouse_y=anim.current_y,
                        mouse_speed=0.0,
                        mouse_distance=9999.0,
                        focus=True,
                        speaking=str(mode).lower() == "speaking",
                        silence_s=0.0,
                        audio_energy=0.0,
                        workspace_density=0.0,
                        fullscreen_pressure=0.0,
                        drag_pressure=0.0,
                        interaction_intensity=0.0,
                        screen_width=anim.screen_w,
                        screen_height=anim.screen_h,
                        avatar_scale=float(body[2].item() if hasattr(body[2], "item") else body[2]) if hasattr(body, "__len__") and len(body) > 2 else 1.0,
                        avatar_alpha=1.0,
                    )
                    anim_out = anim.step(ctx)
                    if hasattr(body, "__len__") and len(body) >= 6:
                        try:
                            result = result.clone()
                        except Exception:
                            pass
                    return result
                except Exception:
                    return result

            cls.forward = forward

        cls._advanced_animation_installed = True

    _install_motion_engine(MotionEngine, core_style=bool(module_globals.get("AvatarTransformerBrain")))
    # patch main app constructors to propagate actual workspace metrics
    for app_cls in (AvatarApp, AvatarAppV2):
        if app_cls is None or getattr(app_cls, "_advanced_animation_app_installed", False):
            continue
        orig_init = getattr(app_cls, "__init__", None)
        if orig_init is None:
            continue

        def __init__(self, *args, __orig_init=orig_init, **kwargs):
            __orig_init(self, *args, **kwargs)
            try:
                if hasattr(self, "motion") and hasattr(self.motion, "advanced_animation"):
                    sw = float(getattr(getattr(self, "space_controller", None), "screen_width", getattr(self.root, "winfo_screenwidth", lambda: 1280)()))
                    sh = float(getattr(getattr(self, "space_controller", None), "screen_height", getattr(self.root, "winfo_screenheight", lambda: 720)()))
                    avatar_size = float(getattr(self, "display_res", 320))
                    self.motion.advanced_animation.set_workspace_metrics(sw, sh, avatar_size)
            except Exception:
                pass
        app_cls.__init__ = __init__
        app_cls._advanced_animation_app_installed = True

    # renderer patch: keep a stable front overlay when animation_meta is present
    if AvatarRenderer is not None and not getattr(AvatarRenderer, "_advanced_animation_renderer_installed", False):
        orig_render = getattr(AvatarRenderer, "render", None)
        if orig_render is not None:
            def render(self, *args, **kwargs):
                motion_data = kwargs.get("motion_data", None)
                if motion_data is None and len(args) >= 8:
                    motion_data = args[7]
                frame = orig_render(self, *args, **kwargs)
                # The main runtime already handles alpha for the workspace; we leave pixels untouched here.
                return frame
            AvatarRenderer.render = render
        AvatarRenderer._advanced_animation_renderer_installed = True


__all__ = [
    "AnimationContext",
    "AnimationOutput",
    "AdvancedAnimationController",
    "SecondaryMotionTransformer",
    "MicroLifeTransformer",
    "SquashStretchTransformer",
    "AnticipationRecoilTransformer",
    "FolderTransitionTransformer",
    "FixedSlotNavigator",
    "install_advanced_animation_transformers",
]
