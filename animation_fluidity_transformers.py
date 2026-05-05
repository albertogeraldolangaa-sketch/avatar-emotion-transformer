from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
import math
import random

import torch
import torch.nn as nn

DTYPE = torch.float32
DEVICE = torch.device('cpu')


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-dt / tau)
    return lerp(current, target, k)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
class AnimationFluidityInput:
    dt: float = 0.016
    state_name: str = 'idle'
    mode_name: str = 'idle'
    intensity: float = 0.25
    energy: float = 0.35
    arousal: float = 0.25
    curiosity: float = 0.20
    tension: float = 0.10
    attention: float = 0.45
    silence: float = 0.0
    speaking: bool = False
    gesture_strength: float = 0.0
    motion_speed: float = 0.35
    mouse_speed: float = 0.0
    mouse_pressure: float = 0.0
    drag_pressure: float = 0.0
    fullscreen_pressure: float = 0.0
    workspace_density: float = 0.0
    occupancy_pressure: float = 0.0
    visibility_alpha: float = 1.0
    body_scale: float = 1.0
    body_lean_x: float = 0.0
    body_lean_y: float = 0.0
    head_x: float = 0.0
    head_y: float = 0.0
    rotation: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.5
    target_alpha: float = 1.0
    transition_bias: float = 0.5
    anticipation: float = 0.0
    recoil: float = 0.0
    allow_micro_motion: bool = True


@dataclass
class AnimationFluidityState:
    secondary_x: float = 0.0
    secondary_y: float = 0.0
    secondary_rot: float = 0.0
    micro_x: float = 0.0
    micro_y: float = 0.0
    micro_rot: float = 0.0
    squash: float = 1.0
    stretch: float = 1.0
    scale: float = 1.0
    alpha: float = 1.0
    z_order: float = 0.5
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_scale: float = 0.0
    phase: float = 0.0
    transition: float = 0.0
    fluid_gate: float = 1.0
    last_target_x: float = 0.0
    last_target_y: float = 0.0
    last_state: str = 'idle'
    last_action: str = 'none'


class SecondaryMotionTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer('state', torch.zeros(3, device=device, dtype=dtype))

    def forward(self, velocity_x: float, velocity_y: float, rotation: float, dt: float, tension: float, energy: float) -> Tuple[float, float, float]:
        lag = 0.18 + 0.22 * clamp(tension, 0.0, 1.0)
        target_x = -velocity_x * (0.14 + 0.12 * energy)
        target_y = -velocity_y * (0.15 + 0.11 * energy)
        target_rot = -rotation * (0.06 + 0.03 * tension)
        sx = exp_smooth(float(self.state[0].item()), target_x, dt, lag)
        sy = exp_smooth(float(self.state[1].item()), target_y, dt, lag)
        sr = exp_smooth(float(self.state[2].item()), target_rot, dt, lag)
        self.state[0] = sx
        self.state[1] = sy
        self.state[2] = sr
        return sx, sy, sr


class MicroMotionTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer('state', torch.zeros(3, device=device, dtype=dtype))
        self.seed = random.random() * 1000.0

    def forward(self, t: float, dt: float, attention: float, tension: float, energy: float, allow_micro_motion: bool = True) -> Tuple[float, float, float]:
        if not allow_micro_motion:
            self.state.mul_(0.92)
            return 0.0, 0.0, 0.0
        base = 0.08 + 0.16 * clamp(energy, 0.0, 1.0)
        base *= 1.0 - 0.72 * clamp(tension, 0.0, 1.0)
        base *= 0.55 + 0.45 * clamp(attention, 0.0, 1.0)
        n1 = math.sin(t * (0.55 + 0.18 * energy) + self.seed) + 0.5 * math.sin(t * 1.17 + self.seed * 0.3)
        n2 = math.sin(t * (0.43 + 0.12 * attention) + self.seed * 0.7) + 0.5 * math.sin(t * 1.41 + self.seed * 0.2)
        n3 = math.sin(t * (0.36 + 0.10 * energy) + self.seed * 0.9)
        target_x = clamp(n1 * base * 0.18, -0.08, 0.08)
        target_y = clamp(n2 * base * 0.16, -0.08, 0.08)
        target_r = clamp(n3 * base * 0.14, -0.06, 0.06)
        sx = exp_smooth(float(self.state[0].item()), target_x, dt, 0.28)
        sy = exp_smooth(float(self.state[1].item()), target_y, dt, 0.28)
        sr = exp_smooth(float(self.state[2].item()), target_r, dt, 0.32)
        self.state[0] = sx
        self.state[1] = sy
        self.state[2] = sr
        return sx, sy, sr


class SquashStretchTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer('scale_x', torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer('scale_y', torch.tensor(1.0, device=device, dtype=dtype))

    def forward(self, speed: float, acceleration: float, energy: float, arousal: float, mode_name: str) -> Tuple[float, float]:
        squash = clamp(speed * 0.40 + max(0.0, -acceleration) * 0.18 + energy * 0.05, 0.0, 1.0)
        stretch = clamp(max(0.0, acceleration) * 0.22 + arousal * 0.10, 0.0, 1.0)
        if mode_name in {'rest', 'observe'}:
            squash *= 0.70
            stretch *= 0.60
        if mode_name in {'speaking', 'reacting'}:
            stretch *= 1.15
        target_x = 1.0 - 0.045 * squash + 0.018 * stretch
        target_y = 1.0 + 0.055 * stretch - 0.028 * squash
        self.scale_x = torch.tensor(exp_smooth(float(self.scale_x.item()), target_x, 0.016, 0.18), device=self.device, dtype=self.dtype)
        self.scale_y = torch.tensor(exp_smooth(float(self.scale_y.item()), target_y, 0.016, 0.18), device=self.device, dtype=self.dtype)
        return float(self.scale_x.item()), float(self.scale_y.item())


class AnticipationRecoilTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer('phase', torch.tensor(0.0, device=device, dtype=dtype))
        self.register_buffer('recoil', torch.tensor(0.0, device=device, dtype=dtype))

    def forward(self, target_change: float, gesture_strength: float, dt: float, anticipation_bias: float = 0.5) -> Tuple[float, float]:
        anticipation = clamp(target_change * (0.08 + 0.16 * anticipation_bias) + gesture_strength * 0.12, 0.0, 1.0)
        recoil_target = clamp(target_change * 0.10 + gesture_strength * 0.14, 0.0, 1.0)
        self.phase.add_(dt * (1.6 + 0.6 * anticipation))
        self.recoil = torch.tensor(exp_smooth(float(self.recoil.item()), recoil_target, dt, 0.18), device=self.device, dtype=self.dtype)
        ease_in = smoothstep(0.0, 1.0, anticipation)
        ease_out = smoothstep(0.0, 1.0, float(self.recoil.item()))
        return ease_in, ease_out


class AnimationFluidityTransformer(nn.Module):
    def __init__(self, device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.secondary = SecondaryMotionTransformer(device=device, dtype=dtype)
        self.micro = MicroMotionTransformer(device=device, dtype=dtype)
        self.squash = SquashStretchTransformer(device=device, dtype=dtype)
        self.anticipation = AnticipationRecoilTransformer(device=device, dtype=dtype)
        self.state = AnimationFluidityState()

    def step(self, inp: AnimationFluidityInput) -> Dict[str, float]:
        dt = max(1e-3, float(inp.dt))
        speed = math.hypot(inp.body_lean_x - self.state.secondary_x, inp.body_lean_y - self.state.secondary_y) / max(dt, 1e-3)
        acceleration = (inp.motion_speed - self.state.scale) / max(dt, 1e-3)

        sec_x, sec_y, sec_r = self.secondary(
            velocity_x=inp.body_lean_x,
            velocity_y=inp.body_lean_y,
            rotation=inp.rotation,
            dt=dt,
            tension=inp.tension,
            energy=inp.energy,
        )
        mic_x, mic_y, mic_r = self.micro(
            t=self.state.phase,
            dt=dt,
            attention=inp.attention,
            tension=inp.tension,
            energy=inp.energy,
            allow_micro_motion=inp.allow_micro_motion,
        )
        scale_x, scale_y = self.squash(speed=speed, acceleration=acceleration, energy=inp.energy, arousal=inp.arousal, mode_name=inp.mode_name)
        anticipation, recoil = self.anticipation(target_change=abs(inp.motion_speed - self.state.scale), gesture_strength=inp.gesture_strength, dt=dt, anticipation_bias=inp.transition_bias)

        target_scale = clamp(inp.body_scale * (1.0 + 0.020 * anticipation - 0.018 * recoil), 0.70, 1.65)
        target_alpha = clamp(inp.visibility_alpha * (0.92 + 0.08 * (1.0 - inp.occupancy_pressure)), 0.08, 1.0)
        target_z = clamp(0.35 + 0.42 * inp.target_z + 0.10 * inp.fullscreen_pressure - 0.14 * inp.occupancy_pressure, 0.0, 1.0)

        self.state.secondary_x = exp_smooth(self.state.secondary_x, sec_x + mic_x, dt, 0.16)
        self.state.secondary_y = exp_smooth(self.state.secondary_y, sec_y + mic_y, dt, 0.16)
        self.state.secondary_rot = exp_smooth(self.state.secondary_rot, sec_r + mic_r, dt, 0.18)
        self.state.micro_x = exp_smooth(self.state.micro_x, mic_x, dt, 0.24)
        self.state.micro_y = exp_smooth(self.state.micro_y, mic_y, dt, 0.24)
        self.state.micro_rot = exp_smooth(self.state.micro_rot, mic_r, dt, 0.24)
        self.state.scale = exp_smooth(self.state.scale, target_scale * 0.5 + 0.5 * (0.98 * scale_x + 0.99 * scale_y), dt, 0.12)
        self.state.alpha = exp_smooth(self.state.alpha, target_alpha, dt, 0.18)
        self.state.z_order = exp_smooth(self.state.z_order, target_z, dt, 0.20)
        self.state.transition = exp_smooth(self.state.transition, anticipation - recoil, dt, 0.20)
        self.state.fluid_gate = exp_smooth(self.state.fluid_gate, 1.0 - 0.55 * clamp(inp.tension, 0.0, 1.0), dt, 0.22)
        self.state.phase += dt * (0.55 + 0.35 * inp.energy + 0.20 * inp.attention)
        self.state.last_state = inp.state_name
        self.state.last_action = 'steady'
        return self.to_dict()

    def to_dict(self) -> Dict[str, float]:
        return {
            'secondary_x': float(self.state.secondary_x),
            'secondary_y': float(self.state.secondary_y),
            'secondary_rot': float(self.state.secondary_rot),
            'micro_x': float(self.state.micro_x),
            'micro_y': float(self.state.micro_y),
            'micro_rot': float(self.state.micro_rot),
            'scale': float(self.state.scale),
            'alpha': float(self.state.alpha),
            'z_order': float(self.state.z_order),
            'transition': float(self.state.transition),
            'fluid_gate': float(self.state.fluid_gate),
        }


__all__ = [
    'AnimationFluidityInput',
    'AnimationFluidityState',
    'SecondaryMotionTransformer',
    'MicroMotionTransformer',
    'SquashStretchTransformer',
    'AnticipationRecoilTransformer',
    'AnimationFluidityTransformer',
    'clamp', 'lerp', 'exp_smooth', 'smoothstep', 'smooth_damp',
]


# ----------------------------------------------------------------------
#  Re-export higher-level body/orbit motion utilities
# ----------------------------------------------------------------------
try:
    from spatial_motion_intelligence import (
        BodyRotationTransformer,
        IntentTransformer,
        MotionPlanningTransformer,
        TrajectoryTransformer,
        PhysicsTransformer,
        MotionOrchestrator,
    )
    __all__ += [
        'BodyRotationTransformer',
        'IntentTransformer',
        'MotionPlanningTransformer',
        'TrajectoryTransformer',
        'PhysicsTransformer',
        'MotionOrchestrator',
    ]
except Exception:
    pass
