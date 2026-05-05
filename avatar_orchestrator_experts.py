from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    span = edge1 - edge0
    return 1.0 if span == 0.0 and x >= edge1 else (0.0 if span == 0.0 else ((t := clamp01((x - edge0) / span)) * t * (3.0 - 2.0 * t)))


def exp_smooth(current: torch.Tensor, target: torch.Tensor, tau: float, dt: float) -> torch.Tensor:
    alpha = 1.0 - math.exp(-max(dt, 0.0) / max(tau, 1e-6))
    return current + (target - current) * alpha


def norm2(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=keepdim).clamp_min(eps)


def softmax_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    t = max(temperature, 1e-5)
    return torch.softmax(logits / t, dim=-1)


def weighted_sample(items: Sequence[Tuple[Any, float]]) -> Any:
    if not items:
        return None
    weights = [max(0.0, float(w)) for _, w in items]
    total = sum(weights)
    if total <= 1e-9:
        return items[0][0]
    r = random.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item[0]
    return items[-1][0]


# ============================================================
# Scene and expert dataclasses
# ============================================================


@dataclass
class SceneState:
    t: float = 0.0
    dt: float = 0.016
    screen_w: int = 1920
    screen_h: int = 1080
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    mouse_near_avatar: float = 0.0
    mouse_near_click_zone: float = 0.0
    click: float = 0.0
    double_click: float = 0.0
    drag_strength: float = 0.0
    focus: float = 1.0
    silence: float = 0.0
    audio_energy: float = 0.0
    audio_peak: float = 0.0
    audio_attack: float = 0.0
    speaking: float = 0.0
    text_len: float = 0.0
    window_fullscreen: float = 0.0
    window_moving: float = 0.0
    window_overlap: float = 0.0
    dragged_window_rect: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    occupied_rects: Tuple[Tuple[float, float, float, float], ...] = ()
    heatmap: Optional[torch.Tensor] = None
    avatar_x: float = 0.0
    avatar_y: float = 0.0
    avatar_w: float = 256.0
    avatar_h: float = 256.0
    alpha: float = 1.0
    z_order: float = 0.5
    can_click_through: float = 0.0
    visible: float = 1.0
    interaction_mode: float = 0.0


@dataclass
class ExpertOutput:
    intent_logits: torch.Tensor
    motion_target: torch.Tensor
    velocity_target: torch.Tensor
    alpha: torch.Tensor
    z_order: torch.Tensor
    gaze_target: torch.Tensor
    body_scale: torch.Tensor
    body_rotation: torch.Tensor
    head_rotation: torch.Tensor
    eye_rotation: torch.Tensor
    mouth: torch.Tensor
    safety: torch.Tensor
    slot_id: torch.Tensor
    confidence: torch.Tensor
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)


# ============================================================
# Transformer / Expert stack
# ============================================================


class IntentTransformer(nn.Module):
    def __init__(self, d_model: int = 512, num_intents: int = 8):
        super().__init__()
        self.d_model = d_model
        self.backbone = nn.Sequential(
            nn.Linear(32, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_intents),
        )
        self.intent_names = (
            "idle", "approach", "avoid", "focus", "respect", "hide", "peek", "speak"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)

    def decode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        return {name: logits[..., idx] for idx, name in enumerate(self.intent_names)}


class MotionPlanningTransformer(nn.Module):
    def __init__(self, slots: int = 24, d_model: int = 256):
        super().__init__()
        self.slots = slots
        self.encoder = nn.Sequential(
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.slot_head = nn.Linear(d_model, slots)
        self.pos_head = nn.Linear(d_model, 4)
        self.meta_head = nn.Linear(d_model, 8)

    def forward(self, scene_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(scene_vec)
        return {
            "slot_logits": self.slot_head(h),
            "pos": torch.tanh(self.pos_head(h)),
            "meta": self.meta_head(h),
        }


class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsTransformer(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 12),
        )
        self.register_buffer("state", torch.zeros(12))

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        target = torch.tanh(self.net(x))
        self.state.copy_(exp_smooth(self.state, target.squeeze(0), tau=0.18, dt=float(dt.item())))
        return self.state


class SecondaryMotionTransformer(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class GazeSaccadeTransformer(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class FacialDynamicsTransformer(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 18),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class OromandibularKinematicTransformer(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(48, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, 18)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.encoder(x)))


class SpatialSafetyExpert(nn.Module):
    def __init__(self, slots: int = 24):
        super().__init__()
        self.slots = slots
        self.scorer = nn.Sequential(
            nn.Linear(72, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, slots),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x)


class VisibilityTransformer(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertRouter(nn.Module):
    def __init__(self, intents: int = 8, slots: int = 24):
        super().__init__()
        self.intent = IntentTransformer(d_model=512, num_intents=intents)
        self.motion = MotionPlanningTransformer(slots=slots)
        self.trajectory = TrajectoryTransformer()
        self.physics = PhysicsTransformer()
        self.secondary = SecondaryMotionTransformer()
        self.gaze = GazeSaccadeTransformer()
        self.face = FacialDynamicsTransformer()
        self.mouth = OromandibularKinematicTransformer()
        self.space = SpatialSafetyExpert(slots=slots)
        self.visibility = VisibilityTransformer()
        self.gating = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 10),
        )

    def build_scene_vec(self, scene: SceneState) -> torch.Tensor:
        heat = scene.heatmap
        heat_summary = torch.zeros(16) if heat is None else torch.tensor([
            float(heat.mean().item()), float(heat.max().item()), float(heat.min().item()), float(heat.std(unbiased=False).item())
        ] + [0.0] * 12)
        drag = torch.tensor(list(scene.dragged_window_rect) + list(scene.occupied_rects[0])[:0])
        base = torch.tensor([
            scene.mouse_speed, scene.mouse_near_avatar, scene.mouse_near_click_zone,
            scene.click, scene.double_click, scene.drag_strength,
            scene.focus, scene.silence, scene.audio_energy, scene.audio_peak,
            scene.audio_attack, scene.speaking, scene.text_len,
            scene.window_fullscreen, scene.window_moving, scene.window_overlap,
            scene.avatar_x / max(scene.screen_w, 1), scene.avatar_y / max(scene.screen_h, 1),
            scene.avatar_w / max(scene.screen_w, 1), scene.avatar_h / max(scene.screen_h, 1),
            scene.alpha, scene.z_order, scene.can_click_through, scene.visible,
            scene.interaction_mode, scene.mouse_x / max(scene.screen_w, 1), scene.mouse_y / max(scene.screen_h, 1),
            scene.mouse_dx, scene.mouse_dy, scene.t, scene.dt,
        ], dtype=torch.float32)
        return torch.cat([base, heat_summary], dim=0).unsqueeze(0)

    def forward(self, scene: SceneState) -> ExpertOutput:
        scene_vec = self.build_scene_vec(scene)
        intent_logits = self.intent(scene_vec[:, :32])
        gate = softmax_temperature(self.gating(scene_vec[:, :128]), temperature=0.85)
        slot_logits = self.motion(scene_vec[:, :64])["slot_logits"]
        slot_id = torch.argmax(slot_logits, dim=-1)
        pos = self.motion(scene_vec[:, :64])["pos"]
        meta = self.motion(scene_vec[:, :64])["meta"]
        physics = self.physics(scene_vec[:, :48], torch.tensor(scene.dt))
        traj = self.trajectory(scene_vec[:, :48])
        sec = self.secondary(scene_vec[:, :32])
        gaze = self.gaze(scene_vec[:, :24])
        face = self.face(scene_vec[:, :40])
        mouth = self.mouth(scene_vec[:, :48])
        vis = self.visibility(scene_vec[:, :32])
        space_logits = self.space(scene_vec[:, :72])
        z = torch.tanh(vis[:, 0:1])
        alpha = torch.sigmoid(vis[:, 1:2])
        safety = torch.sigmoid(vis[:, 2:3])
        motion_target = torch.cat([pos, traj[:, :2]], dim=-1)
        velocity_target = traj[:, 2:4]
        body_scale = torch.sigmoid(meta[:, 0:1]) * 1.1 + 0.6
        body_rotation = torch.tanh(meta[:, 1:2]) * 15.0
        head_rotation = torch.tanh(meta[:, 2:3]) * 10.0
        eye_rotation = torch.tanh(gaze[:, :2])
        confidence = torch.sigmoid(meta[:, 3:4])
        return ExpertOutput(
            intent_logits=intent_logits,
            motion_target=motion_target,
            velocity_target=velocity_target,
            alpha=alpha,
            z_order=z,
            gaze_target=gaze[:, :2],
            body_scale=body_scale,
            body_rotation=body_rotation,
            head_rotation=head_rotation,
            eye_rotation=eye_rotation,
            mouth=mouth,
            safety=safety,
            slot_id=slot_id,
            confidence=confidence,
            extras={
                "gate": gate,
                "physics": physics,
                "secondary": sec,
                "face": face,
                "space_logits": space_logits,
            },
        )


# ============================================================
# Compatibility shims for existing modules
# ============================================================


def _wrap_space_init(cls: Any) -> None:
    fields = getattr(cls, "__dataclass_fields__", {})
    original = cls.__init__

    def _coerce(self, *args: Any, **kwargs: Any) -> None:
        alias_map = {
            "moving_window_rect": "dragged_window_rect",
            "dragged_rect": "dragged_window_rect",
            "drag_window_rect": "dragged_window_rect",
        }
        normalized: Dict[str, Any] = {}
        normalized.update({k: v for k, v in kwargs.items() if k in fields})
        normalized.update({alias_map.get(k, k): v for k, v in kwargs.items() if alias_map.get(k, k) in fields})
        for key, field in fields.items():
            if key not in normalized:
                default = field.default if field.default is not field.default_factory else None
                normalized[key] = default if default is not None else (field.default_factory() if getattr(field, "default_factory", None) is not None else None)
        try:
            original(self, *args, **normalized)
        except TypeError:
            original(self, **normalized)

    cls.__init__ = _coerce


def patch_occupancy(glb: Dict[str, Any]) -> None:
    cls = glb.get("SpaceOccupancyInput")
    if cls is not None:
        _wrap_space_init(cls)

    controller = glb.get("OccupancyVisibilityController")
    if controller is not None and not hasattr(controller, "expert_router"):
        controller.expert_router = ExpertRouter()


def patch_phoneme(glb: Dict[str, Any]) -> None:
    timeline = glb.get("PhonemeVisemeTimeline")
    snapshot = glb.get("VisemeSnapshot")
    if timeline is None or snapshot is None:
        return

    def _augment(snapshot_obj: Any, mouth: torch.Tensor, face: torch.Tensor) -> Any:
        extras = {
            "lips_together": float(torch.sigmoid(mouth[..., 0]).item()),
            "lip_corner_up": float(torch.sigmoid(face[..., 0]).item()),
            "lip_corner_down": float(torch.sigmoid(face[..., 1]).item()),
            "jaw_clench": float(torch.sigmoid(mouth[..., 1]).item()),
            "tongue_out": float(torch.sigmoid(mouth[..., 2]).item()),
            "tongue_tip_interdental": float(torch.sigmoid(mouth[..., 3]).item()),
            "upper_lip_raise": float(torch.sigmoid(mouth[..., 4]).item()),
            "lower_lip_depress": float(torch.sigmoid(mouth[..., 5]).item()),
            "cheek_puff": float(torch.sigmoid(face[..., 2]).item()),
            "cheek_suck": float(torch.sigmoid(face[..., 3]).item()),
            "mouth_corner_stretch": float(torch.sigmoid(face[..., 4]).item()),
        }
        for k, v in extras.items():
            setattr(snapshot_obj, k, v)
        return snapshot_obj

    original_sample = timeline.sample

    def sample(self, t: float, *, speaking: bool = True, audio_energy: float = 0.0):
        s = original_sample(self, t, speaking=speaking, audio_energy=audio_energy)
        return _augment(s, torch.tensor([s.mouth_open, s.jaw_drop, s.tongue_up, s.lip_round, s.mouth_wide, s.smile]), torch.tensor([s.cheek, s.smile, s.cheek, s.cheek, s.smile]))

    timeline.sample = sample


def patch_core(glb: Dict[str, Any]) -> None:
    brain = glb.get("AvatarTransformerBrain")
    if brain is not None and not hasattr(brain, "expert_router"):
        brain.expert_router = ExpertRouter()


def patch_runtime(glb: Dict[str, Any]) -> None:
    app = glb.get("AvatarAppV2")
    renderer = glb.get("AvatarRendererV2")
    motion = glb.get("MotionEngineV2")
    behavior = glb.get("BehaviorEngineV2")
    state_input = glb.get("SpaceOccupancyInput")
    controller = glb.get("OccupancyVisibilityController")

    if state_input is not None:
        _wrap_space_init(state_input)

    if controller is not None and not hasattr(controller, "expert_router"):
        controller.expert_router = ExpertRouter()

    if behavior is not None:
        original_update = behavior.update

        def update(self, perception, state, plan, t):
            expert = getattr(self, "expert_router", None)
            out = expert.forward(getattr(self, "last_scene", SceneState(t=t))) if expert is not None else None
            result = original_update(self, perception, state, plan, t)
            setattr(self, "expert_scene_output", out)
            return result

        behavior.update = update

    if motion is not None:
        original_update = motion.update

        def update(self, perception, state, behavior_obj, attention, plan, dt):
            expert = getattr(self, "expert_router", None)
            result = original_update(self, perception, state, behavior_obj, attention, plan, dt)
            if expert is not None:
                scene = getattr(self, "last_scene", SceneState(dt=dt))
                exp = expert.forward(scene)
                setattr(self, "last_expert_output", exp)
            return result

        motion.update = update

    if renderer is not None:
        original_render = renderer.render

        def render(self, t, perception, emotion, motion_data, attention, audio, plan):
            out = original_render(self, t, perception, emotion, motion_data, attention, audio, plan)
            return out

        renderer.render = render

    if app is not None:
        original_update_logic = app._update_logic if hasattr(app, "_update_logic") else None
        if original_update_logic is not None:
            def _update_logic(self, user_text: str = ""):
                result = original_update_logic(self, user_text)
                expert = getattr(self, "expert_router", None)
                if expert is not None:
                    scene = SceneState(
                        t=getattr(self, "last_frame_t", 0.0),
                        dt=0.016,
                        mouse_x=float(getattr(self, "mouse_x", 0.0)),
                        mouse_y=float(getattr(self, "mouse_y", 0.0)),
                        focus=float(getattr(getattr(self, "perception_data", None), "focus", 1.0)),
                        silence=float(getattr(getattr(self, "perception_data", None), "silence_s", 0.0)),
                        audio_energy=float(getattr(getattr(self, "audio", None), "energy", 0.0)),
                        audio_peak=float(getattr(getattr(self, "audio", None), "peak", 0.0)),
                        audio_attack=float(getattr(getattr(self, "audio", None), "attack", 0.0)),
                        speaking=float(bool(getattr(self, "is_speaking", False))),
                        text_len=float(len(user_text)),
                    )
                    setattr(self, "expert_scene", scene)
                    setattr(self, "expert_bundle", expert(scene))
                return result

            app._update_logic = _update_logic


def patch_all(glb: Dict[str, Any]) -> None:
    patch_occupancy(glb)
    patch_phoneme(glb)
    patch_core(glb)
    patch_runtime(glb)

