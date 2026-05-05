
from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple

from avata.advanced_animation_transformers import clamp


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-max(0.0, dt) / tau)
    return _lerp(current, target, k)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle



def audio_mouth_blink(audio: Any, emotion: Any, perception: Any) -> float:
    """Tiny, safe blink helper used by the axial renderer.

    It is intentionally lightweight and tolerant of partial audio/state objects.
    """
    energy = 0.0
    speaking = False
    silence_s = 0.0
    try:
        energy = float(getattr(audio, "energy", 0.0) or 0.0)
    except Exception:
        energy = 0.0
    try:
        speaking = bool(getattr(perception, "speaking", False) or getattr(audio, "speaking", False))
    except Exception:
        speaking = False
    try:
        silence_s = float(getattr(perception, "silence_s", 0.0) or 0.0)
    except Exception:
        silence_s = 0.0
    base = 0.05 + 0.55 * max(0.0, min(1.0, energy))
    if speaking:
        base += 0.10
    if silence_s > 0.4:
        base += 0.08
    try:
        intensity = float(getattr(emotion, "intensity", 0.5) or 0.5)
        base += 0.04 * max(0.0, min(1.0, intensity))
    except Exception:
        pass
    return max(0.0, min(1.0, base))


@dataclass
class AxialRotationInput:
    t: float = 0.0
    dt: float = 0.016
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    face_x: float = 0.0
    face_y: float = 0.0
    focus_x: float = 0.0
    focus_y: float = 0.0
    mouse_distance_to_face: float = 9999.0
    mouse_speed: float = 0.0
    attention: float = 0.5
    curiosity: float = 0.5
    tension: float = 0.0
    energy: float = 0.5
    intensity: float = 0.5
    arousal: float = 0.5
    speaking: bool = False
    state_name: str = "idle"
    plan_state: str = ""
    plan_text: str = ""
    plan_intention: str = ""
    silence_s: float = 0.0
    interaction_intensity: float = 0.0
    window_active: bool = True


@dataclass
class AxialRotationState:
    eye_offset_x: float = 0.0
    eye_offset_y: float = 0.0
    eye_roll_left: float = 0.0
    eye_roll_right: float = 0.0
    eye_convergence: float = 0.0
    pupil_scale: float = 1.0
    head_rot: float = 0.0
    head_torsion: float = 0.0
    saccade_phase: float = 0.0
    eye_lead: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    head_lag_x: float = 0.0
    head_lag_y: float = 0.0
    micro_arc: float = 0.0
    focus_depth: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class SphericalEyeRollTransformer:
    def __init__(self):
        self._phase = 0.0
        self._prev_target = (0.0, 0.0)

    def predict(self, *, target_x: float, target_y: float, eye_center_x: float, eye_center_y: float,
                radius: float = 28.0, pupil_scale: float = 1.0, convergence: float = 0.0,
                dt: float = 0.016, lead_strength: float = 0.65) -> Tuple[float, float, float]:
        dx = target_x - eye_center_x
        dy = target_y - eye_center_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0, 0.0

        self._phase = (self._phase + dt * (3.2 + 2.4 * lead_strength)) % (math.tau * 4.0)
        norm = _clamp(dist / max(1.0, radius * 7.0), 0.0, 1.0)
        angle = math.atan2(dy, dx)

        # Curva esférica: o movimento cresce em arco, não em linha seca.
        arc = 0.18 * norm * norm
        orbit = 0.10 * norm * math.sin(self._phase)
        roll_radius = radius * (0.22 + 0.78 * norm) * (0.96 + 0.04 * pupil_scale)
        roll_radius *= (1.0 - 0.20 * abs(convergence))
        x = math.cos(angle) * roll_radius * (1.0 - 0.14 * norm) + math.cos(angle + math.pi / 2.0) * arc * 3.0
        y = math.sin(angle) * roll_radius * (1.0 + 0.10 * norm) - arc * 2.4 + orbit
        return x, y, norm


class ConvergenceTransformer:
    def __init__(self):
        self._state = 0.0

    def predict(self, *, mouse_distance_to_face: float, face_scale: float, attention: float, intensity: float,
                speaking: bool, dt: float) -> float:
        close = 1.0 - _clamp(mouse_distance_to_face / max(90.0, face_scale * 260.0), 0.0, 1.0)
        target = 0.10 + 0.62 * close + 0.10 * attention + 0.06 * intensity
        if speaking:
            target += 0.05
        self._state = _exp_smooth(self._state, _clamp(target, 0.0, 1.0), dt, 0.12)
        return self._state


class HeadEyeCoordinationTransformer:
    def __init__(self):
        self._eye_state_x = 0.0
        self._eye_state_y = 0.0
        self._head_state = 0.0

    def predict(self, *, eye_target_x: float, eye_target_y: float, head_target_roll: float,
                attention: float, dt: float) -> Tuple[float, float, float, float]:
        eye_tau = 0.045 + 0.04 * (1.0 - attention)
        head_tau = 0.18 + 0.12 * (1.0 - attention)
        self._eye_state_x = _exp_smooth(self._eye_state_x, eye_target_x, dt, eye_tau)
        self._eye_state_y = _exp_smooth(self._eye_state_y, eye_target_y, dt, eye_tau)
        self._head_state = _exp_smooth(self._head_state, head_target_roll, dt, head_tau)
        lead_ms = 85.0 + 65.0 * (1.0 - attention)
        return self._eye_state_x, self._eye_state_y, self._head_state, lead_ms


class TorsionalRotationTransformer:
    def __init__(self):
        self._roll = 0.0

    def predict(self, *, curiosity: float, tension: float, intensity: float, plan_text: str,
                plan_intention: str, state_name: str, dt: float) -> float:
        cue = 0.0
        txt = (plan_text or "").lower()
        if "?" in txt or any(w in txt for w in ("não sei", "talvez", "confuso", "incerto", "por quê", "porque")):
            cue += 1.0
        if plan_intention in {"ask", "explain", "observe"}:
            cue += 0.4
        if state_name in {"thinking", "curious", "peek", "observe"}:
            cue += 0.25
        cue += 0.35 * curiosity + 0.25 * tension + 0.12 * intensity
        target = _clamp((cue - 0.55) * 14.0, -10.0, 10.0)
        self._roll = _exp_smooth(self._roll, target, dt, 0.14)
        return self._roll


class AxialRotationEngine:
    def __init__(self):
        self.eye_roll = SphericalEyeRollTransformer()
        self.convergence = ConvergenceTransformer()
        self.head_eye = HeadEyeCoordinationTransformer()
        self.torsion = TorsionalRotationTransformer()
        self.state = AxialRotationState()
        self._last_target = (0.0, 0.0)

    def step(self, inp: AxialRotationInput) -> AxialRotationState:
        face_scale = 1.0 + 0.12 * inp.energy + 0.05 * inp.intensity
        convergence = self.convergence.predict(
            mouse_distance_to_face=inp.mouse_distance_to_face,
            face_scale=face_scale,
            attention=inp.attention,
            intensity=inp.intensity,
            speaking=inp.speaking,
            dt=inp.dt,
        )

        # Micro-alvos orbitais suaves, tipo sacada
        focus_x = _lerp(inp.focus_x, inp.mouse_x, 0.25 + 0.20 * inp.attention)
        focus_y = _lerp(inp.focus_y, inp.mouse_y, 0.25 + 0.20 * inp.attention)

        eye_offset_x, eye_offset_y, saccade_norm = self.eye_roll.predict(
            target_x=focus_x,
            target_y=focus_y,
            eye_center_x=inp.face_x,
            eye_center_y=inp.face_y,
            radius=28.0 + 8.0 * inp.arousal,
            pupil_scale=self.state.pupil_scale,
            convergence=convergence,
            dt=inp.dt,
            lead_strength=0.7 if inp.state_name in {"speaking", "reacting"} else 0.55,
        )

        head_roll_target = self.torsion.predict(
            curiosity=inp.curiosity,
            tension=inp.tension,
            intensity=inp.intensity,
            plan_text=inp.plan_text,
            plan_intention=inp.plan_intention,
            state_name=inp.state_name,
            dt=inp.dt,
        )

        # Head-eye coordination: olhos antecipam, cabeça segue suavemente.
        eye_x, eye_y, head_roll, lead_ms = self.head_eye.predict(
            eye_target_x=eye_offset_x,
            eye_target_y=eye_offset_y,
            head_target_roll=head_roll_target,
            attention=inp.attention,
            dt=inp.dt,
        )

        # Compressão/expansão sutil do olho conforme foco/proximidade.
        pupil_scale_target = 1.0 + 0.10 * inp.attention + 0.08 * saccade_norm - 0.06 * convergence
        if inp.speaking:
            pupil_scale_target += 0.03
        pupil_scale = _exp_smooth(self.state.pupil_scale, _clamp(pupil_scale_target, 0.86, 1.12), inp.dt, 0.10)
        self.state.pupil_scale = pupil_scale

        # Torsão final suave.
        torsion_boost = 0.35 * head_roll_target + 0.15 * inp.tension
        head_torsion = _exp_smooth(self.state.head_torsion, _clamp(torsion_boost, -10.0, 10.0), inp.dt, 0.16)

        # Constrói estado final sem tremor.
        self.state.eye_offset_x = eye_x
        self.state.eye_offset_y = eye_y
        self.state.eye_roll_left = eye_x - 2.6 * convergence
        self.state.eye_roll_right = eye_x + 2.6 * convergence
        self.state.eye_convergence = convergence
        self.state.head_rot = head_roll
        self.state.head_torsion = head_torsion
        self.state.saccade_phase = self.eye_roll._phase
        self.state.eye_lead = lead_ms
        self.state.gaze_x = _exp_smooth(self.state.gaze_x, eye_x, inp.dt, 0.055)
        self.state.gaze_y = _exp_smooth(self.state.gaze_y, eye_y, inp.dt, 0.055)
        self.state.head_lag_x = _exp_smooth(self.state.head_lag_x, 0.0, inp.dt, 0.20)
        self.state.head_lag_y = _exp_smooth(self.state.head_lag_y, 0.0, inp.dt, 0.20)
        self.state.micro_arc = 0.14 * saccade_norm
        self.state.focus_depth = convergence
        return self.state


def install_axial_rotation_pipeline(motion_cls: Any, renderer_cls: Any = None, module_globals: Optional[Dict[str, Any]] = None) -> None:
    """Patch the merged runtime engine to add axial rotation without removing existing logic."""
    if motion_cls is None:
        return
    if getattr(motion_cls, "_axial_rotation_pipeline_applied", False):
        return

    orig_init = getattr(motion_cls, "__init__", None)
    orig_update = getattr(motion_cls, "update", None)
    if orig_init is None or orig_update is None:
        return

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.axial_engine = AxialRotationEngine()
        self._axial_state = AxialRotationState()

    def update(self, perception, emotion, behavior, attention, plan):
        motion_data = orig_update(self, perception, emotion, behavior, attention, plan)
        try:
            state_name = getattr(behavior, "current_state", motion_data.get("state_name", "idle"))
            target_point = getattr(attention, "target_point", (getattr(perception, "mouse_x", 0.0), getattr(perception, "mouse_y", 0.0)))
            focus_x, focus_y = float(target_point[0]), float(target_point[1])
            face_x = 512.0 + float(getattr(self.body, "head_x", 0.0)) * 12.0
            face_y = 512.0 + float(getattr(self.body, "head_y", 0.0)) * 10.0
            plan_text = getattr(plan, "text", "") if plan is not None else ""
            plan_intention = getattr(plan, "intention", "") if plan is not None else ""
            inp = AxialRotationInput(
                t=float(getattr(perception, "t", 0.0)),
                dt=float(getattr(perception, "dt", 0.016)),
                mouse_x=float(getattr(perception, "mouse_x", 0.0)),
                mouse_y=float(getattr(perception, "mouse_y", 0.0)),
                face_x=face_x,
                face_y=face_y,
                focus_x=focus_x,
                focus_y=focus_y,
                mouse_distance_to_face=float(getattr(perception, "mouse_distance_to_face", 9999.0)),
                mouse_speed=float(getattr(perception, "mouse_speed", 0.0)),
                attention=float(getattr(emotion, "attention", 0.5)),
                curiosity=float(getattr(emotion, "curiosity", 0.5)),
                tension=float(getattr(emotion, "tension", 0.0)),
                energy=float(getattr(emotion, "energy", 0.5)),
                intensity=float(getattr(emotion, "intensity", 0.5)),
                arousal=float(getattr(emotion, "arousal", 0.5)),
                speaking=bool(getattr(perception, "speaking", False)),
                state_name=str(state_name),
                plan_state=str(getattr(plan, "state", "")) if plan is not None else "",
                plan_text=plan_text,
                plan_intention=plan_intention,
                silence_s=float(getattr(perception, "silence_s", 0.0)),
                interaction_intensity=float(getattr(motion_data.get("fluidity", {}), "get", lambda *_: 0.0)("stability", 0.0)) if isinstance(motion_data.get("fluidity"), dict) else 0.0,
                window_active=bool(getattr(perception, "window_active", True)),
            )
            axial = self.axial_engine.step(inp)
            self.body.gaze_x = float(axial.gaze_x)
            self.body.gaze_y = float(axial.gaze_y)
            self.body.head_rot = float(axial.head_rot)
            self.body.head_torsion = float(axial.head_torsion)
            self.body.eye_offset_x = float(axial.eye_offset_x)
            self.body.eye_offset_y = float(axial.eye_offset_y)
            self.body.eye_roll_left = float(axial.eye_roll_left)
            self.body.eye_roll_right = float(axial.eye_roll_right)
            self.body.eye_convergence = float(axial.eye_convergence)
            self.body.pupil_scale = float(axial.pupil_scale)
            self.body.saccade_phase = float(axial.saccade_phase)
            motion_data["axial"] = axial.as_dict()
            motion_data["head_rot"] = float(axial.head_rot)
            motion_data["eye_convergence"] = float(axial.eye_convergence)
            motion_data["pupil_scale"] = float(axial.pupil_scale)
        except Exception:
            pass
        return motion_data

    motion_cls.__init__ = __init__
    motion_cls.update = update
    motion_cls._axial_rotation_pipeline_applied = True

    if renderer_cls is None or getattr(renderer_cls, "_axial_rotation_renderer_applied", False):
        return

    orig_draw_eye = getattr(renderer_cls, "_draw_eye", None)
    orig_draw_mouth = getattr(renderer_cls, "_draw_mouth", None)
    orig_render = getattr(renderer_cls, "render", None)
    if not all([orig_draw_eye, orig_draw_mouth, orig_render]):
        return

    def _draw_eye(self, img, ex, ey, state, pose, side, blink):
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        gaze_x = float(getattr(pose, "gaze_x", 0.0))
        gaze_y = float(getattr(pose, "gaze_y", 0.0))
        eye_offset_x = float(getattr(pose, "eye_offset_x", 0.0))
        eye_offset_y = float(getattr(pose, "eye_offset_y", 0.0))
        convergence = float(getattr(pose, "eye_convergence", 0.0))
        pupil_scale = float(getattr(pose, "pupil_scale", 1.0))
        head_rot = float(getattr(pose, "head_rot", 0.0))

        base_x = gaze_x * (14 if side == 0 else 12) + eye_offset_x
        base_y = gaze_y * 10.0 + eye_offset_y

        # Curva esférica: o olho move-se em arco e não em linha seca.
        target_angle = math.atan2(base_y, base_x if abs(base_x) > 1e-6 else 1e-6)
        target_dist = math.hypot(base_x, base_y)
        spherical = _clamp(target_dist / 20.0, 0.0, 1.0)
        arc = 0.18 * spherical * spherical
        compress = 1.0 - 0.16 * spherical
        iris_shift_x = math.cos(target_angle) * target_dist * 0.42 * compress
        iris_shift_y = math.sin(target_angle) * target_dist * 0.34 * (1.0 + arc)

        # Convergência funcional dos olhos.
        convergence_push = (1.0 if side == 0 else -1.0) * convergence * 7.5

        eye_w = 58
        eye_h = 50 * state.arousal
        open_amt = clamp(self._eye_open_for_state(state, pose, blink), 0.25, 1.35)
        eye_h *= open_amt

        d.ellipse((ex - eye_w, ey - eye_h, ex + eye_w, ey + eye_h), fill=(240, 245, 255, 255))

        iris_cx = ex + iris_shift_x + convergence_push + (0.08 * head_rot)
        iris_cy = ey + iris_shift_y - (0.03 * head_rot)
        iris_r = max(18, int(30 * pupil_scale * (1.0 - 0.12 * spherical)))

        for r in range(iris_r, 0, -1):
            rr = int(12 + r * 0.3)
            gg = int(70 + r * 1.8)
            bb = int(150 + r * 1.0)
            d.ellipse((iris_cx - r, iris_cy - r, iris_cx + r, iris_cy + r), fill=(rr, gg, bb, 255))
        d.ellipse((iris_cx - 10, iris_cy - 10, iris_cx + 10, iris_cy + 10), fill=(0, 0, 0, 255))
        d.ellipse((iris_cx - 24, iris_cy - 24, iris_cx - 8, iris_cy - 10), fill=(255, 255, 255, 180))
        if blink > 0.01:
            lid_h = eye_h * blink * 1.55
            d.rounded_rectangle((ex - eye_w - 6, ey - lid_h, ex + eye_w + 6, ey + lid_h), radius=22, fill=(80, 112, 145, 255))
        d.arc((ex - 63, ey - 10, ex + 63, ey + 45), 195, 345, fill=(20, 48, 76, 170), width=6)
        return img

    def render(self, t, perception, emotion, pose, face, audio, plan, blink_amount=0.0):
        from PIL import Image, ImageDraw, ImageFilter
        img = self._base_canvas()
        img = self._body(img, pose, emotion, t)

        head_x = int(self.cx + float(getattr(pose, "head_x", 0.0)) * 12)
        head_y = int(self.cy - 15 + float(getattr(pose, "head_y", 0.0)) * 10)
        radius = self.head_radius * float(getattr(pose, "scale_x", 1.0)) * (0.98 + 0.02 * emotion.energy)

        head_layer = self._head_gradient(head_x, head_y, radius, t, emotion)
        head_layer = self._ambient(head_layer, emotion, pose, t)

        brow_lift = -20 * face.brow_left - 6 * emotion.curiosity + 9 * emotion.tension
        brow_lift_r = -20 * face.brow_right - 6 * emotion.curiosity + 9 * emotion.tension
        if emotion.current_state in {"thinking", "focused"}:
            brow_lift += 6
            brow_lift_r += 6
        if emotion.current_state in {"surprised", "reacting"}:
            brow_lift -= 14
            brow_lift_r -= 14
        if emotion.current_state in {"shy", "hide", "retreat"}:
            brow_lift += 5
            brow_lift_r += 5

        eye_y = head_y - int(radius * 0.22)
        left_eye_x = head_x - int(radius * 0.34)
        right_eye_x = head_x + int(radius * 0.34)

        head_layer = self._draw_brow(head_layer, left_eye_x, eye_y - 34, 80, brow_lift, -5 if getattr(pose, "gaze_target", "mouse") != "mouse" else -2, emotion.intensity)
        head_layer = self._draw_brow(head_layer, right_eye_x, eye_y - 34, 80, brow_lift_r, 5 if getattr(pose, "gaze_target", "mouse") != "mouse" else 2, emotion.intensity)

        blink = max(blink_amount, 0.55 * audio_mouth_blink(audio, emotion, perception))
        head_layer = self._draw_eye(head_layer, left_eye_x, eye_y, emotion, pose, side=0, blink=blink)
        head_layer = self._draw_eye(head_layer, right_eye_x, eye_y, emotion, pose, side=1, blink=blink)

        mouth_open = audio.mouth_open(emotion, plan)
        if plan and plan.state == "speaking":
            mouth_open = max(mouth_open, 0.04 + audio.energy * 0.82)
        else:
            mouth_open = max(mouth_open * 0.35, 0.02 + emotion.energy * 0.04)

        head_layer = self._draw_mouth(head_layer, head_x, head_y, emotion, face, mouth_open, plan, audio.energy)

        # Rotação axial do bloco da cabeça para dar volume e rotação 3D.
        head_angle = float(getattr(pose, "head_rot", 0.0)) + float(getattr(pose, "head_torsion", 0.0))
        if abs(head_angle) > 1e-4:
            head_layer = head_layer.rotate(head_angle, resample=Image.Resampling.BICUBIC, center=(head_x, head_y))

        if getattr(pose, "gaze_target", "mouse") == "mouse" and getattr(perception, "mouse_distance_to_face", 9999.0) < 260:
            hint = Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))
            d = ImageDraw.Draw(hint)
            mx = int(getattr(perception, "mouse_x", head_x))
            my = int(getattr(perception, "mouse_y", head_y))
            d.ellipse((mx - 8, my - 8, mx + 8, my + 8), outline=(180, 240, 255, 36), width=2)
            hint = hint.filter(ImageFilter.GaussianBlur(3))
            head_layer = Image.alpha_composite(head_layer, hint)

        img = Image.alpha_composite(img, head_layer)
        img = self._finish(img)

        mask = Image.new("L", (self.render_res, self.render_res), 0)
        dm = ImageDraw.Draw(mask)
        dm.ellipse((self.cx - radius - 26, self.cy - radius - 26,
                    self.cx + radius + 26, self.cy + radius + 26), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(14))
        img.putalpha(mask)
        return img

    renderer_cls._draw_eye = _draw_eye
    renderer_cls.render = render
    renderer_cls._axial_rotation_renderer_applied = True


def install_axial_rotation_core(module_globals: Optional[Dict[str, Any]] = None) -> None:
    """Patch the core torch-based brain to keep axial motion coherent."""
    if not module_globals:
        return
    motion_cls = module_globals.get("MotionEngine")
    if motion_cls is None or getattr(motion_cls, "_axial_rotation_core_applied", False):
        return

    orig_init = getattr(motion_cls, "__init__", None)
    orig_forward = getattr(motion_cls, "forward", None)
    if orig_init is None or orig_forward is None:
        return

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.register_buffer("axial_eye_roll", getattr(self, "last_scale").clone().fill_(0.0) if hasattr(self, "last_scale") else None)
        self.register_buffer("axial_head_roll", getattr(self, "last_scale").clone().fill_(0.0) if hasattr(self, "last_scale") else None)
        self._axial_core_engine = AxialRotationEngine()

    def forward(self, body, emotion, attention, dt, mode, spatial_features=None, workspace_features=None):
        out = orig_forward(self, body, emotion, attention, dt, mode, spatial_features, workspace_features)
        try:
            # Use available torch tensors to produce very light axial hints without changing the output shape.
            mouse_x = 512.0
            mouse_y = 512.0
            if spatial_features is not None:
                flat = self._pad_features(spatial_features, 8) if hasattr(self, "_pad_features") else spatial_features.flatten()
                mouse_x = float(flat[0].item()) if flat.numel() > 0 else mouse_x
                mouse_y = float(flat[1].item()) if flat.numel() > 1 else mouse_y
            inp = AxialRotationInput(
                dt=float(dt.item() if hasattr(dt, "item") else dt),
                mouse_x=mouse_x,
                mouse_y=mouse_y,
                mouse_distance_to_face=180.0,
                attention=float(attention.mean().item()) if hasattr(attention, "mean") else 0.5,
                curiosity=float(emotion[4].item()) if hasattr(emotion, "__getitem__") else 0.5,
                tension=float(emotion[3].item()) if hasattr(emotion, "__getitem__") else 0.0,
                energy=float(emotion[5].item()) if hasattr(emotion, "__getitem__") else 0.5,
                intensity=float(emotion[0].item()) if hasattr(emotion, "__getitem__") else 0.5,
                arousal=float(emotion[1].item()) if hasattr(emotion, "__getitem__") else 0.5,
                speaking=bool(mode.value == "speaking") if hasattr(mode, "value") else False,
                state_name=str(mode.value) if hasattr(mode, "value") else "idle",
            )
            axial = self._axial_core_engine.step(inp)
            if isinstance(out, tuple) and len(out) >= 1 and hasattr(out[0], "clone"):
                body_t = out[0].clone()
                if body_t.numel() >= 6:
                    body_t[0] = body_t[0] + float(axial.head_rot) * 0.001
                    body_t[1] = body_t[1] + float(axial.eye_convergence) * 0.002
                    body_t[2] = body_t[2] + float(axial.eye_offset_x) * 0.0005
                out = body_t
        except Exception:
            pass
        return out

    motion_cls.__init__ = __init__
    motion_cls.forward = forward
    motion_cls._axial_rotation_core_applied = True


def install_axial_rotation_viseme_hints(module_globals: Optional[Dict[str, Any]] = None) -> None:
    """Add optional axial hints into the viseme bridge for richer eye/head syncing."""
    if not module_globals:
        return
    snap_cls = module_globals.get("VisemeSnapshot")
    if snap_cls is None or getattr(snap_cls, "_axial_rotation_viseme_applied", False):
        return

    if not hasattr(snap_cls, "__dataclass_fields__"):
        return

    # Add class-level defaults for compatibility; existing instances will pick them up via attribute access.
    for name, default in {
        "head_roll_hint": 0.0,
        "eye_focus_hint": 0.5,
        "saccade_hint": 0.0,
        "convergence_hint": 0.0,
    }.items():
        if not hasattr(snap_cls, name):
            setattr(snap_cls, name, default)

    orig_as_dict = getattr(snap_cls, "as_dict", None)
    if orig_as_dict is not None and not getattr(orig_as_dict, "_axial_rotation_viseme_applied", False):
        def as_dict(self):
            data = orig_as_dict(self)
            data.setdefault("head_roll_hint", float(getattr(self, "head_roll_hint", 0.0)))
            data.setdefault("eye_focus_hint", float(getattr(self, "eye_focus_hint", 0.5)))
            data.setdefault("saccade_hint", float(getattr(self, "saccade_hint", 0.0)))
            data.setdefault("convergence_hint", float(getattr(self, "convergence_hint", 0.0)))
            return data
        as_dict._axial_rotation_viseme_applied = True
        snap_cls.as_dict = as_dict

    snap_cls._axial_rotation_viseme_applied = True
