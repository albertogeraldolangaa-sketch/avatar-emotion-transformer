from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------

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
    k = 1.0 - math.exp(-dt / tau)
    return lerp(current, target, k)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if torch is not None and isinstance(v, torch.Tensor):
            return float(v.detach().cpu().flatten()[0].item())
        return float(v)
    except Exception:
        return default


def _point(obj: Any, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return _to_float(obj[0], default[0]), _to_float(obj[1], default[1])
    if torch is not None and isinstance(obj, torch.Tensor) and obj.numel() >= 2:
        flat = obj.detach().cpu().flatten()
        return float(flat[0].item()), float(flat[1].item())
    return default


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass
class SaccadeState:
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    phase: float = 0.0
    orbit_phase: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    settle_timer: float = 0.0


@dataclass
class BlinkState:
    blink: float = 0.0
    timer: float = 0.0
    next_blink: float = 3.2
    close_phase: float = 0.0
    open_phase: float = 0.0
    closed_hold: float = 0.0


@dataclass
class SmirkState:
    left_curve: float = 0.0
    right_curve: float = 0.0
    left_cheek: float = 0.0
    right_cheek: float = 0.0
    left_brow: float = 0.0
    right_brow: float = 0.0
    mouth_round: float = 0.0
    asymmetry: float = 0.0
    side_bias: float = 0.0


@dataclass
class VisemeTransformState:
    mouth_open: float = 0.0
    jaw_drop: float = 0.0
    mouth_wide: float = 0.0
    lip_round: float = 0.0
    smile: float = 0.0
    cheek: float = 0.0
    anticipation: float = 0.0
    recoil: float = 0.0
    lead_in: float = 0.0
    trail_out: float = 0.0


# ---------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------

class GazeSaccadeTransformer:
    """Gera sacadas suaves e micro-destinos em torno do alvo.

    A saída é estável: usa um pequeno órbita determinística e um filtro de
    primeira ordem para evitar tremor ou saltos abruptos.
    """

    def __init__(self):
        self.state = SaccadeState()

    def step(
        self,
        target_point: Tuple[float, float],
        face_center: Tuple[float, float],
        dt: float,
        *,
        focus: float = 0.6,
        curiosity: float = 0.4,
        reading: bool = False,
        cursor_speed: float = 0.0,
        silence: float = 0.0,
        speaking: bool = False,
        state_name: str = "idle",
    ) -> Tuple[float, float, Dict[str, float]]:
        tx, ty = _point(target_point)
        cx, cy = _point(face_center)
        dt = max(0.0, float(dt))
        s = self.state

        if abs(tx - s.target_x) > 18.0 or abs(ty - s.target_y) > 18.0:
            s.target_x = tx
            s.target_y = ty
            s.settle_timer = 0.0
            s.phase = 0.0
            s.orbit_phase = 0.0
        else:
            s.settle_timer += dt

        base_radius = 5.0 + 9.0 * clamp(curiosity, 0.0, 1.0) + 4.0 * clamp(focus, 0.0, 1.0)
        if reading:
            base_radius += 3.5
        if speaking:
            base_radius *= 0.82
        if state_name in {"thinking", "observe"}:
            base_radius *= 1.08
        if state_name in {"shy", "hide", "retreat"}:
            base_radius *= 0.84
        if cursor_speed > 1100.0:
            base_radius *= 0.76
        if silence > 8.0:
            base_radius *= 0.94

        s.phase += dt * (1.8 + 1.3 * clamp(focus, 0.0, 1.0) + 0.7 * clamp(curiosity, 0.0, 1.0))
        s.orbit_phase += dt * (2.6 + 0.9 * clamp(curiosity, 0.0, 1.0))

        lead = 0.18 * math.sin(s.phase * 1.7)
        orbit_x = math.cos(s.orbit_phase) * base_radius * 0.55 + math.sin(s.orbit_phase * 1.9) * base_radius * 0.12
        orbit_y = math.sin(s.orbit_phase * 0.93) * base_radius * 0.38 + math.cos(s.orbit_phase * 1.2) * base_radius * 0.08

        if state_name == "thinking":
            orbit_x *= 0.72
            orbit_y *= 0.82
        if state_name in {"speaking", "reacting"}:
            orbit_x *= 0.90
            orbit_y *= 0.90
        if reading:
            orbit_x += 2.0 * math.sin(s.phase * 2.2)

        desired_x = tx + orbit_x
        desired_y = ty + orbit_y

        # movimento normativo para normalized gaze (−1..1)
        gx_target = clamp((desired_x - cx) / 30.0, -1.0, 1.0)
        gy_target = clamp((desired_y - cy) / 28.0, -1.0, 1.0)

        if state_name in {"shy", "hide", "retreat"}:
            gx_target *= 0.84
            gy_target *= 0.90
        elif state_name in {"curious", "peek"}:
            gx_target *= 1.04
            gy_target *= 1.03

        s.gaze_x = exp_smooth(s.gaze_x, gx_target, dt, 0.045)
        s.gaze_y = exp_smooth(s.gaze_y, gy_target, dt, 0.045)

        meta = {
            "orbit_x": orbit_x,
            "orbit_y": orbit_y,
            "lead": lead,
            "radius": base_radius,
            "settle": clamp(s.settle_timer / 0.7, 0.0, 1.0),
        }
        return s.gaze_x, s.gaze_y, meta


class EmotionalBlinkTransformer:
    """Pestanejo contextual com fechamento/abertura suave."""

    def __init__(self):
        self.state = BlinkState()

    def step(
        self,
        dt: float,
        *,
        tension: float = 0.2,
        attention: float = 0.5,
        fatigue: float = 0.1,
        arousal: float = 0.4,
        speaking: bool = False,
        focus: bool = True,
        silence: float = 0.0,
        mouse_near: float = 0.0,
        state_name: str = "idle",
    ) -> float:
        dt = max(0.0, float(dt))
        s = self.state

        tension = clamp(float(tension), 0.0, 1.0)
        attention = clamp(float(attention), 0.0, 1.0)
        fatigue = clamp(float(fatigue), 0.0, 1.0)
        arousal = clamp(float(arousal), 0.0, 1.0)
        mouse_near = clamp(float(mouse_near), 0.0, 1.0)
        silence = clamp(float(silence), 0.0, 1.0)

        interval = 3.4
        interval += 1.6 * (1.0 - attention)
        interval -= 1.1 * tension
        interval -= 0.9 * fatigue
        interval += 0.6 * arousal
        interval += 0.25 if speaking else 0.0
        interval += 0.18 if focus else -0.20
        interval -= 0.35 * mouse_near
        interval += 0.15 * silence
        if state_name in {"thinking", "observe"}:
            interval -= 0.15
        if state_name in {"rest", "hide"}:
            interval += 0.20
        interval = clamp(interval, 1.0, 6.5)

        s.timer += dt
        if s.blink <= 0.0 and s.timer >= s.next_blink:
            s.blink = 0.02
            s.timer = 0.0
            s.close_phase = 0.0
            s.open_phase = 0.0
            # pesados quando cansaço é alto
            hold = 0.02 + 0.12 * fatigue + 0.02 * tension
            s.closed_hold = hold
            s.next_blink = interval + (0.75 + 0.65 * (1.0 - attention))

        if s.blink > 0.0:
            if s.closed_hold > 0.0:
                s.closed_hold = max(0.0, s.closed_hold - dt)
                s.blink = exp_smooth(s.blink, 1.0, dt, 0.018)
            else:
                if s.blink < 0.95:
                    s.blink = exp_smooth(s.blink, 1.0, dt, 0.028)
                else:
                    s.blink = exp_smooth(s.blink, 0.0, dt, 0.050 + 0.04 * fatigue)
                    if s.blink < 0.01:
                        s.blink = 0.0

        # não queremos jitter: suavizamos a saída final
        base = s.blink
        if state_name in {"focused", "thinking"}:
            base *= 0.92
        if state_name in {"tired", "rest"}:
            base *= 1.10
        if state_name in {"surprised", "reacting"}:
            base *= 0.82
        return clamp(base, 0.0, 1.0)


class AsymmetricSmirkTransformer:
    """Quebra a simetria da boca e sobrancelhas de modo sutil."""

    def __init__(self):
        self.state = SmirkState()

    def step(
        self,
        *,
        valence: float = 0.0,
        confidence: float = 0.5,
        curiosity: float = 0.4,
        tension: float = 0.2,
        mood: str = "neutral",
        speaking: bool = False,
        gesture: str = "none",
        intensity: float = 0.4,
    ) -> Dict[str, float]:
        valence = clamp(float(valence), -1.0, 1.0)
        confidence = clamp(float(confidence), 0.0, 1.0)
        curiosity = clamp(float(curiosity), 0.0, 1.0)
        tension = clamp(float(tension), 0.0, 1.0)
        intensity = clamp(float(intensity), 0.0, 1.0)

        side_bias = 0.22 * valence + 0.16 * confidence - 0.12 * tension + 0.08 * curiosity
        if mood in {"happy", "confident"}:
            side_bias += 0.06
        if mood in {"annoyed", "shy", "retreat"}:
            side_bias -= 0.05
        if gesture in {"nod", "present"}:
            side_bias += 0.03
        if speaking:
            side_bias *= 0.92

        side_bias = clamp(side_bias, -0.42, 0.42)
        self.state.side_bias = exp_smooth(self.state.side_bias, side_bias, 0.016, 0.10)

        left_curve_target = 0.12 + 0.48 * max(0.0, valence) + 0.10 * confidence - 0.06 * tension
        right_curve_target = 0.12 + 0.32 * max(0.0, valence) + 0.08 * confidence - 0.05 * tension
        if mood in {"curious", "focused"}:
            left_curve_target *= 0.92
            right_curve_target *= 0.98
        if mood in {"annoyed"}:
            left_curve_target -= 0.10
            right_curve_target -= 0.08

        if self.state.side_bias >= 0.0:
            left_curve_target += self.state.side_bias * 0.42
            right_curve_target -= self.state.side_bias * 0.18
            left_cheek_target = 0.08 + 0.12 * max(0.0, valence) + 0.06 * confidence
            right_cheek_target = 0.04 + 0.04 * max(0.0, valence)
        else:
            b = abs(self.state.side_bias)
            right_curve_target += b * 0.42
            left_curve_target -= b * 0.18
            right_cheek_target = 0.08 + 0.12 * max(0.0, valence) + 0.06 * confidence
            left_cheek_target = 0.04 + 0.04 * max(0.0, valence)

        mouth_round_target = 0.02 + 0.26 * max(0.0, tension) + 0.08 * (1.0 - confidence)
        if speaking:
            mouth_round_target *= 0.65

        self.state.left_curve = exp_smooth(self.state.left_curve, clamp(left_curve_target, -0.4, 0.9), 0.016, 0.12)
        self.state.right_curve = exp_smooth(self.state.right_curve, clamp(right_curve_target, -0.4, 0.9), 0.016, 0.12)
        self.state.left_cheek = exp_smooth(self.state.left_cheek, clamp(left_cheek_target, 0.0, 0.5), 0.016, 0.12)
        self.state.right_cheek = exp_smooth(self.state.right_cheek, clamp(right_cheek_target, 0.0, 0.5), 0.016, 0.12)
        self.state.left_brow = exp_smooth(self.state.left_brow, clamp(-0.08 * valence + 0.10 * tension, -0.4, 0.4), 0.016, 0.12)
        self.state.right_brow = exp_smooth(self.state.right_brow, clamp(-0.06 * valence + 0.08 * tension, -0.4, 0.4), 0.016, 0.12)
        self.state.mouth_round = exp_smooth(self.state.mouth_round, clamp(mouth_round_target, 0.0, 0.5), 0.016, 0.12)
        self.state.asymmetry = self.state.side_bias

        return {
            "mouth_curve_left": self.state.left_curve,
            "mouth_curve_right": self.state.right_curve,
            "mouth_round": self.state.mouth_round,
            "cheek_left": self.state.left_cheek,
            "cheek_right": self.state.right_cheek,
            "brow_left": self.state.left_brow,
            "brow_right": self.state.right_brow,
            "asymmetry": self.state.asymmetry,
            "side_bias": self.state.side_bias,
        }


class PhonemeToVisemeTransformer:
    """Refina a boca com antecipação e coarticulação suave."""

    def __init__(self):
        self.state = VisemeTransformState()

    def step(
        self,
        *,
        current: Optional[Dict[str, Any]] = None,
        next_viseme: Optional[Dict[str, Any]] = None,
        local: float = 0.0,
        speaking: bool = True,
        audio_energy: float = 0.0,
    ) -> Dict[str, float]:
        cur = current or {}
        nxt = next_viseme or cur
        local = clamp(float(local), 0.0, 1.0)
        energy = clamp(float(audio_energy), 0.0, 1.0)

        mouth_open = clamp(float(cur.get("mouth_open", 0.05)), 0.0, 1.0)
        jaw_drop = clamp(float(cur.get("jaw_drop", 0.03)), 0.0, 1.0)
        mouth_wide = clamp(float(cur.get("mouth_wide", 0.10)), 0.0, 1.0)
        lip_round = clamp(float(cur.get("lip_round", 0.0)), 0.0, 1.0)
        smile = clamp(float(cur.get("smile", 0.0)), -0.25, 0.25)
        cheek = clamp(float(cur.get("cheek", 0.0)), 0.0, 0.5)

        nxt_open = clamp(float(nxt.get("mouth_open", mouth_open)), 0.0, 1.0)
        nxt_round = clamp(float(nxt.get("lip_round", lip_round)), 0.0, 1.0)
        nxt_wide = clamp(float(nxt.get("mouth_wide", mouth_wide)), 0.0, 1.0)

        anticipation = 0.22 * (1.0 - local)
        recoil = 0.14 * local
        lead_in = smoothstep(0.0, 0.45, 1.0 - local)
        trail_out = smoothstep(0.55, 1.0, local)

        # plosivas fecham um pouco antes do som
        if cur.get("viseme") in {"MBP", "FV"}:
            mouth_open *= 0.82 + 0.10 * lead_in
            jaw_drop *= 0.72 + 0.12 * lead_in
        if nxt.get("viseme") in {"MBP", "FV"}:
            mouth_open = mouth_open * (0.90 - 0.12 * trail_out) + nxt_open * (0.10 + 0.12 * trail_out)
        if nxt.get("viseme") in {"O", "U"}:
            lip_round = max(lip_round, nxt_round * (0.55 + 0.25 * lead_in))
        if nxt.get("viseme") in {"A", "E", "I"}:
            mouth_wide = max(mouth_wide, nxt_wide * (0.55 + 0.30 * lead_in))

        mouth_open = mouth_open * (0.66 + 0.22 * energy) + energy * 0.22 + anticipation * 0.08
        jaw_drop = jaw_drop * (0.72 + 0.16 * energy) + energy * 0.18
        mouth_wide = mouth_wide * (0.80 + 0.10 * energy) + nxt_wide * 0.10
        lip_round = lip_round * 0.86 + nxt_round * 0.14
        cheek = cheek * 0.82 + 0.04 * energy
        smile += 0.03 * (1.0 - lip_round)

        self.state.mouth_open = exp_smooth(self.state.mouth_open, clamp(mouth_open, 0.0, 1.0), 0.016, 0.06)
        self.state.jaw_drop = exp_smooth(self.state.jaw_drop, clamp(jaw_drop, 0.0, 1.0), 0.016, 0.06)
        self.state.mouth_wide = exp_smooth(self.state.mouth_wide, clamp(mouth_wide, 0.0, 1.0), 0.016, 0.08)
        self.state.lip_round = exp_smooth(self.state.lip_round, clamp(lip_round, 0.0, 1.0), 0.016, 0.08)
        self.state.smile = exp_smooth(self.state.smile, clamp(smile, -0.25, 0.25), 0.016, 0.08)
        self.state.cheek = exp_smooth(self.state.cheek, clamp(cheek, 0.0, 0.5), 0.016, 0.08)
        self.state.anticipation = exp_smooth(self.state.anticipation, anticipation, 0.016, 0.08)
        self.state.recoil = exp_smooth(self.state.recoil, recoil, 0.016, 0.08)
        self.state.lead_in = exp_smooth(self.state.lead_in, lead_in, 0.016, 0.08)
        self.state.trail_out = exp_smooth(self.state.trail_out, trail_out, 0.016, 0.08)

        return {
            "mouth_open": self.state.mouth_open,
            "jaw_drop": self.state.jaw_drop,
            "mouth_wide": self.state.mouth_wide,
            "lip_round": self.state.lip_round,
            "smile": self.state.smile,
            "cheek": self.state.cheek,
            "anticipation": self.state.anticipation,
            "recoil": self.state.recoil,
            "lead_in": self.state.lead_in,
            "trail_out": self.state.trail_out,
        }


# ---------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------

def _resolve(obj: Any, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


def _patch_method(cls: Any, name: str, factory) -> None:
    if cls is None or getattr(cls, f"_microexpr_{name}_patched", False):
        return
    orig = getattr(cls, name, None)
    if orig is None:
        return
    setattr(cls, name, factory(orig))
    setattr(cls, f"_microexpr_{name}_patched", True)


def _face_attr(face: Any, name: str, default: float = 0.0) -> float:
    return _to_float(getattr(face, name, default), default)


def _overlay_asym_mouth(img, cx: float, cy: float, face: Any, *, alpha_scale: float = 1.0):
    try:
        from PIL import ImageDraw, ImageFilter
    except Exception:
        return img

    draw = ImageDraw.Draw(img)
    width = 82.0 * max(0.65, _face_attr(face, "mouth_width", 1.0))
    open_amt = clamp(_face_attr(face, "mouth_open", 0.05), 0.0, 1.0)
    round_amt = clamp(_face_attr(face, "mouth_round", 0.0), 0.0, 1.0)
    curve_l = _face_attr(face, "mouth_curve_left", _face_attr(face, "mouth_curve", 0.0))
    curve_r = _face_attr(face, "mouth_curve_right", _face_attr(face, "mouth_curve", 0.0))
    cheek_l = _face_attr(face, "cheek_left", _face_attr(face, "cheek_lift", 0.0))
    cheek_r = _face_attr(face, "cheek_right", _face_attr(face, "cheek_lift", 0.0))
    smirk = clamp(_face_attr(face, "smirk_bias", _face_attr(face, "mouth_asymmetry", 0.0)), -0.6, 0.6)

    y = cy + 102.0
    mouth_h = 18.0 + 38.0 * open_amt
    pts = []
    for i in range(-9, 10):
        x = cx + i * 8
        u = i / 9.0
        arch = math.sin((u + 1.0) * math.pi * 0.5)
        curve = curve_l if u < 0.0 else curve_r
        cheek = cheek_l if u < 0.0 else cheek_r
        bias = smirk if u >= 0.0 else -smirk * 0.45
        yy = y + curve * 18.0 * arch - open_amt * 8.0 * (1.0 - abs(u)) - round_amt * 4.0 * (1.0 - abs(u))
        yy += bias * 3.5 * (1.0 - abs(u)) - cheek * 2.0
        pts.append((x, yy))

    # subtle lip shadow and line
    draw.ellipse((cx - width - 12, y - 4, cx + width + 12, y + mouth_h + 18), fill=(30, 0, 0, int(28 * alpha_scale)))
    if open_amt > 0.06:
        draw.ellipse((cx - width * 0.7, y + 6, cx + width * 0.7, y + mouth_h + 10), fill=(42, 12, 16, int(220 * alpha_scale)))
    draw.line(pts, fill=(112, 52, 64, int(255 * alpha_scale)), width=8)
    return img


def install_facial_micro_expression_transformers(module_globals: Optional[Dict[str, Any]] = None) -> None:
    """Patch any known avatar classes present in the provided globals dict."""
    if not module_globals:
        return

    motion_classes = [module_globals.get("MotionEngine"), module_globals.get("MotionEngineV2")]
    attention_classes = [module_globals.get("AttentionEngine"), module_globals.get("AttentionEngineV2")]
    renderer_classes = [module_globals.get("AvatarRenderer"), module_globals.get("AvatarRendererV2")]
    audio_classes = [module_globals.get("AudioSyncEngine"), module_globals.get("AudioSyncEngineV2")]

    # --------------------------------------------------
    # Attention: gaze saccades + blink context
    # --------------------------------------------------
    for cls in attention_classes:
        if cls is None or getattr(cls, "_microexpr_attention_patched", False):
            continue

        _patch_method(cls, "update_gaze", lambda orig: (
            lambda self, *args, **kwargs: _attention_update_gaze(self, orig, *args, **kwargs)
        ))
        _patch_method(cls, "update_blink", lambda orig: (
            lambda self, *args, **kwargs: _attention_update_blink(self, orig, *args, **kwargs)
        ))
        setattr(cls, "_microexpr_attention_patched", True)

    # --------------------------------------------------
    # Motion: mouth asymmetry + gaze context injection
    # --------------------------------------------------
    for cls in motion_classes:
        if cls is None or getattr(cls, "_microexpr_motion_patched", False):
            continue
        if hasattr(cls, "update"):
            _patch_method(cls, "update", lambda orig: (
                lambda self, *args, **kwargs: _motion_update_wrapper(self, orig, *args, **kwargs)
            ))
        if hasattr(cls, "forward"):
            _patch_method(cls, "forward", lambda orig: (
                lambda self, *args, **kwargs: _motion_forward_wrapper(self, orig, *args, **kwargs)
            ))
        setattr(cls, "_microexpr_motion_patched", True)

    # --------------------------------------------------
    # Renderer: optional asym mouth overlay
    # --------------------------------------------------
    for cls in renderer_classes:
        if cls is None or getattr(cls, "_microexpr_renderer_patched", False):
            continue
        _patch_method(cls, "render", lambda orig: (
            lambda self, *args, **kwargs: _renderer_render_wrapper(self, orig, *args, **kwargs)
        ))
        setattr(cls, "_microexpr_renderer_patched", True)

    # --------------------------------------------------
    # Audio proxy: keep the mouth coherent with visemes
    # --------------------------------------------------
    for cls in audio_classes:
        if cls is None or getattr(cls, "_microexpr_audio_patched", False):
            continue
        if hasattr(cls, "mouth_open"):
            _patch_method(cls, "mouth_open", lambda orig: (
                lambda self, *args, **kwargs: _audio_mouth_open_wrapper(self, orig, *args, **kwargs)
            ))
        setattr(cls, "_microexpr_audio_patched", True)


# ---------------------------------------------------------------------
# Wrapped behaviors
# ---------------------------------------------------------------------

def _get_ctx(obj: Any) -> Dict[str, Any]:
    return getattr(obj, "_microexpr_ctx", {}) or {}


def _attention_update_gaze(self, orig, *args, **kwargs):
    # preserve original behavior if anything fails
    try:
        result = orig(self, *args, **kwargs)
    except Exception:
        result = (0.0, 0.0)

    dt = _to_float(args[0] if args else kwargs.get("dt", 0.016), 0.016)
    target_point = _point(args[1] if len(args) > 1 else kwargs.get("target_point", (0.0, 0.0)))
    face_center = _point(args[2] if len(args) > 2 else kwargs.get("face_center", (0.0, 0.0)))
    ctx = _get_ctx(self)

    trans = getattr(self, "_gaze_saccade_transformer", None)
    if trans is None:
        trans = GazeSaccadeTransformer()
        setattr(self, "_gaze_saccade_transformer", trans)

    gaze_x, gaze_y, _meta = trans.step(
        target_point,
        face_center,
        dt,
        focus=_to_float(ctx.get("focus", 0.6), 0.6),
        curiosity=_to_float(ctx.get("curiosity", 0.4), 0.4),
        reading=bool(ctx.get("reading", False)),
        cursor_speed=_to_float(ctx.get("mouse_speed", 0.0), 0.0),
        silence=_to_float(ctx.get("silence", 0.0), 0.0),
        speaking=bool(ctx.get("speaking", False)),
        state_name=str(ctx.get("state_name", "idle")),
    )
    try:
        self.gaze_x = gaze_x
        self.gaze_y = gaze_y
    except Exception:
        pass
    return gaze_x, gaze_y


def _attention_update_blink(self, orig, *args, **kwargs):
    try:
        result = orig(self, *args, **kwargs)
    except Exception:
        result = 0.0

    dt = _to_float(args[0] if args else kwargs.get("dt", 0.016), 0.016)
    ctx = _get_ctx(self)
    state_obj = args[1] if len(args) > 1 else kwargs.get("state") or kwargs.get("emotion") or None
    # pull a few soft values from state object if present
    tension = _to_float(getattr(state_obj, "tension", ctx.get("tension", 0.2)), _to_float(ctx.get("tension", 0.2), 0.2))
    attention = _to_float(getattr(state_obj, "attention", ctx.get("attention", 0.5)), _to_float(ctx.get("attention", 0.5), 0.5))
    fatigue = _to_float(getattr(state_obj, "fatigue", ctx.get("fatigue", 0.1)), _to_float(ctx.get("fatigue", 0.1), 0.1))
    arousal = _to_float(getattr(state_obj, "arousal", ctx.get("arousal", 0.4)), _to_float(ctx.get("arousal", 0.4), 0.4))
    speaking = bool(ctx.get("speaking", False))
    focus = bool(ctx.get("focus", True))
    silence = _to_float(ctx.get("silence", 0.0), 0.0)
    mouse_near = _to_float(ctx.get("mouse_near", 0.0), 0.0)
    state_name = str(ctx.get("state_name", getattr(state_obj, "current_state", "idle")))

    trans = getattr(self, "_emotional_blink_transformer", None)
    if trans is None:
        trans = EmotionalBlinkTransformer()
        setattr(self, "_emotional_blink_transformer", trans)

    blink = trans.step(
        dt,
        tension=tension,
        attention=attention,
        fatigue=fatigue,
        arousal=arousal,
        speaking=speaking,
        focus=focus,
        silence=silence,
        mouse_near=mouse_near,
        state_name=state_name,
    )
    try:
        self.blink_amount = blink
    except Exception:
        pass
    return blink


def _motion_update_wrapper(self, orig, *args, **kwargs):
    # Set a context that attention/blink/gaze wrappers can read
    try:
        perception = args[0] if args else kwargs.get("perception")
        state = args[1] if len(args) > 1 else kwargs.get("state")
        behavior = args[2] if len(args) > 2 else kwargs.get("behavior")
        attention = args[3] if len(args) > 3 else kwargs.get("attention")
        plan = args[4] if len(args) > 4 else kwargs.get("plan")
        dt = args[5] if len(args) > 5 else kwargs.get("dt", 0.016)
        ctx = {
            "state_name": getattr(behavior, "current_state", getattr(state, "current_state", "idle")),
            "curiosity": _to_float(getattr(state, "curiosity", 0.4), 0.4),
            "focus": _to_float(getattr(state, "attention", 0.5), 0.5),
            "tension": _to_float(getattr(state, "tension", 0.2), 0.2),
            "fatigue": _to_float(getattr(state, "fatigue", 0.1), 0.1),
            "arousal": _to_float(getattr(state, "arousal", 0.4), 0.4),
            "speaking": bool(getattr(perception, "speaking", False) if perception is not None else False),
            "mouse_speed": _to_float(getattr(perception, "mouse_speed", 0.0), 0.0),
            "silence": _to_float(getattr(perception, "silence_s", 0.0), 0.0),
            "reading": bool(getattr(perception, "user_text_len", 0) > 12),
        }
        if attention is not None:
            setattr(attention, "_microexpr_ctx", ctx)
            try:
                ctx["mouse_near"] = clamp(1.0 - _to_float(getattr(perception, "mouse_distance_to_face", 999.0), 999.0) / 280.0, 0.0, 1.0)
            except Exception:
                ctx["mouse_near"] = 0.0
    except Exception:
        pass

    out = orig(self, *args, **kwargs)

    try:
        perception = args[0] if args else kwargs.get("perception")
        state = args[1] if len(args) > 1 else kwargs.get("state")
        behavior = args[2] if len(args) > 2 else kwargs.get("behavior")
        attention = args[3] if len(args) > 3 else kwargs.get("attention")
        plan = args[4] if len(args) > 4 else kwargs.get("plan")
        ctx = _get_ctx(attention)

        smirk = getattr(self, "_asymmetric_smirk_transformer", None)
        if smirk is None:
            smirk = AsymmetricSmirkTransformer()
            setattr(self, "_asymmetric_smirk_transformer", smirk)

        mood = str(getattr(state, "emotion", getattr(state, "current_state", "neutral")))
        smirk_out = smirk.step(
            valence=_to_float(getattr(state, "valence", 0.0), 0.0),
            confidence=_to_float(getattr(state, "confidence", 0.5), 0.5),
            curiosity=_to_float(getattr(state, "curiosity", 0.4), 0.4),
            tension=_to_float(getattr(state, "tension", 0.2), 0.2),
            mood=mood,
            speaking=bool(getattr(perception, "speaking", False)),
            gesture=str(getattr(behavior, "last_micro_action", "none")),
            intensity=_to_float(getattr(state, "intensity", 0.4), 0.4),
        )

        if isinstance(out, dict):
            face = out.get("face")
            if isinstance(face, dict):
                face["mouth_curve_left"] = smirk_out["mouth_curve_left"]
                face["mouth_curve_right"] = smirk_out["mouth_curve_right"]
                face["mouth_round"] = smirk_out["mouth_round"]
                face["cheek_left"] = smirk_out["cheek_left"]
                face["cheek_right"] = smirk_out["cheek_right"]
                face["brow_left"] = smirk_out["brow_left"]
                face["brow_right"] = smirk_out["brow_right"]
                face["smirk_bias"] = smirk_out["asymmetry"]
            out["blink_amount"] = clamp(_to_float(out.get("blink_amount", 0.0), 0.0) * (0.92 + 0.08 * smirk_out["asymmetry"]) + _to_float(ctx.get("blink", 0.0), 0.0), 0.0, 1.0)
            out["micro_expression"] = smirk_out
            out["gaze_meta"] = ctx
        elif torch is not None and isinstance(out, torch.Tensor):
            # nudge body/posture a little for style; keep it smooth and subtle
            out = out.clone()
            if out.numel() >= 2:
                out[0] = out[0] + out.new_tensor(smirk_out["asymmetry"] * 0.04)
                out[1] = out[1] + out.new_tensor(smirk_out["mouth_round"] * 0.02)
            if out.numel() >= 4:
                out[3] = out[3] * out.new_tensor(0.98 + 0.02 * (1.0 - abs(smirk_out["asymmetry"])))
        return out
    except Exception:
        return out


def _motion_forward_wrapper(self, orig, *args, **kwargs):
    out = orig(self, *args, **kwargs)
    try:
        if torch is not None and isinstance(out, torch.Tensor):
            if out.numel() >= 6:
                out = out.clone()
                # smooth, non-jittery extras
                out[0] = out[0] * 0.995
                out[1] = out[1] * 0.995
                out[2] = out[2] * 0.997
                out[3] = out[3] * 0.998
                out[4] = out[4] * 0.998
                out[5] = out[5] * 0.996
        return out
    except Exception:
        return out


def _renderer_render_wrapper(self, orig, *args, **kwargs):
    img = orig(self, *args, **kwargs)
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img

    # Runtime renderer signature: (t, perception, emotion, pose, face, audio, plan, blink_amount=0.0)
    # Core renderer signature:     (t, perception, state, motion, attention, audio, plan)
    face = None
    motion = None
    pose = None
    state = None
    if len(args) >= 5:
        if hasattr(args[3], "gaze_x") and hasattr(args[4], "mouth_open"):
            pose = args[3]
            face = args[4]
            state = args[2] if len(args) > 2 else None
        elif isinstance(args[3], dict):
            motion = args[3]
            state = args[2] if len(args) > 2 else None

    if face is None and motion is None:
        return img

    try:
        overlay = img.copy()
        if face is not None:
            cx = getattr(self, "cx", img.width // 2)
            cy = getattr(self, "cy", img.height // 2)
            head_x = cx + _to_float(getattr(pose, "head_x", 0.0), 0.0) * 12.0
            head_y = cy - 15.0 + _to_float(getattr(pose, "head_y", 0.0), 0.0) * 10.0
            _overlay_asym_mouth(overlay, head_x, head_y, face, alpha_scale=0.96)
        elif motion is not None:
            cx = getattr(self, "cx", img.width // 2)
            cy = getattr(self, "cy", img.height // 2)
            head_x = cx + _to_float(motion.get("head_x", 0.0), 0.0) * 12.0
            head_y = cy - 15.0 + _to_float(motion.get("head_y", 0.0), 0.0) * 10.0
            class _Tmp:
                pass
            tmp = _Tmp()
            for k, v in motion.items():
                if isinstance(v, (int, float)):
                    setattr(tmp, k, v)
            _overlay_asym_mouth(overlay, head_x, head_y, tmp, alpha_scale=0.92)
        # Soft blend to keep the effect integrated and smooth
        return Image.blend(img, overlay, 0.22)
    except Exception:
        return img


def _audio_mouth_open_wrapper(self, orig, *args, **kwargs):
    base = orig(self, *args, **kwargs)
    try:
        viseme = getattr(self, "viseme_snapshot", None)
        if callable(viseme):
            snap = viseme() or {}
            strength = _to_float(snap.get("strength", 0.0), 0.0)
            base = clamp(_to_float(base, 0.0) * (1.0 - 0.20 * strength) + _to_float(snap.get("mouth_open", 0.0), 0.0) * 0.20, 0.0, 1.0)
    except Exception:
        pass
    return base


__all__ = [
    "GazeSaccadeTransformer",
    "EmotionalBlinkTransformer",
    "AsymmetricSmirkTransformer",
    "PhonemeToVisemeTransformer",
    "install_facial_micro_expression_transformers",
]
