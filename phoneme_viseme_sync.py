from __future__ import annotations

import copy
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

VISEME_TEMPLATES: Dict[str, Dict[str, float]] = {
    "neutral": {"mouth_open": 0.05, "mouth_wide": 0.10, "lip_round": 0.00, "jaw_drop": 0.03, "teeth": 0.00, "smile": 0.00, "tongue_up": 0.00, "cheek": 0.00, "lips_together": 0.62, "lip_corner_up": 0.00, "lip_corner_down": 0.00, "jaw_clench": 0.05, "tongue_out": 0.00, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.00, "lower_lip_depress": 0.00, "cheek_puff": 0.00, "cheek_suck": 0.00, "mouth_corner_stretch": 0.06},
    "A": {"mouth_open": 0.92, "mouth_wide": 0.22, "lip_round": 0.00, "jaw_drop": 0.92, "teeth": 0.04, "smile": 0.00, "tongue_up": 0.08, "cheek": 0.00, "lips_together": 0.04, "lip_corner_up": 0.02, "lip_corner_down": 0.00, "jaw_clench": 0.02, "tongue_out": 0.02, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.00, "lower_lip_depress": 0.10, "cheek_puff": 0.00, "cheek_suck": 0.00, "mouth_corner_stretch": 0.10},
    "E": {"mouth_open": 0.45, "mouth_wide": 0.92, "lip_round": 0.00, "jaw_drop": 0.30, "teeth": 0.34, "smile": 0.11, "tongue_up": 0.12, "cheek": 0.00, "lips_together": 0.10, "lip_corner_up": 0.10, "lip_corner_down": 0.00, "jaw_clench": 0.02, "tongue_out": 0.03, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.10, "lower_lip_depress": 0.02, "cheek_puff": 0.00, "cheek_suck": 0.04, "mouth_corner_stretch": 0.20},
    "I": {"mouth_open": 0.25, "mouth_wide": 1.00, "lip_round": 0.00, "jaw_drop": 0.16, "teeth": 0.44, "smile": 0.09, "tongue_up": 0.18, "cheek": 0.00, "lips_together": 0.12, "lip_corner_up": 0.08, "lip_corner_down": 0.00, "jaw_clench": 0.02, "tongue_out": 0.02, "tongue_tip_interdental": 0.02, "upper_lip_raise": 0.12, "lower_lip_depress": 0.00, "cheek_puff": 0.00, "cheek_suck": 0.02, "mouth_corner_stretch": 0.22},
    "O": {"mouth_open": 0.42, "mouth_wide": 0.18, "lip_round": 0.98, "jaw_drop": 0.26, "teeth": 0.06, "smile": -0.02, "tongue_up": 0.04, "cheek": 0.02, "lips_together": 0.08, "lip_corner_up": 0.00, "lip_corner_down": 0.06, "jaw_clench": 0.04, "tongue_out": 0.00, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.00, "lower_lip_depress": 0.04, "cheek_puff": 0.08, "cheek_suck": 0.00, "mouth_corner_stretch": 0.04},
    "U": {"mouth_open": 0.33, "mouth_wide": 0.08, "lip_round": 1.00, "jaw_drop": 0.18, "teeth": 0.03, "smile": -0.02, "tongue_up": 0.04, "cheek": 0.02, "lips_together": 0.05, "lip_corner_up": 0.00, "lip_corner_down": 0.04, "jaw_clench": 0.03, "tongue_out": 0.00, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.00, "lower_lip_depress": 0.02, "cheek_puff": 0.10, "cheek_suck": 0.00, "mouth_corner_stretch": 0.02},
    "MBP": {"mouth_open": 0.02, "mouth_wide": 0.02, "lip_round": 0.00, "jaw_drop": 0.00, "teeth": 0.00, "smile": 0.00, "tongue_up": 0.00, "cheek": 0.00, "lips_together": 1.00, "lip_corner_up": 0.00, "lip_corner_down": 0.00, "jaw_clench": 0.48, "tongue_out": 0.00, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.00, "lower_lip_depress": 0.00, "cheek_puff": 0.00, "cheek_suck": 0.00, "mouth_corner_stretch": 0.00},
    "FV": {"mouth_open": 0.12, "mouth_wide": 0.24, "lip_round": 0.00, "jaw_drop": 0.08, "teeth": 0.44, "smile": 0.00, "tongue_up": 0.04, "cheek": 0.00, "lips_together": 0.14, "lip_corner_up": 0.00, "lip_corner_down": 0.10, "jaw_clench": 0.10, "tongue_out": 0.00, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.08, "lower_lip_depress": 0.06, "cheek_puff": 0.00, "cheek_suck": 0.00, "mouth_corner_stretch": 0.08},
    "LNTD": {"mouth_open": 0.27, "mouth_wide": 0.46, "lip_round": 0.00, "jaw_drop": 0.22, "teeth": 0.18, "smile": 0.00, "tongue_up": 0.46, "cheek": 0.00, "lips_together": 0.22, "lip_corner_up": 0.02, "lip_corner_down": 0.02, "jaw_clench": 0.08, "tongue_out": 0.08, "tongue_tip_interdental": 0.72, "upper_lip_raise": 0.04, "lower_lip_depress": 0.06, "cheek_puff": 0.00, "cheek_suck": 0.02, "mouth_corner_stretch": 0.12},
    "SZCHJ": {"mouth_open": 0.14, "mouth_wide": 0.70, "lip_round": 0.00, "jaw_drop": 0.09, "teeth": 0.54, "smile": 0.02, "tongue_up": 0.10, "cheek": 0.00, "lips_together": 0.16, "lip_corner_up": 0.04, "lip_corner_down": 0.00, "jaw_clench": 0.05, "tongue_out": 0.01, "tongue_tip_interdental": 0.04, "upper_lip_raise": 0.08, "lower_lip_depress": 0.02, "cheek_puff": 0.00, "cheek_suck": 0.04, "mouth_corner_stretch": 0.18},
    "RRGK": {"mouth_open": 0.34, "mouth_wide": 0.28, "lip_round": 0.18, "jaw_drop": 0.26, "teeth": 0.10, "smile": 0.00, "tongue_up": 0.10, "cheek": 0.00, "lips_together": 0.16, "lip_corner_up": 0.00, "lip_corner_down": 0.04, "jaw_clench": 0.06, "tongue_out": 0.03, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.02, "lower_lip_depress": 0.02, "cheek_puff": 0.00, "cheek_suck": 0.00, "mouth_corner_stretch": 0.06},
    "NASAL": {"mouth_open": 0.06, "mouth_wide": 0.06, "lip_round": 0.00, "jaw_drop": 0.02, "teeth": 0.00, "smile": 0.00, "tongue_up": 0.00, "cheek": 0.00, "lips_together": 0.86, "lip_corner_up": 0.00, "lip_corner_down": 0.00, "jaw_clench": 0.08, "tongue_out": 0.00, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.00, "lower_lip_depress": 0.00, "cheek_puff": 0.00, "cheek_suck": 0.00, "mouth_corner_stretch": 0.00},
    "SILENCE": {"mouth_open": 0.02, "mouth_wide": 0.08, "lip_round": 0.00, "jaw_drop": 0.00, "teeth": 0.00, "smile": 0.00, "tongue_up": 0.00, "cheek": 0.00, "lips_together": 0.92, "lip_corner_up": 0.00, "lip_corner_down": 0.00, "jaw_clench": 0.02, "tongue_out": 0.00, "tongue_tip_interdental": 0.00, "upper_lip_raise": 0.00, "lower_lip_depress": 0.00, "cheek_puff": 0.00, "cheek_suck": 0.00, "mouth_corner_stretch": 0.00},
}
PHONEME_TO_VISEME: Dict[str, str] = {
    "a": "A", "á": "A", "à": "A", "â": "A", "ã": "A",
    "e": "E", "é": "E", "ê": "E",
    "i": "I", "í": "I",
    "o": "O", "ó": "O", "ô": "O", "õ": "O",
    "u": "U", "ú": "U",
    "m": "MBP", "p": "MBP", "b": "MBP",
    "f": "FV", "v": "FV",
    "l": "LNTD", "n": "LNTD", "d": "LNTD", "t": "LNTD", "lh": "LNTD", "nh": "LNTD",
    "s": "SZCHJ", "z": "SZCHJ", "c": "SZCHJ", "ç": "SZCHJ", "j": "SZCHJ", "x": "SZCHJ", "ch": "SZCHJ",
    "r": "RRGK", "rr": "RRGK", "g": "RRGK", "k": "RRGK", "q": "RRGK", "gu": "RRGK", "qu": "RRGK",
    "ã": "NASAL", "õ": "NASAL",
}

WORD_RE = re.compile(r"[A-Za-zÀ-ÿÇç]+(?:'[A-Za-zÀ-ÿÇç]+)?|\d+|[^\w\s]", re.UNICODE)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").replace("\u200b", " ")).strip()


def _blend_dict(a: Dict[str, float], b: Dict[str, float], t: float) -> Dict[str, float]:
    t = _clamp(t, 0.0, 1.0)
    keys = set(a) | set(b)
    return {k: _lerp(float(a.get(k, 0.0)), float(b.get(k, 0.0)), t) for k in keys}


def _token_viseme(token: str) -> str:
    token = (token or "").lower()
    return PHONEME_TO_VISEME.get(token, PHONEME_TO_VISEME.get(token[:1], "neutral"))


def _token_weight(token: str, viseme: str, is_vowel: bool) -> float:
    base = {"A": 1.28, "E": 1.16, "I": 1.10, "O": 1.20, "U": 1.12, "MBP": 0.72, "FV": 0.74, "LNTD": 0.82, "SZCHJ": 0.76, "RRGK": 0.78, "NASAL": 0.68, "SILENCE": 0.25}.get(viseme, 0.55)
    return max(0.05, base * (1.18 if is_vowel else 1.0))


def _simple_grapheme_tokens(word: str) -> List[Tuple[str, bool]]:
    w = unicodedata.normalize("NFC", word.lower())
    out: List[Tuple[str, bool]] = []
    i = 0
    while i < len(w):
        ch = w[i]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch.isspace():
            i += 1
            continue
        pair = ch + nxt
        if pair in {"ch", "lh", "nh", "rr", "ss"}:
            out.append((pair, False))
            i += 2
            continue
        if pair == "qu" and i + 2 < len(w) and w[i + 2] in "aeiouáéíóúâêôàãõ":
            out.append(("qu", False))
            i += 2
            continue
        if pair == "gu" and i + 2 < len(w) and w[i + 2] in "eiéêí":
            out.append(("gu", False))
            i += 2
            continue
        if ch in "aeiouáéíóúâêôàãõ":
            out.append((ch, True))
        elif ch.isalpha() or ch in "ç":
            out.append((ch, False))
        i += 1
    return out


@dataclass
class VisemeFrame:
    start: float
    end: float
    phoneme: str
    viseme: str
    weight: float = 1.0
    confidence: float = 1.0
    blend_in: float = 0.06
    blend_out: float = 0.08
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class VisemeSnapshot:
    name: str = "neutral"
    active: bool = False
    strength: float = 0.0
    time: float = 0.0
    duration: float = 0.0
    mouth_open: float = 0.05
    mouth_wide: float = 0.10
    lip_round: float = 0.0
    jaw_drop: float = 0.03
    teeth: float = 0.0
    smile: float = 0.0
    tongue_up: float = 0.0
    cheek: float = 0.0
    lips_together: float = 0.0
    lip_corner_up: float = 0.0
    lip_corner_down: float = 0.0
    jaw_clench: float = 0.0
    tongue_out: float = 0.0
    tongue_tip_interdental: float = 0.0
    upper_lip_raise: float = 0.0
    lower_lip_depress: float = 0.0
    cheek_puff: float = 0.0
    cheek_suck: float = 0.0
    mouth_corner_stretch: float = 0.0
    blink_suppression: float = 1.0
    confidence: float = 1.0
    phoneme: str = ""
    body_scale_hint: float = 1.0
    visibility_alpha: float = 1.0
    z_order_hint: float = 0.5
    layer_mode: str = "respect_space"

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class VisemeTransitionFilter:
    def __init__(self):
        self.state: Dict[str, float] = {}
        self.energy_gate = 0.0
        self.alpha_gate = 1.0
        self.last_name = 'neutral'

    def reset(self) -> None:
        self.state.clear()
        self.energy_gate = 0.0
        self.alpha_gate = 1.0
        self.last_name = 'neutral'

    def apply(self, params: Dict[str, float], *, speaking: bool, audio_energy: float, body_scale: float, visibility_alpha: float, current_viseme: str) -> Dict[str, float]:
        energy = _clamp(float(audio_energy), 0.0, 1.0)
        body = _clamp(float(body_scale), 0.70, 1.55)
        alpha = _clamp(float(visibility_alpha), 0.0, 1.0)
        self.energy_gate = _lerp(self.energy_gate, energy, 0.12)
        self.alpha_gate = _lerp(self.alpha_gate, alpha, 0.12)
        out: Dict[str, float] = {}
        for key, value in params.items():
            target = float(value)
            current = self.state.get(key, target)
            tau = 0.09 if key in {'mouth_open', 'jaw_drop'} else 0.12 if key in {'mouth_wide', 'lip_round'} else 0.16
            if not speaking:
                tau *= 1.35
            smooth = _lerp(current, target, _clamp(1.0 - math.exp(-0.016 / tau), 0.0, 1.0))
            self.state[key] = smooth
            out[key] = smooth
        out['mouth_open'] = _clamp(out.get('mouth_open', 0.0) * (0.96 + 0.04 * body) + 0.04 * self.energy_gate, 0.0, 1.0)
        out['jaw_drop'] = _clamp(out.get('jaw_drop', 0.0) * (0.94 + 0.06 * body) + 0.03 * self.energy_gate, 0.0, 1.0)
        out['smile'] = _clamp(out.get('smile', 0.0) + 0.04 * (body - 1.0), -0.2, 0.5)
        out['cheek'] = _clamp(out.get('cheek', 0.0) + 0.03 * (self.energy_gate if speaking else 0.0), 0.0, 0.4)
        out['lips_together'] = _clamp(out.get('lips_together', 0.0) * (0.96 + 0.04 * (1.0 - body)) + 0.02 * (1.0 - self.energy_gate), 0.0, 1.0)
        out['lip_corner_up'] = _clamp(out.get('lip_corner_up', 0.0) + 0.03 * max(0.0, out.get('smile', 0.0)), -1.0, 1.0)
        out['lip_corner_down'] = _clamp(out.get('lip_corner_down', 0.0) + 0.03 * max(0.0, -out.get('smile', 0.0)), 0.0, 1.0)
        out['jaw_clench'] = _clamp(out.get('jaw_clench', 0.0) + 0.04 * (1.0 - self.energy_gate), 0.0, 1.0)
        out['tongue_out'] = _clamp(out.get('tongue_out', 0.0) + 0.03 * self.energy_gate, 0.0, 1.0)
        out['tongue_tip_interdental'] = _clamp(out.get('tongue_tip_interdental', 0.0) + 0.02 * (1.0 if current_viseme in {'LNTD', 'SZCHJ'} else 0.0), 0.0, 1.0)
        out['upper_lip_raise'] = _clamp(out.get('upper_lip_raise', 0.0) + 0.03 * max(0.0, out.get('smile', 0.0)), 0.0, 1.0)
        out['lower_lip_depress'] = _clamp(out.get('lower_lip_depress', 0.0) + 0.03 * out.get('jaw_drop', 0.0), 0.0, 1.0)
        out['cheek_puff'] = _clamp(out.get('cheek_puff', 0.0) + 0.02 * out.get('lip_round', 0.0), 0.0, 1.0)
        out['cheek_suck'] = _clamp(out.get('cheek_suck', 0.0) + 0.02 * max(0.0, out.get('smile', 0.0)), 0.0, 1.0)
        out['mouth_corner_stretch'] = _clamp(out.get('mouth_corner_stretch', 0.0) + 0.02 * body, 0.0, 1.0)
        self.last_name = current_viseme
        return out



class PhonemeVisemeTimeline:
    def __init__(self, preload_s: float = 0.50, coarticulation_s: float = 0.12):
        self.preload_s = preload_s
        self.coarticulation_s = coarticulation_s
        self.frames: List[VisemeFrame] = []
        self.duration = 0.0
        self.text = ""
        self.active = False
        self.time = 0.0
        self._last_snapshot = VisemeSnapshot(); self._smoother = VisemeTransitionFilter()

    def reset(self) -> None:
        self.frames.clear(); self.duration = 0.0; self.text = ""; self.active = False; self.time = 0.0; self._last_snapshot = VisemeSnapshot(); self._smoother.reset()

    @staticmethod
    def from_phoneme_marks(text: str, duration: float, marks: Sequence[Any], *, preload_s: float = 0.50, coarticulation_s: float = 0.12) -> "PhonemeVisemeTimeline":
        tl = PhonemeVisemeTimeline(preload_s, coarticulation_s)
        tl.text = _normalize_text(text)
        tl.duration = max(0.05, float(duration))
        for idx, mark in enumerate(marks or []):
            if isinstance(mark, dict):
                phoneme = str(mark.get("phoneme", mark.get("symbol", "")))
                start = float(mark.get("start", mark.get("t0", idx * 0.08)))
                end = float(mark.get("end", mark.get("t1", start + 0.08)))
                conf = float(mark.get("confidence", 1.0))
            else:
                phoneme = str(mark[0]) if len(mark) > 0 else ""
                start = float(mark[1]) if len(mark) > 1 else idx * 0.08
                end = float(mark[2]) if len(mark) > 2 else start + 0.08
                conf = float(mark[3]) if len(mark) > 3 else 1.0
            vis = _token_viseme(phoneme)
            tl.frames.append(VisemeFrame(start=max(0.0, start), end=max(start + 1e-3, end), phoneme=phoneme, viseme=vis, confidence=_clamp(conf, 0.0, 1.0), parameters=dict(VISEME_TEMPLATES.get(vis, VISEME_TEMPLATES["neutral"]))))
        if not tl.frames:
            tl.frames = [VisemeFrame(0.0, tl.duration, "a", "A", parameters=dict(VISEME_TEMPLATES["A"]))]
        tl.active = True; tl._last_snapshot = tl.sample(0.0, speaking=True)
        return tl

    @staticmethod
    def from_text(text: str, duration: float, *, speech_rate_wps: float = 2.6, preload_s: float = 0.50, coarticulation_s: float = 0.12) -> "PhonemeVisemeTimeline":
        tl = PhonemeVisemeTimeline(preload_s, coarticulation_s)
        tl.text = _normalize_text(text)
        tl.duration = max(0.05, float(duration))
        words = [w for w in WORD_RE.findall(tl.text) if w.strip()]
        units: List[Tuple[str, bool, float, float]] = []
        for word in words:
            if not any(ch.isalpha() for ch in word):
                units.append((word, False, 0.18, 0.85)); continue
            letters = [c for c in word if c.isalpha()]
            vowels = sum(1 for c in letters if c.lower() in "aeiouáéíóúâêôàãõ")
            word_dur = max(0.10, 0.04 * max(0, len(letters) - vowels) + 0.11 * max(1, vowels) + 0.05 / max(0.6, speech_rate_wps))
            tokens = _simple_grapheme_tokens(word) or [(word[0], False)]
            total_w = sum(_token_weight(tok, _token_viseme(tok), is_vowel) for tok, is_vowel in tokens) or 1.0
            for tok, is_vowel in tokens:
                w = _token_weight(tok, _token_viseme(tok), is_vowel)
                units.append((tok, is_vowel, word_dur * (w / total_w), 1.0 if is_vowel else 0.92))
        total = sum(max(0.03, seg) for _, _, seg, _ in units) or tl.duration
        scale = tl.duration / max(total, 1e-6)
        t = 0.0
        for tok, is_vowel, seg, conf in units:
            seg = max(0.03, seg * scale)
            vis = _token_viseme(tok)
            tl.frames.append(VisemeFrame(start=t, end=t + seg, phoneme=tok, viseme=vis, weight=_token_weight(tok, vis, is_vowel), confidence=conf, blend_in=min(0.06, max(0.015, seg * 0.22)), blend_out=min(0.08, max(0.02, seg * 0.26)), parameters=dict(VISEME_TEMPLATES.get(vis, VISEME_TEMPLATES["neutral"]))))
            t += seg
        tl.frames = PhonemeVisemeTimeline._merge_redundant_frames(tl.frames) or [VisemeFrame(0.0, tl.duration, "a", "A", parameters=dict(VISEME_TEMPLATES["A"]))]
        tl.active = True; tl._last_snapshot = tl.sample(0.0, speaking=True)
        return tl

    @staticmethod
    def _merge_redundant_frames(frames: List[VisemeFrame]) -> List[VisemeFrame]:
        if not frames: return frames
        merged = [frames[0]]
        for frame in frames[1:]:
            prev = merged[-1]
            if frame.viseme == prev.viseme and frame.start <= prev.end + 1e-3:
                prev.end = max(prev.end, frame.end); prev.weight = max(prev.weight, frame.weight); prev.confidence = max(prev.confidence, frame.confidence)
            else:
                merged.append(frame)
        return merged

    def set_text(self, text: str, duration: float, *, phoneme_marks: Optional[Sequence[Any]] = None, speech_rate_wps: float = 2.6) -> None:
        fresh = PhonemeVisemeTimeline.from_phoneme_marks(text, duration, phoneme_marks, preload_s=self.preload_s, coarticulation_s=self.coarticulation_s) if phoneme_marks else PhonemeVisemeTimeline.from_text(text, duration, speech_rate_wps=speech_rate_wps, preload_s=self.preload_s, coarticulation_s=self.coarticulation_s)
        self.frames, self.duration, self.text, self.active, self.time, self._last_snapshot = fresh.frames, fresh.duration, fresh.text, fresh.active, fresh.time, fresh._last_snapshot

    def _frame_at(self, t: float) -> Tuple[Optional[VisemeFrame], Optional[VisemeFrame], float]:
        if not self.frames: return None, None, 0.0
        if t <= self.frames[0].start: return self.frames[0], self.frames[0], 0.0
        for idx, frame in enumerate(self.frames):
            if frame.start <= t <= frame.end:
                nxt = self.frames[min(idx + 1, len(self.frames) - 1)]
                span = max(1e-6, frame.end - frame.start)
                return frame, nxt, (t - frame.start) / span
        return self.frames[-1], self.frames[-1], 1.0

    def sample(self, t: float, *, speaking: bool = True, audio_energy: float = 0.0, body_scale: float = 1.0, visibility_alpha: float = 1.0, z_order_hint: float = 0.5, layer_mode: str = "respect_space") -> VisemeSnapshot:
        if not self.frames:
            self._last_snapshot = VisemeSnapshot(active=False, time=t, duration=self.duration)
            return self._last_snapshot
        lead = self.preload_s * 0.32 if speaking else 0.0
        q = _clamp(t + lead, 0.0, max(self.duration, self.frames[-1].end))
        current, nxt, local = self._frame_at(q)
        assert current is not None
        base = dict(VISEME_TEMPLATES.get(current.viseme, VISEME_TEMPLATES["neutral"]))
        target = dict(VISEME_TEMPLATES.get(nxt.viseme, base)) if nxt is not None else base
        params = _blend_dict(base, target, _smoothstep(0.0, 1.0, local) * _clamp(self.coarticulation_s / max(current.duration, self.coarticulation_s), 0.15, 0.85))
        energy = _clamp(float(audio_energy), 0.0, 1.0)
        body_scale = _clamp(float(body_scale), 0.70, 1.50)
        visibility_alpha = _clamp(float(visibility_alpha), 0.0, 1.0)
        params["mouth_open"] = _clamp((params.get("mouth_open", 0.0) * 0.68 + energy * 0.22 + current.weight * 0.06) * (0.94 + 0.10 * (body_scale - 1.0)), 0.0, 1.0)
        params["jaw_drop"] = _clamp(params.get("jaw_drop", 0.0) * 0.72 + energy * 0.26, 0.0, 1.0)
        params["smile"] = _clamp(params.get("smile", 0.0) + 0.04 * (body_scale - 1.0), -0.2, 0.5)
        params = self._smoother.apply(params, speaking=speaking, audio_energy=energy, body_scale=body_scale, visibility_alpha=visibility_alpha, current_viseme=current.viseme)
        snap = VisemeSnapshot(name=current.viseme, active=True, strength=_clamp(max(params.get("mouth_open", 0.0), current.weight * 0.18), 0.0, 1.0), time=t, duration=self.duration, mouth_open=params.get("mouth_open", 0.0), mouth_wide=params.get("mouth_wide", 0.0), lip_round=params.get("lip_round", 0.0), jaw_drop=params.get("jaw_drop", 0.0), teeth=params.get("teeth", 0.0), smile=params.get("smile", 0.0), tongue_up=params.get("tongue_up", 0.0), cheek=params.get("cheek", 0.0), lips_together=params.get("lips_together", 0.0), lip_corner_up=params.get("lip_corner_up", 0.0), lip_corner_down=params.get("lip_corner_down", 0.0), jaw_clench=params.get("jaw_clench", 0.0), tongue_out=params.get("tongue_out", 0.0), tongue_tip_interdental=params.get("tongue_tip_interdental", 0.0), upper_lip_raise=params.get("upper_lip_raise", 0.0), lower_lip_depress=params.get("lower_lip_depress", 0.0), cheek_puff=params.get("cheek_puff", 0.0), cheek_suck=params.get("cheek_suck", 0.0), mouth_corner_stretch=params.get("mouth_corner_stretch", 0.0), blink_suppression=_clamp((0.35 if speaking else 1.0) * (0.80 + 0.20 * visibility_alpha), 0.10, 1.0), confidence=current.confidence, phoneme=current.phoneme, body_scale_hint=body_scale, visibility_alpha=visibility_alpha)
        self._last_snapshot = snap
        return snap

    def advance(self, dt: float, speaking: bool, *, audio_energy: float = 0.0, body_scale: float = 1.0, visibility_alpha: float = 1.0, z_order_hint: float = 0.5, layer_mode: str = "respect_space") -> VisemeSnapshot:
        self.time = max(0.0, self.time + max(0.0, float(dt)))
        if self.active:
            self.sample(self.time, speaking=speaking, audio_energy=audio_energy, body_scale=body_scale, visibility_alpha=visibility_alpha, z_order_hint=z_order_hint, layer_mode=layer_mode)
            if self.time > self.duration + 0.28 or (not speaking and self.time > self.duration + 0.08):
                self.active = False
                self._last_snapshot = VisemeSnapshot(active=False, time=self.time, duration=self.duration)
        else:
            self._last_snapshot = VisemeSnapshot(active=False, time=self.time, duration=self.duration)
        return self._last_snapshot

    def snapshot(self) -> Dict[str, Any]:
        return self._last_snapshot.as_dict()

    def mix_mouth(self, base_open: float, *, audio_energy: float = 0.0, body_scale: float = 1.0, visibility_alpha: float = 1.0) -> float:
        vis = self._last_snapshot
        if not vis.active: return float(_clamp(base_open, 0.0, 1.0))
        scale_bias = 0.42 + 0.12 * (_clamp(body_scale, 0.70, 1.50) - 1.0)
        alpha_bias = 0.90 + 0.10 * _clamp(visibility_alpha, 0.0, 1.0)
        return float(_clamp(float(base_open) * scale_bias + vis.mouth_open * 0.34 + vis.jaw_drop * 0.08 + vis.teeth * 0.02 + vis.lips_together * 0.05 + vis.jaw_clench * 0.05 + _clamp(audio_energy, 0.0, 1.0) * 0.06, 0.0, 1.0) * alpha_bias)

    def blink_multiplier(self) -> float:
        return float(self._last_snapshot.blink_suppression)


class PhonemeVisemeSyncEngine:
    def __init__(self, preload_s: float = 0.50, coarticulation_s: float = 0.12, speech_rate_wps: float = 2.6):
        self.timeline = PhonemeVisemeTimeline(preload_s, coarticulation_s)
        self.speech_rate_wps = speech_rate_wps
        self._audio_energy = 0.0

    def reset(self) -> None:
        self.timeline.reset(); self._audio_energy = 0.0

    def start(self, text: str, duration: float, *, phoneme_marks: Optional[Sequence[Any]] = None, speech_rate_wps: Optional[float] = None) -> None:
        if speech_rate_wps is not None: self.speech_rate_wps = float(speech_rate_wps)
        self.timeline.set_text(text, duration, phoneme_marks=phoneme_marks, speech_rate_wps=self.speech_rate_wps)

    def start_from_tts_result(self, text: str, tts_result: Any, *, speech_rate_wps: Optional[float] = None) -> None:
        if isinstance(tts_result, dict):
            duration = tts_result.get("duration_s") or tts_result.get("duration") or tts_result.get("audio_duration") or 0.0
            marks = tts_result.get("phoneme_marks") or tts_result.get("viseme_marks")
        else:
            duration = getattr(tts_result, "duration_s", None) or getattr(tts_result, "duration", None) or getattr(tts_result, "audio_duration", None) or 0.0
            marks = getattr(tts_result, "phoneme_marks", None) or getattr(tts_result, "viseme_marks", None)
        self.start(text, float(duration) if duration else max(0.4, len(text) * 0.04), phoneme_marks=marks, speech_rate_wps=speech_rate_wps)

    def update(self, dt: float, speaking: bool, *, audio_energy: Optional[float] = None, body_scale: float = 1.0, visibility_alpha: float = 1.0, z_order_hint: float = 0.5, layer_mode: str = "respect_space") -> Dict[str, Any]:
        if audio_energy is not None: self._audio_energy = _clamp(float(audio_energy), 0.0, 1.0)
        return self.timeline.advance(dt, speaking, audio_energy=self._audio_energy, body_scale=body_scale, visibility_alpha=visibility_alpha, z_order_hint=z_order_hint, layer_mode=layer_mode).as_dict()

    def snapshot(self) -> Dict[str, Any]:
        return self.timeline.snapshot()

    def mix_mouth(self, base_open: float, *, audio_energy: Optional[float] = None) -> float:
        return self.timeline.mix_mouth(base_open, audio_energy=self._audio_energy if audio_energy is None else audio_energy)

    def blink_multiplier(self) -> float:
        return self.timeline.blink_multiplier()


def _patch_audio_sync_class(audio_sync_cls: Any) -> None:
    if audio_sync_cls is None or getattr(audio_sync_cls, "_phoneme_viseme_bridge_applied", False): return
    orig_reset = getattr(audio_sync_cls, "reset", None)
    orig_start_proxy = getattr(audio_sync_cls, "start_proxy", None)
    orig_update = getattr(audio_sync_cls, "update", None)
    orig_mouth_open = getattr(audio_sync_cls, "mouth_open", None)
    if not all([orig_reset, orig_start_proxy, orig_update, orig_mouth_open]): return

    def reset(self):
        r = orig_reset(self)
        eng = getattr(self, "_phoneme_viseme_engine", None)
        if eng is not None: eng.reset()
        return r

    def start_proxy(self, text: str, duration: float = 2.0, **kwargs):
        r = orig_start_proxy(self, text, duration)
        eng = getattr(self, "_phoneme_viseme_engine", None)
        if eng is None:
            eng = PhonemeVisemeSyncEngine(); self._phoneme_viseme_engine = eng
        eng.start(text, kwargs.get("audio_duration", duration), phoneme_marks=kwargs.get("phoneme_marks") or kwargs.get("marks"), speech_rate_wps=kwargs.get("speech_rate_wps"))
        return r

    def update(self, dt: float, is_speaking: bool):
        r = orig_update(self, dt, is_speaking)
        eng = getattr(self, "_phoneme_viseme_engine", None)
        if eng is not None:
            audio_energy = float(getattr(self, "energy", getattr(self, "audio_rms", 0.0)) or 0.0)
            spatial = getattr(self, "_motion_spatial", None) or {}
            eng.update(
                dt,
                bool(is_speaking or getattr(self, "speaking", False) or getattr(self, "proxy_active", False)),
                audio_energy=audio_energy,
                body_scale=float(spatial.get("body_scale", 1.0)),
                visibility_alpha=float(spatial.get("alpha", 1.0)),
                z_order_hint=float(spatial.get("z_order", spatial.get("target_z", 0.5))),
                layer_mode=str(spatial.get("layer_mode", "respect_space")),
            )
        return r

    def mouth_open(self, *args, **kwargs):
        base = orig_mouth_open(self, *args, **kwargs)
        eng = getattr(self, "_phoneme_viseme_engine", None)
        if eng is not None:
            audio_energy = float(getattr(self, "energy", getattr(self, "audio_rms", 0.0)) or 0.0)
            spatial = getattr(self, "_motion_spatial", None) or {}
            base = eng.mix_mouth(
                base,
                audio_energy=audio_energy,
                body_scale=float(spatial.get("body_scale", 1.0)),
                visibility_alpha=float(spatial.get("alpha", 1.0)),
                z_order_hint=float(spatial.get("z_order", spatial.get("target_z", 0.5))),
                layer_mode=str(spatial.get("layer_mode", "respect_space")),
            )
        return base

    def viseme_snapshot(self):
        eng = getattr(self, "_phoneme_viseme_engine", None)
        return eng.snapshot() if eng is not None else VisemeSnapshot().as_dict()

    def start_viseme(self, text: str, duration: float = 2.0, **kwargs):
        return self.start_proxy(text, duration=duration, **kwargs)

    audio_sync_cls.reset = reset
    audio_sync_cls.start_proxy = start_proxy
    audio_sync_cls.update = update
    audio_sync_cls.mouth_open = mouth_open
    audio_sync_cls.viseme_snapshot = viseme_snapshot
    audio_sync_cls.start_viseme = start_viseme
    audio_sync_cls._phoneme_viseme_bridge_applied = True


def _patch_renderer_class(renderer_cls: Any) -> None:
    if renderer_cls is None or getattr(renderer_cls, "_phoneme_viseme_renderer_applied", False): return
    orig_draw_mouth = getattr(renderer_cls, "_draw_mouth", None)
    orig_render = getattr(renderer_cls, "render", None)
    if not all([orig_draw_mouth, orig_render]): return

    def _draw_mouth(self, img, cx, cy, state, face, mouth_open, plan, audio_energy):
        vis = getattr(self, "_active_viseme", None) or {}
        space = getattr(self, "_active_space", None) or {}
        if vis:
            try:
                face = copy.copy(face)
                face.mouth_width = max(0.2, float(getattr(face, "mouth_width", 1.0)) * (0.94 + 0.22 * float(vis.get("mouth_wide", 0.0)) - 0.10 * float(vis.get("lip_round", 0.0)) + 0.08 * float(vis.get("mouth_corner_stretch", 0.0))))
                face.mouth_curve = float(getattr(face, "mouth_curve", 0.0)) + 0.16 * float(vis.get("smile", 0.0)) - 0.08 * float(vis.get("lip_round", 0.0)) + 0.08 * float(vis.get("lip_corner_up", 0.0)) - 0.10 * float(vis.get("lip_corner_down", 0.0)) + 0.10 * float(vis.get("upper_lip_raise", 0.0)) - 0.08 * float(vis.get("lower_lip_depress", 0.0))
                face.cheek_lift = max(float(getattr(face, "cheek_lift", 0.0)), 0.10 * float(vis.get("cheek", 0.0)) + 0.10 * float(vis.get("cheek_suck", 0.0)) - 0.06 * float(vis.get("cheek_puff", 0.0)))
                face.lid_tension = max(float(getattr(face, "lid_tension", 0.0)), 0.05 * float(vis.get("jaw_clench", 0.0)))
                for k in ["lips_together", "lip_corner_up", "lip_corner_down", "jaw_clench", "tongue_out", "tongue_tip_interdental", "upper_lip_raise", "lower_lip_depress", "cheek_puff", "cheek_suck", "mouth_corner_stretch"]:
                    setattr(face, k, float(vis.get(k, getattr(face, k, 0.0))))
            except Exception:
                pass
            space_alpha = float(space.get("alpha", 1.0))
            space_scale = float(space.get("scale", 1.0))
            mouth_open = max(0.0, min(1.0, float(mouth_open) * (0.26 + 0.38 * max(0.0, 1.0 - space_alpha)) + float(vis.get("mouth_open", 0.0)) * 0.46 + float(vis.get("jaw_drop", 0.0)) * 0.10 + 0.03 * max(0.0, space_scale - 1.0) - 0.10 * float(vis.get("lips_together", 0.0)) + 0.04 * float(vis.get("tongue_out", 0.0))))
        return orig_draw_mouth(self, img, cx, cy, state, face, mouth_open, plan, audio_energy)

    def render(self, *args, **kwargs):
        audio = args[5] if len(args) > 5 else kwargs.get("audio")
        motion_data = kwargs.get("motion_data")
        if motion_data is None and len(args) > 3 and isinstance(args[3], dict):
            motion_data = args[3]
        vis = None
        if audio is not None and hasattr(audio, "viseme_snapshot"):
            try:
                vis = audio.viseme_snapshot()
            except Exception:
                vis = None
        self._active_viseme = vis or VisemeSnapshot().as_dict()
        self._active_space = {}
        if motion_data:
            meta = motion_data.get("motion_meta") or {}
            self._active_space = {
                "alpha": float(meta.get("alpha", motion_data.get("space_alpha", 1.0))),
                "scale": float(meta.get("scale", motion_data.get("space_scale", 1.0))),
                "layer_mode": str(meta.get("layer_mode", motion_data.get("space_layer_mode", "respect_space"))),
            }
        try:
            return orig_render(self, *args, **kwargs)
        finally:
            self._active_viseme = None
            self._active_space = {}

    renderer_cls._draw_mouth = _draw_mouth
    renderer_cls.render = render
    renderer_cls._phoneme_viseme_renderer_applied = True


def _patch_blink_function(module_globals: Optional[Dict[str, Any]] = None) -> None:
    if not module_globals: return
    fn = module_globals.get("audio_mouth_blink")
    if fn is None or getattr(fn, "_phoneme_viseme_blink_applied", False): return
    def audio_mouth_blink(audio, emotion, perception):
        base = fn(audio, emotion, perception)
        try:
            vis = audio.viseme_snapshot() if hasattr(audio, "viseme_snapshot") else None
        except Exception:
            vis = None
        if vis:
            strength = float(vis.get("strength", 0.0))
            scale = float(vis.get("body_scale_hint", 1.0))
            alpha = float(vis.get("visibility_alpha", 1.0))
            base *= max(0.12, 1.0 - 0.72 * strength)
            base *= max(0.30, 0.95 + 0.08 * (scale - 1.0))
            base *= max(0.10, 0.85 + 0.15 * alpha)
            if vis.get("name") in {"A", "E", "I", "O", "U"}:
                base *= 0.92
        return base
    audio_mouth_blink._phoneme_viseme_blink_applied = True
    module_globals["audio_mouth_blink"] = audio_mouth_blink




def _patch_space_visibility(renderer_cls: Any) -> None:
    if renderer_cls is None or getattr(renderer_cls, '_phoneme_viseme_space_applied', False):
        return
    orig_render = getattr(renderer_cls, 'render', None)
    if orig_render is None:
        return

    def render(self, *args, **kwargs):
        motion_data = kwargs.get('motion_data')
        if motion_data is None and len(args) > 7 and isinstance(args[7], dict):
            motion_data = args[7]
        if motion_data and getattr(self, '_space_visibility_state', None) is not None:
            try:
                self._space_visibility_state['alpha'] = float(motion_data.get('space_alpha', self._space_visibility_state.get('alpha', 1.0)))
                self._space_visibility_state['scale'] = float(motion_data.get('space_scale', self._space_visibility_state.get('scale', 1.0)))
            except Exception:
                pass
        return orig_render(self, *args, **kwargs)

    renderer_cls.render = render
    renderer_cls._phoneme_viseme_space_applied = True

def install_phoneme_viseme_sync(audio_sync_cls: Any, renderer_cls: Any = None, module_globals: Optional[Dict[str, Any]] = None) -> None:
    _patch_audio_sync_class(audio_sync_cls)
    _patch_renderer_class(renderer_cls)
    _patch_space_visibility(renderer_cls)
    _patch_blink_function(module_globals)


__all__ = ["VisemeFrame", "VisemeSnapshot", "PhonemeVisemeTimeline", "PhonemeVisemeSyncEngine", "install_phoneme_viseme_sync", "VISEME_TEMPLATES", "PHONEME_TO_VISEME", "SpatialDynamicsConfig", "ScaleTransformer", "PlacementTransformer", "VisibilityTransformer"]

try:
    from avatar_orchestrator_experts import patch_all as _patch_all_experts
    _patch_all_experts(globals())
except Exception:
    pass

# Convenience export for runtime integration.
try:
    install_phoneme_viseme_sync = _patch_audio_sync_class  # type: ignore[assignment]
except Exception:
    pass
