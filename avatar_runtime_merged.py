import json
import math
import queue
import random
import threading
import time
import os
import logging
import gc
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple
import re
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageTk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("avatar")  
try:
    from voice_input_bridge import VoiceInputConfig, install_voice_input_bridge
    print("Loaded: Voice Input Bridge")
except Exception as e:
    print(f"Skipped Voice Input Bridge: {e}")

try:
    from occupancy_visibility_transformer import (
        OccupancyVisibilityController,
        SpaceOccupancyInput,
        SpaceOccupancyState,
    )
except Exception:
    OccupancyVisibilityController = None
    SpaceOccupancyInput = None
    SpaceOccupancyState = None

try:
    from animation_fluidity_transformers import (
        AnimationFluidityTransformer,
        AnimationFluidityInput,
    )
except Exception:
    AnimationFluidityTransformer = None
    AnimationFluidityInput = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from model_manager import ModelManager
except Exception:
    ModelManager = None

try:
    from tts_local import TTSManager as LocalTTSManager
    from tts_local import create_tts_manager as create_local_tts_manager
except Exception:
    
        LocalTTSManager = None
        create_local_tts_manager = None
# ----------------------------------------------------------------------
#  Utilitários matemáticos e de suavização
# ----------------------------------------------------------------------

def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    """Suavização exponencial (filtro de primeira ordem)."""
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-dt / tau)
    return lerp(current, target, k)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_noise(seed: float, t: float, freq: float = 1.0) -> float:
    """Ruído suave baseado em senoides."""
    return (math.sin(t * freq + seed) * 0.5 +
            math.sin(t * freq * 1.71 + seed * 1.37) * 0.32 +
            math.sin(t * freq * 2.63 + seed * 0.73) * 0.18)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rand_choice_weighted(items: List[Tuple[Any, float]]) -> Any:
    total = sum(max(0.0, w) for _, w in items)
    if total <= 1e-9:
        return items[0][0]
    r = random.random() * total
    acc = 0.0
    for item, w in items:
        acc += max(0.0, w)
        if r <= acc:
            return item
    return items[-1][0]


def text_summary(text: str, max_len: int = 120) -> str:
    t = " ".join((text or "").strip().split())
    return t[:max_len]


def strip_json_fences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
    return t.strip()


# ----------------------------------------------------------------------
#  Constantes (estados, emoções, gestos, alvos de olhar)
# ----------------------------------------------------------------------
STATE_NAMES = [
    "idle", "listening", "thinking", "speaking", "reacting",
    "curious", "surprised", "shy", "hide", "peek", "retreat",
    "lean_in", "observe", "rest"
]

EMOTIONS = [
    "neutral", "curious", "happy", "calm", "focused",
    "surprised", "shy", "confident", "tired", "annoyed"
]

GESTURES = [
    "none", "nod", "tilt_left", "tilt_right", "lean_in", "retreat",
    "peek", "micro_shrug", "soft_smile", "glance_away", "present"
]

GAZE_TARGETS = ["user", "mouse", "away", "internal", "edge", "text"]


# ----------------------------------------------------------------------
#  Dataclasses de estado e planos
# ----------------------------------------------------------------------
@dataclass
class PerceptionSnapshot:
    """Tudo o que o avatar percebe do ambiente e do utilizador."""
    t: float = 0.0
    dt: float = 0.016
    mouse_x: int = 0
    mouse_y: int = 0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    mouse_distance_to_center: float = 0.0
    mouse_distance_to_face: float = 0.0
    click: bool = False
    double_click: bool = False
    focus: bool = True
    silence_s: float = 0.0
    audio_rms: float = 0.0
    audio_peak: float = 0.0
    audio_attack: float = 0.0
    audio_release: float = 0.0
    audio_voiced: bool = False
    user_text: str = ""
    user_text_len: int = 0
    speaking: bool = False
    input_active: bool = False
    window_active: bool = True
    turn_index: int = 0


@dataclass
class BodyPose:
    lean_x: float = 0.0
    lean_y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    head_x: float = 0.0
    head_y: float = 0.0
    head_rot: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    gaze_target: str = "user"
    gaze_confidence: float = 0.5


@dataclass
class FacePose:
    eye_open: float = 1.0
    brow_left: float = 0.0
    brow_right: float = 0.0
    mouth_open: float = 0.0
    mouth_width: float = 1.0
    mouth_curve: float = 0.0
    mouth_round: float = 0.0
    cheek_lift: float = 0.0
    lid_tension: float = 0.0
    face_energy: float = 0.5
    micro_smile: float = 0.0
    lips_together: float = 0.0
    lip_corner_up: float = 0.0
    lip_corner_down: float = 0.0
    jaw_clench: float = 0.0
    jaw_drop: float = 0.0      # <--- ADICIONE ESTA LINHA
    tongue_out: float = 0.0
    tongue_tip_interdental: float = 0.0
    upper_lip_raise: float = 0.0
    lower_lip_depress: float = 0.0
    cheek_puff: float = 0.0
    cheek_suck: float = 0.0
    mouth_corner_stretch: float = 0.0

@dataclass
class EmotionState:
    """Estado emocional contínuo (PAD + dimensões adicionais)."""
    emotion: str = "neutral"
    intensity: float = 0.25
    valence: float = 0.0       # prazer – desprazer
    arousal: float = 0.3       # excitação – calma
    dominance: float = 0.35    # controlo – submissão
    energy: float = 0.45
    attention: float = 0.45
    curiosity: float = 0.35
    confidence: float = 0.5
    tension: float = 0.12
    social_drive: float = 0.3
    fatigue: float = 0.08
    mood_duration: float = 0.0
    last_emotion_change_t: float = 0.0
    last_action: str = "none"
    last_reaction_time: float = 0.0
    attention_target: str = "user"
    current_state: str = "idle"
    state_enter_t: float = 0.0
    state_duration: float = 0.0
    state_blend: float = 0.0
    state_velocity: float = 0.0
    repetition: float = 0.0
    # Campos para suavização de emoção
    target_energy: float = 0.45
    target_attention: float = 0.45
    target_curiosity: float = 0.35
    target_confidence: float = 0.5
    target_tension: float = 0.12
    target_social_drive: float = 0.3
    target_fatigue: float = 0.08
    target_valence: float = 0.0
    target_arousal: float = 0.3
    target_dominance: float = 0.35


@dataclass
class MemoryTurn:
    """Uma troca de fala ou ação relevante na memória."""
    t: float
    speaker: str
    text: str
    emotion: str = "neutral"
    intention: str = "none"
    state: str = "idle"
    gesture: str = "none"
    importance: float = 0.5
    # Para memória de longo prazo
    consolidated: bool = False
    last_recall: float = 0.0
    recall_count: int = 0


@dataclass
class LLMPlan:
    """Plano estruturado vindo da LLM."""
    text: str = ""
    emotion: str = "neutral"
    intensity: float = 0.35
    intention: str = "respond"
    state: str = "speaking"
    gaze_target: str = "user"
    gesture: str = "none"
    gesture_strength: float = 0.35
    motion_speed: float = 0.45
    pause_ms: int = 120
    face_energy: float = 0.5
    body_posture: Dict[str, float] = field(default_factory=dict)
    memory_update: Dict[str, Any] = field(default_factory=dict)
    attention_shift: str = "user"
    confidence: float = 0.55
    continuation: str = "neutral"
    curiosity: float = 0.5
    tension: float = 0.2
    silence_response: bool = False


# ----------------------------------------------------------------------
#  MemoryEngine – Memória em camadas com consolidação e esquecimento
# ----------------------------------------------------------------------
class MemoryEngine:
    def __init__(self, short_limit: int = 24, long_limit: int = 200, forget_rate: float = 0.02):
        self.short_term: Deque[MemoryTurn] = deque(maxlen=short_limit)   # memória imediata
        self.long_term: List[MemoryTurn] = []                             # memória consolidada
        self.long_limit = long_limit
        self.forget_rate = forget_rate                                    # taxa de esquecimento por segundo
        self.actions: Deque[str] = deque(maxlen=72)
        self.gestures: Deque[str] = deque(maxlen=72)
        self.emotions: Deque[str] = deque(maxlen=72)
        self.topics: Counter[str] = Counter()
        self.user_preferences: Dict[str, Any] = {}
        self.session_notes: List[str] = []
        self.last_topic: str = ""
        self.last_user_text: str = ""
        self.last_assistant_text: str = ""

    def remember_turn(self, speaker: str, text: str, emotion: str = "neutral",
                      intention: str = "none", state: str = "idle", gesture: str = "none",
                      importance: float = 0.5, t: float = 0.0) -> None:
        turn = MemoryTurn(
            t=t, speaker=speaker, text=text, emotion=emotion,
            intention=intention, state=state, gesture=gesture,
            importance=clamp(importance, 0.0, 1.0),
            consolidated=False, last_recall=t, recall_count=0
        )
        self.short_term.append(turn)
        self.actions.append(state)
        self.gestures.append(gesture)
        self.emotions.append(emotion)
        self.last_user_text = text if speaker == "user" else self.last_user_text
        self.last_assistant_text = text if speaker == "assistant" else self.last_assistant_text

        topic = self.extract_topic(text)
        if topic:
            self.topics[topic] += 1
            self.last_topic = topic

    def remember_plan(self, plan: LLMPlan, speaker: str = "assistant", t: float = 0.0) -> None:
        self.remember_turn(
            speaker=speaker, text=plan.text, emotion=plan.emotion,
            intention=plan.intention, state=plan.state, gesture=plan.gesture,
            importance=0.7 if plan.intention in {"explain", "empathize", "warn", "ask"} else 0.45,
            t=t
        )
        if plan.memory_update:
            self.apply_memory_update(plan.memory_update)

    def apply_memory_update(self, update: Dict[str, Any]) -> None:
        for key, value in update.items():
            if key == "topic":
                self.topics[str(value)] += 1
            elif key == "user_preference":
                if isinstance(value, dict):
                    self.user_preferences.update(value)
            elif key == "note":
                self.session_notes.append(str(value))
            else:
                self.user_preferences[key] = value

    @staticmethod
    def extract_topic(text: str) -> str:
        if not text:
            return ""
        words = [w.strip(".,!?;:()[]{}\"'\"").lower() for w in text.split()]
        words = [w for w in words if len(w) > 3]
        if not words:
            return ""
        stop = {
            "que", "para", "com", "uma", "isso", "esta", "esse", "sobre",
            "como", "mais", "muito", "não", "porque", "onde", "quando", "isso",
            "aqui", "ali", "eles", "elas", "isso"
        }
        filtered = [w for w in words if w not in stop]
        if not filtered:
            filtered = words
        return filtered[0]

    def consolidate(self, current_time: float, importance_threshold: float = 0.65) -> None:
        """Move turnos importantes da curta para a longa memória e aplica esquecimento."""
        # Consolidar novos
        for turn in list(self.short_term):
            if not turn.consolidated and turn.importance >= importance_threshold:
                turn.consolidated = True
                turn.last_recall = current_time
                self.long_term.append(turn)
        # Limitar tamanho da longa
        if len(self.long_term) > self.long_limit:
            self.long_term = self.long_term[-self.long_limit:]

        # Esquecimento: reduz importância ao longo do tempo
        for turn in self.long_term[:]:
            age = current_time - turn.last_recall
            decay = math.exp(-self.forget_rate * age)
            turn.importance *= decay
            if turn.importance < 0.05:
                self.long_term.remove(turn)

    def recall_relevant(self, current_time: float, max_items: int = 5) -> List[MemoryTurn]:
        """Recupera memórias relevantes com base na importância e recência."""
        candidates = list(self.short_term) + self.long_term
        # Pontuação = importância * (1 + 0.5 * recência)
        scored = []
        for turn in candidates:
            recency = 1.0 / (1.0 + (current_time - turn.t) * 0.2)
            score = turn.importance * (0.5 + 0.5 * recency)
            scored.append((score, turn))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Actualizar last_recall para os selecionados
        selected = [turn for _, turn in scored[:max_items]]
        for turn in selected:
            turn.last_recall = current_time
            turn.recall_count += 1
        return selected

    def recent_gesture_repeat(self, gesture: str, window: int = 6) -> float:
        recent = list(self.gestures)[-window:]
        if not recent:
            return 0.0
        return recent.count(gesture) / max(1, len(recent))

    def recent_emotion_repeat(self, emotion: str, window: int = 8) -> float:
        recent = list(self.emotions)[-window:]
        if not recent:
            return 0.0
        return recent.count(emotion) / max(1, len(recent))

    def conversation_topics(self, top_n: int = 5) -> List[Tuple[str, int]]:
        return self.topics.most_common(top_n)

    def context_digest(self) -> str:
        last_user = text_summary(self.last_user_text, 90)
        last_assistant = text_summary(self.last_assistant_text, 90)
        topics = ", ".join([f"{k}:{v}" for k, v in self.conversation_topics(3)])
        prefs = ", ".join([f"{k}={v}" for k, v in list(self.user_preferences.items())[:4]])
        return f"user={last_user} | assistant={last_assistant} | topics={topics} | prefs={prefs}"

    def brief_context(self) -> str:
        return self.context_digest()  # alias


# ----------------------------------------------------------------------
#  AudioSyncEngine – Sincronização com áudio real
# ----------------------------------------------------------------------
class AudioSyncEngine:
    def __init__(self):
        self.energy = 0.0
        self.peak = 0.0
        self.attack = 0.0
        self.release = 0.0
        self.delta = 0.0
        self.voiced = False
        self.prev_env = 0.0
        self.noise_floor = 0.018
        self.speaking = False
        self.audio_time = 0.0
        self.onset_boost = 0.0
        self.phrase_energy = 0.0
        self.last_voice_t = 0.0
        self.last_chunk_t = 0.0
        self.mouth_gate = 0.0
        self.proxy_active = False
        self.proxy_text = ""
        self.proxy_time = 0.0
        self.proxy_duration = 0.0
        self.proxy_rate = 2.4
        self.proxy_seed = random.random() * 100.0
        self.breath_gate = 0.0
        self.speech_gate = 0.0
        self.silence_gate = 1.0

    def reset(self) -> None:
        self.energy = 0.0
        self.peak = 0.0
        self.attack = 0.0
        self.release = 0.0
        self.delta = 0.0
        self.voiced = False
        self.prev_env = 0.0
        self.onset_boost = 0.0
        self.phrase_energy = 0.0
        self.mouth_gate = 0.0
        self.proxy_active = False
        self.proxy_time = 0.0
        self.proxy_duration = 0.0
        self.proxy_text = ""
        self.breath_gate = 0.0
        self.speech_gate = 0.0
        self.silence_gate = 1.0

    def feed(self, chunk: Optional[np.ndarray], sample_rate: int = 24000) -> None:
        if chunk is None:
            self.voiced = False
            self.attack = exp_smooth(self.attack, 0.0, 0.016, 0.10)
            self.release = exp_smooth(self.release, 0.0, 0.016, 0.10)
            self.prev_env = exp_smooth(self.prev_env, 0.0, 0.016, 0.12)
            self.energy = exp_smooth(self.energy, 0.0, 0.016, 0.16)
            self.mouth_gate = exp_smooth(self.mouth_gate, 0.0, 0.016, 0.14)
            return

        data = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return

        if np.max(np.abs(data)) > 1.5:
            data = data / 32768.0

        rms = float(np.sqrt(np.mean(np.square(data))))
        peak = float(np.max(np.abs(data)))
        crest = peak / max(rms, 1e-6)

        self.delta = rms - self.prev_env
        self.attack = clamp(max(0.0, self.delta) * (7.0 + crest * 0.12), 0.0, 1.0)
        self.release = clamp(max(0.0, -self.delta) * 6.0, 0.0, 1.0)
        self.prev_env = exp_smooth(self.prev_env, rms, 0.016, 0.06)
        self.energy = rms
        self.peak = peak
        self.voiced = rms > self.noise_floor or peak > self.noise_floor * 2.4
        if self.attack > 0.14:
            self.onset_boost = min(1.0, self.onset_boost + self.attack * 0.7)
        self.onset_boost = max(0.0, self.onset_boost - 0.045)
        self.phrase_energy = max(self.phrase_energy * 0.994, rms)
        self.last_voice_t = time.time()
        self.last_chunk_t = self.last_voice_t

    def start_proxy(self, text: str, duration: float = 2.0) -> None:
        self.proxy_active = True
        self.proxy_text = text or ""
        self.proxy_time = 0.0
        self.proxy_duration = max(0.6, float(duration))
        words = max(1, len([w for w in self.proxy_text.split() if w.strip()]))
        self.proxy_rate = clamp(words / max(1.0, self.proxy_duration) * 1.5, 1.15, 4.0)
        self.proxy_seed = random.random() * 100.0

    def update(self, dt: float, is_speaking: bool) -> None:
        self.speaking = is_speaking
        if self.proxy_active:
            self.proxy_time += dt
            phase = self.proxy_time * self.proxy_rate
            pulse = (0.26 + 0.30 * math.sin(phase * math.tau) +
                     0.16 * math.sin(phase * math.tau * 2.1 + self.proxy_seed) +
                     0.08 * math.sin(self.proxy_time * 4.8 + self.proxy_seed * 0.1))
            if any(p in self.proxy_text for p in [".", "!", "?", ",", ";"]):
                pulse += 0.05 * math.sin(self.proxy_time * 7.1)
            pulse = clamp(pulse, 0.02, 1.0)
            self.prev_env = exp_smooth(self.prev_env, pulse, dt, 0.08)
            self.energy = self.prev_env
            self.attack = exp_smooth(self.attack, max(0.0, math.sin(phase * math.tau)), dt, 0.09)
            self.release = exp_smooth(self.release, max(0.0, math.sin(phase * math.tau + 1.7) * 0.5), dt, 0.09)
            self.voiced = True
            self.mouth_gate = exp_smooth(self.mouth_gate, 1.0, dt, 0.08)
            self.speech_gate = exp_smooth(self.speech_gate, 1.0, dt, 0.08)
            self.breath_gate = exp_smooth(self.breath_gate, 0.55 + 0.35 * pulse, dt, 0.12)
            self.silence_gate = exp_smooth(self.silence_gate, 0.0, dt, 0.18)
            if self.proxy_time >= self.proxy_duration:
                self.proxy_active = False
        else:
            target = self.prev_env if is_speaking else 0.0
            self.energy = exp_smooth(self.energy, target, dt, 0.08 if is_speaking else 0.14)
            self.mouth_gate = exp_smooth(self.mouth_gate, 1.0 if is_speaking and self.voiced else 0.0, dt, 0.09)
            self.speech_gate = exp_smooth(self.speech_gate, 1.0 if is_speaking else 0.0, dt, 0.12)
            self.breath_gate = exp_smooth(self.breath_gate, 0.32 + 0.28 * self.energy, dt, 0.45)
            self.silence_gate = exp_smooth(self.silence_gate, 0.0 if is_speaking else 1.0, dt, 0.30)
            if not is_speaking:
                self.attack = exp_smooth(self.attack, 0.0, dt, 0.08)
                self.release = exp_smooth(self.release, 0.0, dt, 0.08)
                self.onset_boost = exp_smooth(self.onset_boost, 0.0, dt, 0.3)
                self.voiced = False

    def mouth_open(self, emotion: EmotionState, plan: Optional[LLMPlan] = None) -> float:
        base = 0.02 + self.energy * 0.72 + self.onset_boost * 0.10 + self.attack * 0.24 + self.speech_gate * 0.05
        if plan is not None:
            base *= 0.88 + 0.22 * clamp(plan.face_energy, 0.0, 1.0)
        if emotion.current_state in {"shy", "hide", "retreat"}:
            base *= 0.82
        if emotion.current_state in {"surprised", "reacting"}:
            base *= 1.12
        if emotion.emotion in {"happy", "confident"}:
            base *= 1.04
        if emotion.emotion in {"tired", "calm"}:
            base *= 0.96
        return clamp(base, 0.0, 1.0)


# ----------------------------------------------------------------------
#  AttentionEngine – Atenção saliente e olhar inteligente
# ----------------------------------------------------------------------
class AttentionEngine:
    def __init__(self):
        self.target = "user"
        self.target_point = (0.0, 0.0)
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.blink_timer = 0.0
        self.next_blink = random.uniform(2.0, 4.8)
        self.blink_amount = 0.0
        self.saccade_seed = random.random() * 100.0
        self.micro_target_seed = random.random() * 100.0
        self.last_target = "user"
        self.target_memory: Deque[str] = deque(maxlen=10)

        # Saliencia dos estímulos (decai com o tempo)
        self.stimulus_salience: Dict[str, float] = {
            "mouse": 0.0, "click": 0.0, "speech": 0.0, "silence": 0.0, "text": 0.0
        }
        self.salience_decay = 0.95   # por segundo

    def update_salience(self, perception: PerceptionSnapshot, dt: float):
        """Atualiza a saliência de cada estímulo com base na percepção."""
        # Mouse próximo
        if perception.mouse_distance_to_face < 300:
            self.stimulus_salience["mouse"] = 1.0
        else:
            self.stimulus_salience["mouse"] *= self.salience_decay ** dt

        # Clique
        if perception.click or perception.double_click:
            self.stimulus_salience["click"] = 1.0
        else:
            self.stimulus_salience["click"] *= self.salience_decay ** dt

        # Fala do utilizador
        if perception.audio_voiced or perception.user_text_len > 0:
            self.stimulus_salience["speech"] = 1.0
        else:
            self.stimulus_salience["speech"] *= self.salience_decay ** dt

        # Silêncio prolongado
        if perception.silence_s > 5.0:
            self.stimulus_salience["silence"] = min(1.0, self.stimulus_salience["silence"] + 0.2 * dt)
        else:
            self.stimulus_salience["silence"] *= self.salience_decay ** dt

        # Texto do utilizador recente
        if perception.user_text_len > 0:
            self.stimulus_salience["text"] = 1.0
        else:
            self.stimulus_salience["text"] *= self.salience_decay ** dt

    def choose_target(self, perception: PerceptionSnapshot, emotion: EmotionState,
                      memory: MemoryEngine, plan: Optional[LLMPlan]) -> str:
        self.update_salience(perception, perception.dt)

        # Se plano tem alvo definido, usa-o
        if plan and plan.gaze_target in GAZE_TARGETS:
            return plan.gaze_target

        # Decisão baseada em saliência
        sal = self.stimulus_salience
        if sal["click"] > 0.5:
            return "user"
        if sal["speech"] > 0.4:
            return "user"
        if sal["mouse"] > 0.6 and emotion.curiosity > 0.4:
            return "mouse"
        if sal["silence"] > 0.7 and emotion.curiosity > 0.5:
            return "internal"
        if sal["text"] > 0.6:
            return "user"

        # Fallback com estado interno
        if emotion.current_state in {"thinking", "rest"}:
            return "internal"
        if emotion.current_state in {"shy", "hide", "retreat"}:
            return "away"
        if emotion.attention > 0.6:
            return "user"
        return "user"  # padrão

    def target_point_for(self, target: str, perception: PerceptionSnapshot) -> Tuple[float, float]:
        w = 1024.0
        h = 1024.0
        if target == "user":
            return w / 2, h / 2 - 55
        if target == "mouse":
            return float(perception.mouse_x), float(perception.mouse_y)
        if target == "away":
            return w * 0.70, h * 0.34
        if target == "internal":
            return w * 0.42, h * 0.30
        if target == "edge":
            return w * 0.84, h * 0.66
        if target == "text":
            return w * 0.48, h * 0.58
        return w / 2, h / 2

    def update_blink(self, dt: float, emotion: EmotionState, perception: PerceptionSnapshot) -> float:
        self.blink_timer += dt
        if self.blink_amount <= 0.0 and self.blink_timer >= self.next_blink:
            self.blink_amount = 1.0
            self.blink_timer = 0.0
            base = 2.1 if emotion.current_state in {"speaking", "reacting"} else 3.0
            base += 2.0 * (1.0 - clamp(emotion.attention, 0.0, 1.0))
            if perception.silence_s > 10.0:
                base += 0.8
            self.next_blink = random.uniform(base, base + 2.8)
        if self.blink_amount > 0.0:
            self.blink_amount = max(0.0, self.blink_amount - dt * 10.5)
        return self.blink_amount

    def update_gaze(self, dt: float, perception: PerceptionSnapshot,
                    emotion: EmotionState, target_point: Tuple[float, float],
                    face_center: Tuple[float, float], state: str) -> Tuple[float, float]:
        tx, ty = target_point
        cx, cy = face_center
        dx = tx - cx
        dy = ty - cy
        if state == "thinking":
            dy -= 12.0
            dx += 24.0
        elif state == "shy":
            dx -= 26.0
            dy += 18.0
        elif state == "peek":
            dx += 34.0
        elif state == "retreat":
            dx -= 22.0
            dy += 6.0
        elif state == "lean_in":
            dx *= 1.2
            dy *= 1.1

        mouse_pull = clamp(240.0 / max(40.0, perception.mouse_distance_to_face), 0.0, 1.0)
        if perception.mouse_distance_to_face < 320:
            dx = lerp(dx, perception.mouse_x - cx, 0.22 * mouse_pull)
            dy = lerp(dy, perception.mouse_y - cy, 0.22 * mouse_pull)

        sacc = soft_noise(self.saccade_seed, perception.t, 1.5)
        micro = soft_noise(self.micro_target_seed, perception.t, 2.9)
        dx += sacc * 6.0 + micro * 2.2
        dy += sacc * 4.0 - micro * 1.6

        if perception.audio_voiced:
            dx *= 0.98
            dy *= 0.98

        self.gaze_x = exp_smooth(self.gaze_x, clamp(dx / 24.0, -1.0, 1.0), dt, 0.06)
        self.gaze_y = exp_smooth(self.gaze_y, clamp(dy / 22.0, -1.0, 1.0), dt, 0.06)
        return self.gaze_x, self.gaze_y


# ----------------------------------------------------------------------
#  EmotionEngine – Emoção contínua (PAD + inércia)
# ----------------------------------------------------------------------
class EmotionEngine:
    def __init__(self):
        self.state = EmotionState()
        self._emotion_seed = random.random() * 90.0
        self._mood_seed = random.random() * 70.0
        self._inertia = 0.2   # suavidade das mudanças emocionais

    def _emotion_from_scores(self, valence: float, arousal: float,
                             confidence: float, tension: float,
                             curiosity: float, fatigue: float) -> str:
        if fatigue > 0.72:
            return "tired"
        if tension > 0.74:
            return "annoyed"
        if arousal > 0.70 and valence < -0.12:
            return "surprised"
        if confidence > 0.68 and valence > 0.18:
            return "confident"
        if curiosity > 0.60 and arousal > 0.38:
            return "curious"
        if valence > 0.20 and arousal > 0.40:
            return "happy"
        if arousal < 0.24 and abs(valence) < 0.14:
            return "calm"
        if confidence > 0.55 and arousal > 0.42:
            return "focused"
        return "neutral"

    def _update_targets(self, perception: PerceptionSnapshot, memory: MemoryEngine,
                        plan: Optional[LLMPlan], state_name: str):
        """Calcula os valores alvo para as dimensões emocionais."""
        s = self.state

        # Influências externas
        audio_drive = clamp(perception.audio_rms * 9.0 + perception.audio_peak * 3.5, 0.0, 1.0)
        silence_drive = clamp(perception.silence_s / 12.0, 0.0, 1.0)
        mouse_drive = clamp(240.0 / max(40.0, perception.mouse_distance_to_face), 0.0, 1.0)
        repetition = 0.0
        if memory.gestures:
            recent = list(memory.gestures)[-6:]
            repetition = max((recent.count(g) / max(1, len(recent))) for g in set(recent)) if recent else 0.0

        # Energia
        target_energy = 0.28 + audio_drive * 0.45 + (0.14 if perception.speaking else 0.0)
        target_energy += 0.08 * mouse_drive - 0.10 * silence_drive
        if plan:
            target_energy *= 0.88 + 0.22 * clamp(plan.motion_speed, 0.0, 1.0)
        s.target_energy = clamp(target_energy, 0.0, 1.0)

        # Atenção
        target_attention = 0.25 + 0.38 * audio_drive + 0.24 * mouse_drive + 0.10 * (1.0 - silence_drive)
        if perception.focus:
            target_attention += 0.07
        if state_name in {"thinking", "observe", "rest"}:
            target_attention -= 0.08
        s.target_attention = clamp(target_attention, 0.0, 1.0)

        # Curiosidade
        target_curiosity = 0.22 + 0.38 * mouse_drive + 0.18 * silence_drive + 0.10 * (1.0 if perception.click else 0.0)
        target_curiosity += 0.12 * (1.0 - s.fatigue)
        if state_name in {"curious", "peek", "lean_in"}:
            target_curiosity += 0.12
        s.target_curiosity = clamp(target_curiosity, 0.0, 1.0)

        # Confiança - CORRIGIDO: usar short_term em vez de turns
        target_confidence = 0.42 + 0.16 * s.energy + 0.14 * (1.0 if perception.speaking else 0.0)
        target_confidence += 0.10 * (0.5 + 0.5 * sigmoid((len(memory.short_term) - 4) / 3.0))
        if state_name in {"shy", "hide", "retreat"}:
            target_confidence -= 0.22
        if state_name == "speaking":
            target_confidence += 0.10
        s.target_confidence = clamp(target_confidence, 0.0, 1.0)

        # Tensão
        target_tension = 0.10 + 0.16 * repetition + 0.12 * (1.0 if perception.double_click else 0.0)
        target_tension += 0.15 * clamp((240.0 - perception.mouse_distance_to_face) / 240.0, 0.0, 1.0)
        if perception.silence_s > 16.0:
            target_tension += 0.10
        if state_name in {"hide", "retreat", "shy"}:
            target_tension += 0.12
        s.target_tension = clamp(target_tension, 0.0, 1.0)

        # Social drive - CORRIGIDO: usar short_term em vez de turns
        target_social = 0.18 + 0.25 * (1.0 if perception.speaking else 0.0) + 0.12 * audio_drive + 0.06 * len(memory.short_term) / 20.0
        s.target_social_drive = clamp(target_social, 0.0, 1.0)

        # Fadiga
        target_fatigue = 0.05 + 0.05 * silence_drive + 0.03 * (1.0 - s.energy)
        if perception.silence_s > 24.0:
            target_fatigue += 0.10
        s.target_fatigue = clamp(target_fatigue, 0.0, 1.0)

        # Valência e excitação baseadas no plano ou no contexto
        if plan:
            s.target_valence = clamp((plan.intensity - 0.5) * 0.6 +
                                     (0.12 if plan.emotion in {"happy", "confident"} else 0.0) -
                                     (0.08 if plan.emotion in {"annoyed", "surprised"} else 0.0),
                                     -1.0, 1.0)
            s.target_arousal = clamp(0.28 + plan.intensity * 0.42 +
                                     (0.12 if plan.state == "speaking" else 0.0),
                                     0.0, 1.0)
        else:
            s.target_valence = clamp(0.08 * s.curiosity + 0.12 * s.social_drive - 0.10 * s.tension - 0.06 * s.fatigue,
                                     -1.0, 1.0)
            s.target_arousal = clamp(0.20 + 0.30 * s.energy + 0.12 * s.curiosity, 0.0, 1.0)

        # Dominância
        if state_name == "speaking":
            s.target_dominance = 0.65
        elif state_name in {"hide", "retreat", "shy"}:
            s.target_dominance = 0.22
        else:
            s.target_dominance = 0.45 + 0.08 * s.confidence - 0.05 * s.tension

    def update(self, perception: PerceptionSnapshot, memory: MemoryEngine,
               plan: Optional[LLMPlan], state_name: str) -> EmotionState:
        s = self.state
        dt = perception.dt

        self._update_targets(perception, memory, plan, state_name)

        # Suavização com inércia
        tau = 0.20  # constante de tempo para todas as dimensões
        s.energy = exp_smooth(s.energy, s.target_energy, dt, tau * self._inertia)
        s.attention = exp_smooth(s.attention, s.target_attention, dt, tau)
        s.curiosity = exp_smooth(s.curiosity, s.target_curiosity, dt, tau)
        s.confidence = exp_smooth(s.confidence, s.target_confidence, dt, tau)
        s.tension = exp_smooth(s.tension, s.target_tension, dt, tau)
        s.social_drive = exp_smooth(s.social_drive, s.target_social_drive, dt, tau)
        s.fatigue = exp_smooth(s.fatigue, s.target_fatigue, dt, tau * 1.5)
        s.valence = exp_smooth(s.valence, s.target_valence, dt, tau)
        s.arousal = exp_smooth(s.arousal, s.target_arousal, dt, tau)
        s.dominance = exp_smooth(s.dominance, s.target_dominance, dt, tau)

        # Intensidade global
        s.intensity = exp_smooth(s.intensity,
                                 clamp((s.energy * 0.45 + s.curiosity * 0.18 + s.arousal * 0.28 + abs(s.valence) * 0.10),
                                       0.0, 1.0),
                                 dt, tau)

        # Atualizar a label de emoção
        if plan and plan.emotion in EMOTIONS:
            if s.emotion != plan.emotion and perception.t - s.last_emotion_change_t > 0.65:
                s.emotion = plan.emotion
                s.last_emotion_change_t = perception.t
        else:
            target_emotion = self._emotion_from_scores(s.valence, s.arousal, s.confidence,
                                                       s.tension, s.curiosity, s.fatigue)
            if target_emotion != s.emotion and perception.t - s.last_emotion_change_t > 0.85:
                s.emotion = target_emotion
                s.last_emotion_change_t = perception.t

        s.mood_duration += dt
        s.current_state = state_name
        s.repetition = 0.0  # seria atualizado externamente
        s.last_action = plan.gesture if plan else s.last_action
        s.last_reaction_time = perception.t if (perception.click or perception.double_click or
                                                 perception.audio_voiced or perception.speaking) else s.last_reaction_time
        return s


# ----------------------------------------------------------------------
#  BehaviorEngine – Máquina de estados viva com transições suaves
# ----------------------------------------------------------------------
class BehaviorEngine:
    def __init__(self):
        self.current_state = "idle"
        self.state_enter_t = 0.0
        self.state_dwell = 0.0
        self.state_cooldown = 0.0
        self.last_plan: Optional[LLMPlan] = None
        self.last_micro_action_t = 0.0
        self.last_micro_action = "none"
        self.next_micro_action_t = 0.0
        self.state_history: Deque[str] = deque(maxlen=24)
        self.force_target_state: Optional[str] = None
        self.reaction_decay = 0.0
        self.peeking = False

        self.state_profiles: Dict[str, Dict[str, Any]] = {
            "idle":       {"emotion": "neutral", "speed": 0.30, "attention": 0.45, "energy": 0.35, "gaze": "user"},
            "listening":  {"emotion": "focused", "speed": 0.38, "attention": 0.78, "energy": 0.42, "gaze": "user"},
            "thinking":   {"emotion": "focused", "speed": 0.22, "attention": 0.38, "energy": 0.28, "gaze": "internal"},
            "speaking":   {"emotion": "confident","speed": 0.78, "attention": 0.88, "energy": 0.76, "gaze": "user"},
            "reacting":   {"emotion": "surprised","speed": 0.95, "attention": 0.92, "energy": 0.90, "gaze": "user"},
            "curious":    {"emotion": "curious",  "speed": 0.42, "attention": 0.82, "energy": 0.52, "gaze": "mouse"},
            "surprised":  {"emotion": "surprised","speed": 0.96, "attention": 1.00, "energy": 0.95, "gaze": "user"},
            "shy":        {"emotion": "shy",      "speed": 0.25, "attention": 0.48, "energy": 0.33, "gaze": "away"},
            "hide":       {"emotion": "shy",      "speed": 0.18, "attention": 0.28, "energy": 0.22, "gaze": "away"},
            "peek":       {"emotion": "curious",  "speed": 0.34, "attention": 0.70, "energy": 0.41, "gaze": "mouse"},
            "retreat":    {"emotion": "shy",      "speed": 0.28, "attention": 0.36, "energy": 0.24, "gaze": "away"},
            "lean_in":    {"emotion": "curious",  "speed": 0.55, "attention": 0.86, "energy": 0.66, "gaze": "user"},
            "observe":    {"emotion": "calm",     "speed": 0.22, "attention": 0.50, "energy": 0.29, "gaze": "user"},
            "rest":       {"emotion": "calm",     "speed": 0.14, "attention": 0.22, "energy": 0.16, "gaze": "away"},
        }

        self._gesture_memory: Deque[str] = deque(maxlen=16)
        self._state_seed = random.random() * 100.0
        self._idle_seed = random.random() * 100.0

    def _state_priority(self, perception: PerceptionSnapshot, emotion: EmotionState,
                        memory: MemoryEngine) -> List[Tuple[str, float]]:
        options = []

        if perception.speaking:
            options.append(("speaking", 10.0))
        if perception.click:
            options.append(("reacting", 8.0))
        if perception.double_click:
            options.append(("surprised", 8.5))
        if perception.audio_voiced and not perception.speaking:
            options.append(("listening", 7.0))
        if perception.user_text_len > 0 and not perception.speaking:
            options.append(("thinking", 6.5))
        if perception.mouse_distance_to_face < 150 and emotion.confidence < 0.55:
            options.append(("retreat", 8.5))
            options.append(("shy", 7.5))
            options.append(("hide", 6.0))
        if perception.mouse_distance_to_face < 200 and emotion.curiosity > 0.48:
            options.append(("curious", 7.0))
            options.append(("peek", 6.5))
        if perception.mouse_distance_to_face < 260 and emotion.confidence > 0.55:
            options.append(("lean_in", 6.0))
        if perception.silence_s > 8.0:
            options.append(("observe", 5.0))
        if perception.silence_s > 14.0:
            options.append(("rest", 7.0))
        if perception.silence_s > 5.0 and emotion.curiosity > 0.52:
            options.append(("thinking", 5.5))
        if perception.focus:
            options.append(("idle", 2.0))
        else:
            options.append(("observe", 4.0))
        if not options:
            options.append(("idle", 1.0))
        return options

    def _schedule_micro_action(self, t: float, state_name: str,
                               emotion: EmotionState, memory: MemoryEngine) -> str:
        if t < self.next_micro_action_t:
            return self.last_micro_action

        actions = ["none", "blink_pair", "micro_sigh", "small_nod", "glance_away",
                   "tiny_smile", "posture_reset", "breath_emphasis"]
        weights = {
            "none": 3.0,
            "blink_pair": 2.0,
            "micro_sigh": 1.0 if emotion.fatigue > 0.42 else 0.3,
            "small_nod": 1.3 if state_name in {"listening", "speaking"} else 0.5,
            "glance_away": 1.1 if state_name in {"thinking", "observe", "rest"} else 0.4,
            "tiny_smile": 1.2 if emotion.valence > 0.08 else 0.5,
            "posture_reset": 1.1 if state_name in {"idle", "observe", "rest"} else 0.6,
            "breath_emphasis": 1.5,
        }
        recent = list(self._gesture_memory)[-5:]
        for a in recent:
            if a in weights:
                weights[a] *= 0.45
        action = rand_choice_weighted([(a, weights[a]) for a in actions])
        self.last_micro_action = action
        self.last_micro_action_t = t
        self._gesture_memory.append(action)

        base_delay = {
            "none": random.uniform(2.1, 4.2),
            "blink_pair": random.uniform(3.0, 5.4),
            "micro_sigh": random.uniform(4.0, 7.5),
            "small_nod": random.uniform(5.0, 9.0),
            "glance_away": random.uniform(3.5, 7.0),
            "tiny_smile": random.uniform(4.0, 8.0),
            "posture_reset": random.uniform(4.0, 9.0),
            "breath_emphasis": random.uniform(2.0, 4.5),
        }[action]
        if state_name in {"speaking", "reacting", "surprised"}:
            base_delay *= 0.75
        if memory.recent_gesture_repeat(action) > 0.35:
            base_delay *= 1.4
        self.next_micro_action_t = t + base_delay
        return action

    def decide_state(self, perception: PerceptionSnapshot, emotion: EmotionState,
                     memory: MemoryEngine) -> str:
        if self.force_target_state:
            forced = self.force_target_state
            self.force_target_state = None
            return forced

        if self.state_dwell < 0.16:
            return self.current_state

        priority = self._state_priority(perception, emotion, memory)
        best = rand_choice_weighted(priority)

        # Regras de persistência para estados específicos
        if self.current_state in {"hide", "retreat", "shy"}:
            if perception.mouse_distance_to_face < 190 and emotion.confidence < 0.6:
                return self.current_state
            if perception.silence_s > 10.0:
                return "observe"
            if perception.audio_voiced or perception.user_text_len > 0:
                return "listening"
            if emotion.confidence > 0.62:
                return "peek" if perception.mouse_distance_to_face < 260 else "idle"

        if self.current_state == "speaking":
            if perception.speaking:
                return "speaking"
            if perception.audio_voiced and not perception.speaking:
                return "listening"

        if self.current_state == "reacting" and self.state_dwell < 0.55:
            return "reacting"

        if best == "idle" and perception.silence_s < 3.0 and emotion.attention > 0.55:
            return "observe"

        if best in {"curious", "peek"} and emotion.curiosity < 0.28 and perception.silence_s < 4.0:
            best = "idle"

        if best == self.current_state and self.state_dwell < 1.15:
            return self.current_state

        return best

    def update(self, perception: PerceptionSnapshot, emotion: EmotionState,
               memory: MemoryEngine, plan: Optional[LLMPlan]) -> str:
        t = perception.t
        if self.current_state == "idle" and self.state_enter_t == 0.0:
            self.state_enter_t = t

        candidate = self.decide_state(perception, emotion, memory)
        if plan and plan.state in STATE_NAMES:
            if candidate != plan.state:
                if perception.t - self.state_enter_t > 0.30 or plan.state in {"speaking", "reacting", "surprised"}:
                    candidate = plan.state

        if candidate != self.current_state:
            dwell = t - self.state_enter_t
            min_dwell = 0.18 if candidate in {"speaking", "reacting", "surprised"} else 0.34
            if self.current_state in {"hide", "retreat"} and candidate in {"idle", "observe"} and emotion.confidence < 0.56:
                min_dwell = 0.75
            if dwell >= min_dwell:
                self.current_state = candidate
                self.state_enter_t = t
                self.state_history.append(candidate)

        self.state_dwell = t - self.state_enter_t
        self.state_cooldown = max(0.0, 1.0 - self.state_dwell)
        self.reaction_decay = exp_smooth(self.reaction_decay, 0.0, perception.dt, 0.60)
        if perception.click or perception.double_click:
            self.reaction_decay = 1.0
        if plan:
            self.last_plan = plan
        self._schedule_micro_action(t, self.current_state, emotion, memory)
        emotion.current_state = self.current_state
        emotion.state_enter_t = self.state_enter_t
        emotion.state_duration = self.state_dwell
        return self.current_state

    def state_profile(self, state_name: Optional[str] = None) -> Dict[str, Any]:
        return self.state_profiles.get(state_name or self.current_state, self.state_profiles["idle"])


# ----------------------------------------------------------------------
#  Spatial Transformers – escala, posicionamento e visibilidade
# ----------------------------------------------------------------------

@dataclass
class SpatialDynamicsConfig:
    body_scale_min: float = 0.70
    body_scale_max: float = 1.60
    alpha_min: float = 0.05
    alpha_max: float = 1.00
    placement_range: float = 0.46
    occupancy_bias: float = 0.35
    visibility_softness: float = 0.40
    slot_hold_seconds: float = 0.85
    return_bias: float = 0.35
    front_bias: float = 0.60
    fullscreen_alpha_floor: float = 0.30
    escape_alpha_floor: float = 0.14
    z_top_threshold: float = 0.72


class ScaleTransformer:
    def __init__(self, config: Optional[SpatialDynamicsConfig] = None):
        self.config = config or SpatialDynamicsConfig()
        self.state = 1.0
        self.last_drive = 0.0

    def predict(self, emotion_intensity: float, emotion_energy: float, audio_energy: float, mouse_pressure: float, workspace_density: float, silence_pressure: float, attention: float, curiosity: float, fullscreen_pressure: float = 0.0, drag_pressure: float = 0.0, interaction_intensity: float = 0.0) -> float:
        drive = (
            0.32 * emotion_intensity + 0.20 * emotion_energy + 0.12 * audio_energy +
            0.12 * curiosity + 0.11 * attention + 0.09 * interaction_intensity -
            0.14 * workspace_density - 0.10 * silence_pressure - 0.11 * mouse_pressure -
            0.08 * drag_pressure + 0.06 * fullscreen_pressure
        )
        drive = clamp(drive, -1.5, 1.5)
        self.last_drive = exp_smooth(self.last_drive, drive, 0.016, 0.22)
        target = self.config.body_scale_min + (self.config.body_scale_max - self.config.body_scale_min) / (1.0 + math.exp(-3.1 * self.last_drive))
        if fullscreen_pressure > 0.55 and interaction_intensity > 0.15:
            target *= 1.02
        if workspace_density > 0.65:
            target *= 0.92
        if drag_pressure > 0.40:
            target *= 0.96
        self.state = exp_smooth(self.state, target, 0.016, 0.14 if target < self.state else 0.20)
        return self.state


class PlacementTransformer:
    def __init__(self, config: Optional[SpatialDynamicsConfig] = None):
        self.config = config or SpatialDynamicsConfig()
        self.last = (0.0, 0.0, 1.0)
        self.slot_hold = 0.0

    def predict(self, mouse_x: float, mouse_y: float, center_x: float, center_y: float, focus: bool, silence_pressure: float, workspace_density: float, mouse_pressure: float, fullscreen_pressure: float = 0.0, drag_pressure: float = 0.0, front_bias: float = 0.5, return_bias: float = 0.35) -> Tuple[float, float, float]:
        dx = (mouse_x - center_x) / max(1.0, center_x)
        dy = (mouse_y - center_y) / max(1.0, center_y)
        avoid = clamp(mouse_pressure + 0.35 * workspace_density + 0.18 * silence_pressure + 0.18 * drag_pressure + 0.10 * fullscreen_pressure, 0.0, 1.0)
        if not focus:
            avoid = min(1.0, avoid + 0.12)
        pos_x = clamp(-dx * (0.18 + 0.42 * avoid) + (0.10 - 0.20 * return_bias) * (1.0 - avoid), -self.config.placement_range, self.config.placement_range)
        pos_y = clamp(-dy * (0.16 + 0.36 * avoid) - 0.03 * fullscreen_pressure, -self.config.placement_range, self.config.placement_range)
        alpha = clamp(1.0 - 0.44 * avoid + 0.10 * front_bias * (1.0 - avoid), self.config.alpha_min, self.config.alpha_max)
        if fullscreen_pressure > 0.55:
            alpha = max(alpha, self.config.fullscreen_alpha_floor)
        self.last = (
            exp_smooth(self.last[0], pos_x, 0.016, 0.14),
            exp_smooth(self.last[1], pos_y, 0.016, 0.14),
            exp_smooth(self.last[2], alpha, 0.016, 0.18),
        )
        return self.last


class VisibilityTransformer:
    def __init__(self, config: Optional[SpatialDynamicsConfig] = None):
        self.config = config or SpatialDynamicsConfig()
        self.alpha = 1.0
        self.z_state = 0.5

    def predict(self, mouse_speed: float, focus: bool, silence_s: float, overlap_pressure: float, click_pressure: float, scale: float, fullscreen_pressure: float = 0.0, talking: bool = False, interaction_intensity: float = 0.0, topmost_hint: float = 0.5) -> float:
        hidden = (
            0.18 * (mouse_speed / 1200.0) + 0.28 * overlap_pressure + 0.20 * click_pressure +
            0.16 * clamp(silence_s / 18.0, 0.0, 1.0) + 0.10 * max(0.0, scale - 1.0) +
            0.18 * fullscreen_pressure + 0.08 * interaction_intensity
        )
        if not focus:
            hidden += 0.10
        alpha = clamp(1.0 - hidden, self.config.alpha_min, self.config.alpha_max)
        if fullscreen_pressure > 0.55:
            alpha = max(alpha, self.config.fullscreen_alpha_floor)
        if talking and fullscreen_pressure > 0.45:
            pulse = 0.5 + 0.5 * math.sin(silence_s * 1.7 + interaction_intensity * 4.0)
            alpha = max(alpha, min(0.85, self.config.fullscreen_alpha_floor + 0.35 * pulse))
        self.alpha = exp_smooth(self.alpha, alpha, 0.016, 0.20)
        target_z = clamp(0.35 + 0.45 * topmost_hint + 0.18 * fullscreen_pressure - 0.12 * overlap_pressure, 0.0, 1.0)
        self.z_state = exp_smooth(self.z_state, target_z, 0.016, 0.22)
        return self.alpha


# ----------------------------------------------------------------------
#  MotionEngine – Movimento contínuo (respiração, micro‑movimentos, gestos)
# ----------------------------------------------------------------------
class MotionEngine:
    def __init__(self):
        self.seed_a = random.random() * 200.0
        self.seed_b = random.random() * 200.0
        self.seed_c = random.random() * 200.0
        self.seed_d = random.random() * 200.0
        self.seed_e = random.random() * 200.0
        self.body = BodyPose()
        self.face = FacePose()
        self.sway_phase = random.random() * 40.0
        self.micro_phase = random.random() * 40.0
        self.head_phase = random.random() * 40.0
        self.breath_phase = random.random() * 40.0
        self.gesture_timer = 0.0
        self.last_gesture = "none"
        self.posture_strength = 0.0
        self.mouse_follow_strength = 0.0
        self.spatial_cfg = SpatialDynamicsConfig()
        self.scale_transformer = ScaleTransformer(self.spatial_cfg)
        self.placement_transformer = PlacementTransformer(self.spatial_cfg)
        self.visibility_transformer = VisibilityTransformer(self.spatial_cfg)
        self.animation_fluidity = AnimationFluidityTransformer() if AnimationFluidityTransformer is not None else None
        self.last_spatial = {"body_scale": 1.0, "alpha": 1.0, "placement_x": 0.0, "placement_y": 0.0, "workspace_density": 0.0, "fluidity": {}}

    def _micro_gesture(self, state_name: str, emotion: EmotionState, action: str) -> Tuple[float, float, float]:
        gx = gy = gr = 0.0
        if action == "small_nod":
            gy = 6.5 * math.sin(self.gesture_timer * 10.0)
            gr = 1.2 * math.sin(self.gesture_timer * 8.0)
        elif action == "tiny_smile":
            self.face.micro_smile += 0.08
        elif action == "glance_away":
            gx = -8.0 if emotion.current_state not in {"shy", "hide"} else -14.0
            gy = 4.0
        elif action == "posture_reset":
            gy = -3.5
            gr = -0.8
        elif action == "micro_sigh":
            gy = 5.0
        elif action == "breath_emphasis":
            gy = -2.5
        return gx, gy, gr

    def _continuous_micro_motion(self, t: float, state_name: str, emotion: EmotionState) -> Tuple[float, float, float, float, float, float]:
        """Gera movimentos micro contínuos (respiração, oscilação, etc.)"""
        # Respiração
        breath = 0.5 + 0.5 * math.sin(t * 1.02 + self.breath_phase)
        breath = clamp(breath, 0.0, 1.0)

        # Micro balanço corporal
        sway_x = soft_noise(self.seed_a, t, 0.26) * 1.35
        sway_y = soft_noise(self.seed_b, t, 0.23) * 0.95

        # Pequenos ajustes de cabeça
        head_sway_x = soft_noise(self.seed_c, t, 0.45) * 0.55
        head_sway_y = soft_noise(self.seed_d, t, 0.42) * 0.42
        head_rot = soft_noise(self.seed_e, t, 0.58) * 0.24

        # Ajuste baseado no estado
        if state_name == "rest":
            breath *= 0.6
            sway_x *= 0.4
            sway_y *= 0.4
        elif state_name in {"speaking", "reacting"}:
            breath *= 1.2
            sway_x *= 0.7
            sway_y *= 0.7

        # Influência emocional
        breath *= (0.7 + 0.6 * emotion.energy)
        sway_x *= (0.8 + 0.4 * emotion.curiosity)
        sway_y *= (0.8 + 0.4 * emotion.curiosity)

        return breath, sway_x, sway_y, head_sway_x, head_sway_y, head_rot

    def update(self, perception: PerceptionSnapshot, emotion: EmotionState,
               behavior: BehaviorEngine, attention: AttentionEngine,
               plan: Optional[LLMPlan]) -> Dict[str, Any]:
        profile = behavior.state_profile()
        state_name = behavior.current_state
        dt = perception.dt
        t = perception.t

        self.gesture_timer += dt
        self.face.face_energy = exp_smooth(self.face.face_energy,
                                           clamp(emotion.energy * 0.55 + emotion.arousal * 0.35 + emotion.intensity * 0.25, 0.0, 1.0),
                                           dt, 0.14)

        # Movimento contínuo
        breath, sway_x, sway_y, head_sway_x, head_sway_y, head_rot_cont = self._continuous_micro_motion(t, state_name, emotion)

        energy_bias = profile["energy"] * (0.65 + 0.55 * emotion.energy)
        speed_bias = profile["speed"] * (0.65 + 0.55 * emotion.arousal)
        att_bias = profile["attention"]

        if plan:
            speed_bias *= 0.82 + 0.40 * plan.motion_speed
            energy_bias *= 0.82 + 0.40 * plan.face_energy

        workspace_density = clamp(0.12 + 0.28 * (0.0 if perception.focus else 1.0) + 0.20 * (perception.mouse_speed / 1200.0) + 0.18 * clamp(perception.silence_s / 18.0, 0.0, 1.0), 0.0, 1.0)
        fullscreen_pressure = clamp(0.55 if not perception.focus else 0.15 * workspace_density, 0.0, 1.0)
        drag_pressure = clamp(perception.mouse_speed / 1500.0, 0.0, 1.0)
        interaction_intensity = clamp(0.35 * (1.0 if perception.speaking else 0.0) + 0.25 * (1.0 if perception.audio_voiced else 0.0) + 0.20 * (1.0 if perception.click or perception.double_click else 0.0) + 0.20 * emotion.intensity, 0.0, 1.0)
        mouse_pressure = clamp(1.0 - perception.mouse_distance_to_face / 320.0, 0.0, 1.0)
        silence_pressure = clamp(perception.silence_s / 18.0, 0.0, 1.0)
        click_pressure = 1.0 if perception.click or perception.double_click else 0.0
        scale_goal = self.scale_transformer.predict(emotion.intensity, emotion.energy, perception.audio_rms, mouse_pressure, workspace_density, silence_pressure, emotion.attention, emotion.curiosity, fullscreen_pressure=fullscreen_pressure, drag_pressure=drag_pressure, interaction_intensity=interaction_intensity)
        placement_x, placement_y, placement_alpha = self.placement_transformer.predict(perception.mouse_x, perception.mouse_y, 512.0, 512.0, perception.focus, silence_pressure, workspace_density, mouse_pressure, fullscreen_pressure=fullscreen_pressure, drag_pressure=drag_pressure, front_bias=0.62 if perception.silence_s > 5.0 else 0.35, return_bias=0.35)
        visibility_alpha = self.visibility_transformer.predict(perception.mouse_speed, perception.focus, perception.silence_s, workspace_density, click_pressure, scale_goal, fullscreen_pressure=fullscreen_pressure, talking=perception.speaking, interaction_intensity=interaction_intensity, topmost_hint=0.8 if perception.silence_s > 5.0 else 0.45)

        body_scale = (1.0 + 0.010 * (breath - 0.5) * energy_bias + 0.004 * soft_noise(self.seed_c, t, 1.1) + 0.010 * (scale_goal - 1.0))
        body_scale += 0.005 * emotion.energy
        body_scale *= 0.95 + 0.05 * float(placement_alpha)

        lean_x_target = sway_x * (4.0 + 10.0 * emotion.curiosity) * 0.12
        lean_y_target = sway_y * (3.0 + 7.0 * emotion.energy) * 0.12 - 3.5 * (emotion.fatigue * 0.3)
        rot_target = soft_noise(self.seed_d, t, 0.19) * 2.3 * (0.55 + emotion.intensity) + 0.2 * math.sin(t * 1.7)

        # Ajustes por estado
        if state_name == "speaking":
            lean_y_target -= 3.0
            rot_target += 0.4 * math.sin(t * 2.8)
        elif state_name == "lean_in":
            lean_y_target -= 8.0
            body_scale += 0.008
        elif state_name in {"retreat", "hide", "shy"}:
            lean_y_target += 6.0
            lean_x_target -= 4.0
            body_scale -= 0.004
        elif state_name == "curious":
            lean_x_target += 3.2
            lean_y_target -= 2.0
        elif state_name == "rest":
            body_scale -= 0.008
            lean_y_target += 2.0

        # Mouse following
        if perception.mouse_distance_to_face < 220:
            mouse_bias = clamp((220.0 - perception.mouse_distance_to_face) / 220.0, 0.0, 1.0)
            lean_x_target += (perception.mouse_x - 512.0) / 512.0 * 4.0 * mouse_bias
            lean_y_target += (perception.mouse_y - 512.0) / 512.0 * 3.0 * mouse_bias
            self.mouse_follow_strength = exp_smooth(self.mouse_follow_strength, mouse_bias, dt, 0.15)
        else:
            self.mouse_follow_strength = exp_smooth(self.mouse_follow_strength, 0.0, dt, 0.20)

        lean_x_target += float(placement_x) * 10.0
        lean_y_target += float(placement_y) * 10.0
        body_scale *= 0.82 + 0.18 * float(scale_goal)

        # Plano pode sobrescrever postura
        if plan and plan.body_posture:
            lean_x_target += float(plan.body_posture.get("lean_x", 0.0)) * 100.0
            lean_y_target += float(plan.body_posture.get("lean_y", 0.0)) * 100.0
            rot_target += float(plan.body_posture.get("rotation", 0.0))
            body_scale *= float(plan.body_posture.get("scale_x", 1.0))
            body_scale *= float(plan.body_posture.get("scale_y", 1.0)) ** 0.5

        body_scale *= 0.90 + 0.10 * float(visibility_alpha)

                # Micro gestos
        action = getattr(behavior, "last_micro_action", None) or (plan.gesture if plan else "none")

        fluidity: Dict[str, float] = {}
        if self.animation_fluidity is not None:
            fluidity = self.animation_fluidity.step(AnimationFluidityInput(
                dt=float(dt),
                state_name=state_name,
                mode_name=state_name,
                intensity=float(emotion.intensity),
                energy=float(emotion.energy),
                arousal=float(emotion.arousal),
                curiosity=float(emotion.curiosity),
                tension=float(emotion.tension),
                attention=float(emotion.attention),
                silence=float(perception.silence_s),
                speaking=(state_name == "speaking"),
                gesture_strength=0.42 if action != "none" else 0.12,
                motion_speed=float(speed_bias),
                mouse_speed=float(perception.mouse_speed),
                mouse_pressure=float(mouse_pressure),
                drag_pressure=float(drag_pressure),
                fullscreen_pressure=float(fullscreen_pressure),
                workspace_density=float(workspace_density),
                occupancy_pressure=float(clamp(0.42 * workspace_density + 0.24 * fullscreen_pressure + 0.18 * drag_pressure, 0.0, 1.0)),
                visibility_alpha=float(visibility_alpha),
                body_scale=float(body_scale),
                body_lean_x=float(self.body.lean_x),
                body_lean_y=float(self.body.lean_y),
                head_x=float(self.body.head_x),
                head_y=float(self.body.head_y),
                rotation=float(self.body.rotation),
                target_x=float(placement_x),
                target_y=float(placement_y),
                target_z=0.72 if not perception.focus else 0.48,
                target_alpha=float(visibility_alpha),
                transition_bias=0.58 if state_name in {"speaking", "reacting", "lean_in"} else 0.42,
                allow_micro_motion=(state_name not in {"rest", "hide"}),
            ))
        # Micro gestos
        action = behavior.last_micro_action
        gx, gy, gr = self._micro_gesture(state_name, emotion, action)
        if behavior.state_dwell < 0.25 and state_name in {"reacting", "surprised"}:
            gy -= 5.0
            gr += 1.0
        if behavior.state_dwell < 0.35 and state_name in {"shy", "retreat"}:
            gx -= 2.5
        if state_name == "speaking":
            gx += 1.6 * math.sin(t * 3.5)
            gy += 1.4 * math.sin(t * 2.8)

        # Movimentos contínuos de cabeça
        head_x_target = gx + head_sway_x * 4.0 + 6.0 * soft_noise(self.seed_e, t, 1.18)
        head_y_target = gy + head_sway_y * 3.5 + 5.0 * soft_noise(self.seed_d, t, 1.05) - 1.0 * emotion.fatigue
        head_rot_target = gr + head_rot_cont + 1.1 * soft_noise(self.seed_b, t, 1.4)

        # Olhar
        gaze_target = attention.target
        target_point = attention.target_point_for(gaze_target, perception)
        gaze_x, gaze_y = attention.update_gaze(dt, perception, emotion, target_point, (512.0, 512.0), state_name)
        blink = attention.update_blink(dt, emotion, perception)

        # Energia facial baseada em áudio
        if perception.audio_voiced and state_name == "speaking":
            face_energy_boost = perception.audio_rms * 0.9 + perception.audio_peak * 0.3 + perception.audio_attack * 0.2
        else:
            face_energy_boost = emotion.energy * 0.55 + emotion.intensity * 0.15

        emotion_scale = 1.0 + 0.02 * emotion.energy + 0.008 * emotion.arousal
        if state_name in {"surprised", "reacting"}:
            emotion_scale += 0.015
        if state_name in {"hide", "retreat"}:
            emotion_scale -= 0.012

        # Suavização
        self.body.lean_x = exp_smooth(self.body.lean_x, lean_x_target, dt, 0.14)
        self.body.lean_y = exp_smooth(self.body.lean_y, lean_y_target, dt, 0.14)
        self.body.rotation = exp_smooth(self.body.rotation, rot_target, dt, 0.16)
        self.body.scale_x = exp_smooth(self.body.scale_x, body_scale * emotion_scale, dt, 0.12)
        self.body.scale_y = exp_smooth(self.body.scale_y, body_scale * (1.0 + 0.012 * breath), dt, 0.12)
        self.body.head_x = exp_smooth(self.body.head_x, head_x_target, dt, 0.13)
        self.body.head_y = exp_smooth(self.body.head_y, head_y_target, dt, 0.13)
        self.body.head_rot = exp_smooth(self.body.head_rot, head_rot_target, dt, 0.13)
        self.body.gaze_x = exp_smooth(self.body.gaze_x, gaze_x, dt, 0.08)
        self.body.gaze_y = exp_smooth(self.body.gaze_y, gaze_y, dt, 0.08)
        self.body.gaze_target = gaze_target
        self.body.gaze_confidence = exp_smooth(self.body.gaze_confidence, att_bias, dt, 0.16)

        # Expressões faciais baseadas em emoção e micro‑gestos
        eye_open = 1.0
        brow_left = brow_right = 0.0
        mouth_curve = 0.0
        mouth_round = 0.0
        cheek_lift = 0.0
        lid_tension = 0.0
        mouth_width = 1.0

        emo = emotion.emotion
        if emo == "happy":
            eye_open = 1.03
            brow_left = brow_right = -0.05
            mouth_curve = 0.65
            cheek_lift = 0.22
        elif emo == "curious":
            eye_open = 1.10
            brow_left = -0.12
            brow_right = -0.08
            mouth_curve = 0.20
            mouth_round = 0.06
        elif emo == "calm":
            eye_open = 0.96
            brow_left = brow_right = 0.04
            mouth_curve = 0.16
        elif emo == "focused":
            eye_open = 0.92
            brow_left = brow_right = 0.08
            mouth_curve = 0.00
            lid_tension = 0.15
        elif emo == "surprised":
            eye_open = 1.20
            brow_left = brow_right = -0.18
            mouth_curve = 0.10
            mouth_round = 0.22
            mouth_width = 0.95
        elif emo == "shy":
            eye_open = 0.92
            brow_left = brow_right = 0.10
            mouth_curve = 0.06
            mouth_width = 0.90
        elif emo == "confident":
            eye_open = 1.00
            brow_left = brow_right = -0.02
            mouth_curve = 0.28
            cheek_lift = 0.10
        elif emo == "tired":
            eye_open = 0.84
            brow_left = brow_right = 0.08
            mouth_curve = -0.06
            lid_tension = 0.24
        elif emo == "annoyed":
            eye_open = 0.88
            brow_left = brow_right = 0.16
            mouth_curve = -0.12
            lid_tension = 0.20

        # Ajustes por estado
        if state_name == "speaking":
            mouth_curve += 0.05
            lid_tension *= 0.7
        elif state_name == "thinking":
            mouth_curve -= 0.05
            brow_left += 0.03
            brow_right += 0.03
        elif state_name == "reacting":
            eye_open *= 1.06
        elif state_name in {"hide", "retreat"}:
            eye_open *= 0.90
            mouth_curve -= 0.04
        elif state_name == "peek":
            eye_open *= 1.06
            mouth_width *= 0.96
        elif state_name == "lean_in":
            eye_open *= 1.03

        # Reações ao mouse
        if perception.mouse_distance_to_face < 180 and emotion.confidence < 0.55:
            lid_tension += 0.06
            brow_left += 0.06
            brow_right += 0.06

        if perception.click:
            mouth_curve += 0.08
            eye_open += 0.06
        if perception.double_click:
            mouth_round += 0.18
            eye_open += 0.10

        self.face.eye_open = exp_smooth(self.face.eye_open, clamp(eye_open, 0.65, 1.32), dt, 0.10)
        self.face.brow_left = exp_smooth(self.face.brow_left, clamp(brow_left, -0.5, 0.5), dt, 0.12)
        self.face.brow_right = exp_smooth(self.face.brow_right, clamp(brow_right, -0.5, 0.5), dt, 0.12)
        self.face.mouth_curve = exp_smooth(self.face.mouth_curve, clamp(mouth_curve, -0.5, 0.9), dt, 0.12)
        self.face.mouth_round = exp_smooth(self.face.mouth_round, clamp(mouth_round, 0.0, 0.5), dt, 0.12)
        self.face.cheek_lift = exp_smooth(self.face.cheek_lift, clamp(cheek_lift, 0.0, 0.5), dt, 0.12)
        self.face.lid_tension = exp_smooth(self.face.lid_tension, clamp(lid_tension, 0.0, 0.5), dt, 0.12)
        self.face.mouth_width = exp_smooth(self.face.mouth_width, clamp(mouth_width, 0.75, 1.15), dt, 0.12)
        self.face.face_energy = exp_smooth(self.face.face_energy, clamp(face_energy_boost, 0.0, 1.0), dt, 0.12)

        # Micro sorriso contínuo
        self.face.micro_smile = exp_smooth(self.face.micro_smile,
                                           0.12 * emotion.valence + 0.08 * soft_noise(self.seed_a, t, 2.1),
                                           dt, 0.18)

        # Abertura da boca (será usado pelo renderer, mas aqui mantemos)
        self.face.mouth_open = 0.0  # será definido pelo áudio no renderer

        self.last_spatial = {
            "body_scale": float(body_scale),
            "alpha": float(visibility_alpha),
            "placement_x": float(placement_x),
            "placement_y": float(placement_y),
            "workspace_density": float(workspace_density),
            "scale_goal": float(scale_goal),
            "fluidity": fluidity,
        }

        return {
            "body": asdict(self.body),
            "face": asdict(self.face),
            "gaze_target": gaze_target,
            "blink_amount": blink,
            "micro_action": action,
            "state_name": state_name,
            "motion_meta": self.last_spatial,
            "fluidity": fluidity,
        }


# ----------------------------------------------------------------------
#  LLMController – Plano estruturado com personalidade
# ----------------------------------------------------------------------
class LLMController:
    def __init__(self):
        self.model = None
        self.available = False
        self.history: Deque[Dict[str, str]] = deque(maxlen=16)
        self.persona = (
            "Você é um avatar digital vivo, com presença, intenção, continuidade psicológica, memória de sessão, "
            "atenção inteligente, coerência emocional e variação não repetitiva. "
            "Responda somente em JSON válido e completo, com os campos: text, emotion, intensity, intention, state, "
            "gaze_target, gesture, gesture_strength, motion_speed, pause_ms, face_energy, body_posture, memory_update, "
            "attention_shift, confidence, continuation, curiosity, tension, silence_response. "
            "Escolha ações com intenção. Evite repetição. Mantenha personalidade estável. "
            "Não use markdown, não explique nada, não escreva fora do JSON."
        )

    def load(self) -> None:
        if ModelManager is None:
            self.available = False
            return
        try:
            self.model = ModelManager(models_folder="models")
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def _fallback(self, user_text: str, memory: MemoryEngine,
                  emotion: EmotionState, perception: PerceptionSnapshot) -> LLMPlan:
        txt = text_summary(user_text, 160)
        lower = txt.lower()
        if any(w in lower for w in ["obrigado", "boa", "bom", "perfeito"]):
            return LLMPlan(
                text="Gosto desse rumo. Estou contigo.",
                emotion="happy",
                intensity=0.68,
                intention="empathize",
                state="speaking",
                gaze_target="user",
                gesture="soft_smile",
                gesture_strength=0.55,
                motion_speed=0.62,
                pause_ms=100,
                face_energy=0.72,
                body_posture={"lean_x": 0.02, "lean_y": -0.02, "rotation": -2.0, "scale_x": 1.01, "scale_y": 1.01},
                memory_update={"topic": memory.extract_topic(user_text), "note": txt},
                attention_shift="user",
                confidence=0.72,
                continuation="warm",
                curiosity=0.6,
                tension=0.1,
                silence_response=False,
            )
        if any(w in lower for w in ["como", "por que", "porque", "explica", "ajuda", "refatora"]):
            return LLMPlan(
                text="Vou organizar isso de forma clara e viva.",
                emotion="focused",
                intensity=0.74,
                intention="explain",
                state="speaking",
                gaze_target="user",
                gesture="present",
                gesture_strength=0.58,
                motion_speed=0.66,
                pause_ms=120,
                face_energy=0.78,
                body_posture={"lean_x": 0.03, "lean_y": -0.01, "rotation": -1.0, "scale_x": 1.02, "scale_y": 1.01},
                memory_update={"topic": memory.extract_topic(user_text), "note": txt},
                attention_shift="user",
                confidence=0.78,
                continuation="clear",
                curiosity=0.7,
                tension=0.2,
                silence_response=False,
            )
        if perception.silence_s > 8.0:
            return LLMPlan(
                text="Estou observando o espaço e ajustando o ritmo.",
                emotion="calm",
                intensity=0.42,
                intention="observe",
                state="observe",
                gaze_target="internal",
                gesture="glance_away",
                gesture_strength=0.33,
                motion_speed=0.28,
                pause_ms=180,
                face_energy=0.35,
                body_posture={"lean_x": 0.00, "lean_y": 0.00, "rotation": 0.5, "scale_x": 0.995, "scale_y": 0.995},
                memory_update={"note": "silence_observation"},
                attention_shift="internal",
                confidence=0.45,
                continuation="monitor",
                curiosity=0.4,
                tension=0.15,
                silence_response=True,
            )
        if perception.mouse_distance_to_face < 180:
            return LLMPlan(
                text="Estou te percebendo.",
                emotion="curious",
                intensity=0.72,
                intention="react",
                state="curious",
                gaze_target="mouse",
                gesture="peek",
                gesture_strength=0.66,
                motion_speed=0.58,
                pause_ms=90,
                face_energy=0.74,
                body_posture={"lean_x": 0.03, "lean_y": -0.05, "rotation": 1.2, "scale_x": 1.01, "scale_y": 1.00},
                memory_update={"note": "mouse_close"},
                attention_shift="mouse",
                confidence=0.60,
                continuation="watchful",
                curiosity=0.8,
                tension=0.25,
                silence_response=False,
            )
        return LLMPlan(
            text="Entendi. Vou responder com presença e continuidade.",
            emotion="confident",
            intensity=0.64,
            intention="respond",
            state="speaking",
            gaze_target="user",
            gesture="nod",
            gesture_strength=0.45,
            motion_speed=0.56,
            pause_ms=110,
            face_energy=0.63,
            body_posture={"lean_x": 0.02, "lean_y": -0.01, "rotation": -0.5, "scale_x": 1.01, "scale_y": 1.00},
            memory_update={"topic": memory.extract_topic(user_text), "note": txt},
            attention_shift="user",
            confidence=0.66,
            continuation="steady",
            curiosity=0.55,
            tension=0.18,
            silence_response=False,
        )

    def _parse_plan(self, payload: str, fallback: LLMPlan) -> LLMPlan:
        try:
            data = json.loads(_extract_json_payload(strip_json_fences(payload)))
            plan = LLMPlan(
                text=_limit_short_reply(_final_assistant_text(str(data.get("text", fallback.text)), fallback.text), 2),
                emotion=str(data.get("emotion", fallback.emotion)),
                intensity=clamp(float(data.get("intensity", fallback.intensity)), 0.0, 1.0),
                intention=str(data.get("intention", fallback.intention)),
                state=str(data.get("state", fallback.state)),
                gaze_target=str(data.get("gaze_target", fallback.gaze_target)),
                gesture=str(data.get("gesture", fallback.gesture)),
                gesture_strength=clamp(float(data.get("gesture_strength", fallback.gesture_strength)), 0.0, 1.0),
                motion_speed=clamp(float(data.get("motion_speed", fallback.motion_speed)), 0.0, 1.0),
                pause_ms=int(data.get("pause_ms", fallback.pause_ms)),
                face_energy=clamp(float(data.get("face_energy", fallback.face_energy)), 0.0, 1.0),
                body_posture=dict(data.get("body_posture", fallback.body_posture)),
                memory_update=dict(data.get("memory_update", fallback.memory_update)),
                attention_shift=str(data.get("attention_shift", fallback.attention_shift)),
                confidence=clamp(float(data.get("confidence", fallback.confidence)), 0.0, 1.0),
                continuation=str(data.get("continuation", fallback.continuation)),
                curiosity=clamp(float(data.get("curiosity", fallback.curiosity)), 0.0, 1.0),
                tension=clamp(float(data.get("tension", fallback.tension)), 0.0, 1.0),
                silence_response=bool(data.get("silence_response", fallback.silence_response)),
            )
            if plan.gesture not in GESTURES:
                plan.gesture = "none"
            if plan.state not in STATE_NAMES:
                plan.state = fallback.state
            if plan.gaze_target not in GAZE_TARGETS:
                plan.gaze_target = fallback.gaze_target
            plan.text = _assistant_reply_text(plan.text, fallback.text, 2) or fallback.text
            return plan
        except Exception:
            return fallback

    def build_plan(self, user_text: str, memory: MemoryEngine, emotion: EmotionState,
                   perception: PerceptionSnapshot) -> LLMPlan:
        fallback = self._fallback(user_text, memory, emotion, perception)
        if not self.available or self.model is None:
            return fallback

        try:
            context = memory.brief_context() if hasattr(memory, "brief_context") else memory.context_digest()
            prompt = (
                f"{self.persona}\n\n"
                f"Contexto atual:\n{context}\n\n"
                f"Estado emocional:\n{json.dumps(asdict(emotion), ensure_ascii=False)}\n\n"
                f"Entrada do usuário:\n{user_text}\n\n"
                f"Retorne apenas JSON."
            )
            messages = [
                {"role": "system", "content": self.persona},
                {"role": "user", "content": prompt},
            ]
            result = self.model.chat(messages, max_tokens=320, temperature=0.7)
            if isinstance(result, dict) and "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
            else:
                content = str(result)
            plan = self._parse_plan(content, fallback)
            self.history.append({"user": user_text, "assistant": plan.text})
            return plan
        except Exception:
            return fallback


# ----------------------------------------------------------------------
#  AvatarRenderer – Renderização visual com micro‑expressões
# ----------------------------------------------------------------------
class AvatarRenderer:
    def __init__(self, render_res: int = 1400, display_res: int = 900):
        self.render_res = render_res
        self.display_res = display_res
        self.cx = render_res // 2
        self.cy = render_res // 2 + 45
        self.head_radius = int(render_res * 0.22)
        self.bg_seed = random.random() * 20.0
        self.material_seed = random.random() * 20.0
        self._build_layers()

    def _build_layers(self) -> None:
        self.atmosphere = Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))
        da = ImageDraw.Draw(self.atmosphere)
        da.ellipse((self.render_res * 0.10, self.render_res * 0.06,
                    self.render_res * 0.90, self.render_res * 0.92),
                   fill=(18, 90, 145, 22))
        da.ellipse((self.render_res * 0.18, self.render_res * 0.12,
                    self.render_res * 0.82, self.render_res * 0.84),
                   fill=(12, 40, 76, 18))
        self.atmosphere = self.atmosphere.filter(ImageFilter.GaussianBlur(75))

        self.speckle = Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))
        ds = ImageDraw.Draw(self.speckle)
        for _ in range(1800):
            x = random.randint(0, self.render_res - 1)
            y = random.randint(0, self.render_res - 1)
            a = random.randint(5, 25)
            ds.point((x, y), fill=(150, 220, 255, a))
        self.speckle = self.speckle.filter(ImageFilter.GaussianBlur(1.0))

    def _base_canvas(self) -> Image.Image:
        return Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))

    def _head_gradient(self, cx: int, cy: int, radius: float, t: float,
                       emotion: EmotionState) -> Image.Image:
        img = self._base_canvas()
        draw = ImageDraw.Draw(img)
        layers = 58
        tint = 1.0 + 0.08 * emotion.energy + 0.05 * emotion.intensity
        for i in range(layers, 0, -1):
            r = radius * (i / layers) ** 0.92
            depth = 1.0 - i / layers
            cr = int(20 + 14 * depth + 6 * emotion.valence * 10)
            cg = int(88 + 78 * depth + 10 * emotion.energy * 8)
            cb = int(180 + 38 * depth + 12 * emotion.curiosity * 7)
            sh = 0.63 + 0.42 * depth
            hl = 1.0 + 0.40 * (1.0 - depth)
            wobble = 1.0 + 0.004 * soft_noise(self.material_seed, t, 1.2)
            rr = int(cr * sh * hl * tint * wobble)
            gg = int(cg * sh * hl * tint * wobble)
            bb = int(cb * sh * hl * tint * wobble)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                         fill=(clamp(rr, 0, 255), clamp(gg, 0, 255), clamp(bb, 0, 255), 255))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     outline=(140, 200, 255, 48), width=3)
        return img

    def _draw_eye(self, img: Image.Image, ex: float, ey: float, state: EmotionState,
                  pose: BodyPose, side: int, blink: float) -> Image.Image:
        d = ImageDraw.Draw(img)
        gaze_x = pose.gaze_x * (14 if side == 0 else 12)
        gaze_y = pose.gaze_y * 10.0
        eye_w = 58
        eye_h = 50 * state.arousal
        open_amt = clamp(self._eye_open_for_state(state, pose, blink), 0.25, 1.35)
        eye_h *= open_amt

        d.ellipse((ex - eye_w, ey - eye_h, ex + eye_w, ey + eye_h), fill=(240, 245, 255, 255))
        iris_cx = ex + gaze_x * 0.45
        iris_cy = ey + gaze_y * 0.40
        iris_r = 30
        for r in range(iris_r, 0, -1):
            rr = int(12 + r * 0.3)
            gg = int(70 + r * 1.8)
            bb = int(150 + r * 1.0)
            d.ellipse((iris_cx - r, iris_cy - r, iris_cx + r, iris_cy + r), fill=(rr, gg, bb, 255))
        d.ellipse((iris_cx - 10, iris_cy - 10, iris_cx + 10, iris_cy + 10), fill=(0, 0, 0, 255))
        d.ellipse((iris_cx - 24, iris_cy - 24, iris_cx - 8, iris_cy - 10), fill=(255, 255, 255, 180))
        if blink > 0.01:
            lid_h = eye_h * blink * 1.55
            d.rounded_rectangle((ex - eye_w - 6, ey - lid_h, ex + eye_w + 6, ey + lid_h),
                                radius=22, fill=(80, 112, 145, 255))
        d.arc((ex - 63, ey - 10, ex + 63, ey + 45), 195, 345, fill=(20, 48, 76, 170), width=6)
        return img

    def _eye_open_for_state(self, emotion: EmotionState, pose: BodyPose, blink: float) -> float:
        base = emotion.intensity
        if emotion.current_state in {"thinking", "rest"}:
            base *= 0.86
        if emotion.current_state in {"surprised", "reacting"}:
            base *= 1.18
        if emotion.current_state in {"shy", "hide", "retreat"}:
            base *= 0.92
        if pose.gaze_target == "mouse":
            base *= 1.03
        return clamp(1.0 - blink * 0.95, 0.18, 1.35) * clamp(0.82 + base * 0.26, 0.60, 1.18)

    def _draw_brow(self, img: Image.Image, bx: float, by: float, length: int,
                   lift: float, tilt: float, intensity: float) -> Image.Image:
        d = ImageDraw.Draw(img)
        pts = []
        for i in range(18):
            u = i / 17.0
            x = bx - length / 2 + length * u
            arch = math.sin(u * math.pi)
            y = by + lift + tilt * (u - 0.5) - arch * (10 + 6 * intensity)
            pts.append((x, y))
        d.line(pts, fill=(22, 44, 70, 225), width=10)
        return img

    def _draw_mouth(self, img: Image.Image, cx: float, cy: float, state: EmotionState,
                    face: FacePose, mouth_open: float, plan: Optional[LLMPlan],
                    audio_energy: float) -> Image.Image:
        d = ImageDraw.Draw(img)
        y = cy + 102
        width = 82 * face.mouth_width * (1.0 + 0.12 * face.mouth_corner_stretch - 0.06 * face.cheek_puff + 0.05 * face.cheek_suck)
        open_amt = mouth_open
        if plan:
            open_amt *= 0.86 + 0.22 * plan.gesture_strength
        if state.current_state in {"surprised", "reacting"}:
            open_amt = max(open_amt, 0.18 + audio_energy * 0.24)
        if state.emotion in {"happy", "confident"}:
            curve = 0.60 + face.mouth_curve + face.micro_smile * 0.4 + 0.16 * face.lip_corner_up - 0.10 * face.lip_corner_down
        elif state.emotion == "curious":
            curve = 0.18 + face.mouth_curve * 0.8 + 0.10 * face.upper_lip_raise
        elif state.emotion == "focused":
            curve = 0.03 + face.mouth_curve * 0.45 - 0.08 * face.jaw_clench
        elif state.emotion == "shy":
            curve = 0.10 + face.mouth_curve * 0.3 - 0.04 * face.lip_corner_down
        elif state.emotion == "annoyed":
            curve = -0.15 + face.mouth_curve * 0.2 - 0.12 * face.jaw_clench
        else:
            curve = 0.12 + face.mouth_curve * 0.55 + 0.08 * face.lip_corner_up - 0.06 * face.lip_corner_down

        curve += 0.04 * soft_noise(self.bg_seed, time.time(), 1.1)
        open_amt = clamp(open_amt * (1.0 - 0.32 * face.lips_together) + 0.10 * face.jaw_drop + 0.05 * face.tongue_out, 0.0, 1.0)
        mouth_h = 18 + 38 * open_amt

        d.ellipse((cx - width - 14, y - 4, cx + width + 14, y + mouth_h + 18), fill=(30, 0, 0, 34))
        if open_amt > 0.08:
            d.ellipse((cx - width * 0.7, y + 6, cx + width * 0.7, y + mouth_h + 10), fill=(42, 12, 16, 248))
        if face.tongue_out > 0.08 or face.tongue_tip_interdental > 0.08:
            tx = cx + (width * 0.04 * face.tongue_out)
            ty = y + mouth_h * 0.54
            tw = width * (0.18 + 0.18 * face.tongue_out)
            th = mouth_h * (0.22 + 0.12 * face.tongue_tip_interdental)
            d.ellipse((tx - tw, ty - th, tx + tw, ty + th), fill=(194, 120, 126, 220))
        pts = []
        for i in range(-9, 10):
            x = cx + i * 8
            u = i / 9.0
            arch = math.sin((u + 1.0) * math.pi * 0.5)
            yy = y + curve * 18 * arch - open_amt * 8 * (1 - abs(u)) + 1.5 * face.lower_lip_depress - 1.2 * face.upper_lip_raise
            pts.append((x, yy))
        d.line(pts, fill=(112, 52, 64, 255), width=8)
        d.arc((cx - width, y - 2, cx + width, y + mouth_h + 8), 10, 170, fill=(210, 120, 138, 120), width=4)
        return img

    def _body(self, img: Image.Image, pose: BodyPose, emotion: EmotionState, t: float) -> Image.Image:
        d = ImageDraw.Draw(img)
        torso_cx = self.cx + pose.lean_x * 40
        torso_cy = self.cy + 255 + pose.lean_y * 24
        torso_w = 210 * pose.scale_x
        torso_h = 245 * pose.scale_y
        glow = 18 + int(22 * emotion.energy)
        d.ellipse((torso_cx - torso_w, torso_cy - torso_h, torso_cx + torso_w, torso_cy + torso_h),
                  fill=(18, 72, 112, 184))
        d.ellipse((torso_cx - torso_w * 0.90, torso_cy - torso_h * 0.92,
                   torso_cx + torso_w * 0.90, torso_cy + torso_h * 0.92),
                  fill=(22, 110, 156, 164))
        d.ellipse((torso_cx - torso_w * 0.70, torso_cy - torso_h * 0.78,
                   torso_cx + torso_w * 0.70, torso_cy + torso_h * 0.78),
                  fill=(28, 146, 198, 90))
        d.ellipse((torso_cx - torso_w * 0.56, torso_cy - torso_h * 0.60,
                   torso_cx + torso_w * 0.56, torso_cy + torso_h * 0.60),
                  fill=(36, 170, 230, glow))
        d.ellipse((torso_cx - 108, torso_cy - 180, torso_cx + 108, torso_cy + 140),
                  fill=(16, 56, 88, 124))
        return img

    def _ambient(self, img: Image.Image, emotion: EmotionState, pose: BodyPose, t: float) -> Image.Image:
        overlay = Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        center_x = self.cx + pose.head_x * 4
        center_y = self.cy + pose.head_y * 3 - 42
        for i in range(20):
            ang = t * 0.22 + i * 0.37
            rr = self.head_radius * 1.15 + 22 * math.sin(t * 1.2 + i)
            x = center_x + math.cos(ang) * rr * 0.68
            y = center_y + math.sin(ang * 0.8) * rr * 0.20 - 100 + 12 * math.sin(t + i * 0.5)
            a = int(20 + 18 * (0.5 + 0.5 * math.sin(t * 1.4 + i)))
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(150, 220, 255, a))
        if emotion.current_state in {"surprised", "reacting"}:
            d.ellipse((center_x - 260, center_y - 240, center_x + 260, center_y + 180),
                      fill=(255, 255, 255, 10))
        overlay = overlay.filter(ImageFilter.GaussianBlur(10))
        return Image.alpha_composite(img, overlay)

    def _finish(self, img: Image.Image) -> Image.Image:
        img = Image.alpha_composite(img, self.atmosphere)
        img = Image.alpha_composite(img, self.speckle)
        img = ImageEnhance.Brightness(img).enhance(1.03)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Color(img).enhance(1.06)
        return img

    def render(self, t: float, perception: PerceptionSnapshot, emotion: EmotionState,
               pose: BodyPose, face: FacePose, audio: AudioSyncEngine,
               plan: Optional[LLMPlan], motion_data: Optional[Dict[str, Any]] = None, blink_amount: float = 0.0) -> Image.Image:
        img = self._base_canvas()
        img = self._body(img, pose, emotion, t)

        head_x = int(self.cx + pose.head_x * 12)
        head_y = int(self.cy - 15 + pose.head_y * 10)
        radius = self.head_radius * pose.scale_x * (0.98 + 0.02 * emotion.energy)

        head = self._head_gradient(head_x, head_y, radius, t, emotion)
        head = Image.alpha_composite(img, head)
        img = head

        img = self._ambient(img, emotion, pose, t)

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

        img = self._draw_brow(img, left_eye_x, eye_y - 34, 80, brow_lift, -5 if pose.gaze_target != "mouse" else -2, emotion.intensity)
        img = self._draw_brow(img, right_eye_x, eye_y - 34, 80, brow_lift_r, 5 if pose.gaze_target != "mouse" else 2, emotion.intensity)

        blink = max(blink_amount, 0.55 * audio_mouth_blink(audio, emotion, perception))
        img = self._draw_eye(img, left_eye_x, eye_y, emotion, pose, side=0, blink=blink)
        img = self._draw_eye(img, right_eye_x, eye_y, emotion, pose, side=1, blink=blink)

        mouth_open = audio.mouth_open(emotion, plan)
        if plan and plan.state == "speaking":
            mouth_open = max(mouth_open, 0.04 + audio.energy * 0.82)
        else:
            mouth_open = max(mouth_open * 0.35, 0.02 + emotion.energy * 0.04)

        img = self._draw_mouth(img, head_x, head_y, emotion, face, mouth_open, plan, audio.energy)

        if pose.gaze_target == "mouse" and perception.mouse_distance_to_face < 260:
            hint = Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))
            d = ImageDraw.Draw(hint)
            mx = perception.mouse_x
            my = perception.mouse_y
            d.ellipse((mx - 8, my - 8, mx + 8, my + 8), outline=(180, 240, 255, 36), width=2)
            hint = hint.filter(ImageFilter.GaussianBlur(3))
            img = Image.alpha_composite(img, hint)

        if motion_data and motion_data.get("space_alpha") is not None:
            try:
                alpha_boost = float(motion_data.get("space_alpha", 1.0))
                if alpha_boost < 0.995:
                    img = img.copy()
                    r, g, b, a = img.split()
                    a = a.point(lambda px: int(px * clamp(alpha_boost, 0.0, 1.0)))
                    img = Image.merge("RGBA", (r, g, b, a))
            except Exception:
                pass
        img = self._finish(img)
        mask = Image.new("L", (self.render_res, self.render_res), 0)
        dm = ImageDraw.Draw(mask)
        dm.ellipse((self.cx - radius - 26, self.cy - radius - 26,
                    self.cx + radius + 26, self.cy + radius + 26), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(14))
        visibility_alpha = 1.0
        if motion_data:
            try:
                visibility_alpha = clamp(float(motion_data.get("alpha", motion_data.get("visibility_alpha", 1.0))), 0.0, 1.0)
            except Exception:
                visibility_alpha = 1.0
        if visibility_alpha < 0.999:
            mask = ImageEnhance.Brightness(mask).enhance(visibility_alpha)
        img.putalpha(mask)
        return img


def audio_mouth_blink(audio: AudioSyncEngine, emotion: EmotionState, perception: PerceptionSnapshot) -> float:
    base = 0.0
    if perception.speaking or audio.speaking:
        base = clamp(1.0 - audio.energy * 0.7, 0.0, 1.0)
    else:
        base = 0.0
    if emotion.current_state in {"surprised", "reacting"}:
        base *= 0.32
    elif emotion.current_state in {"thinking", "rest"}:
        base *= 0.85
    return base


# ----------------------------------------------------------------------
#  AvatarApp – Integração principal com loop Tkinter
# ----------------------------------------------------------------------
class AvatarApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Avatar Vivo")
        self.root.overrideredirect(True)
        self.root.configure(bg="#ffffff")
        try:
            self.root.attributes("-transparentcolor", "#ffffff")
        except Exception:
            pass

        self.display_res = 900
        self.target_fps = max(30, min(60, int(os.getenv("AVATAR_FPS", "60"))))
        self.root.geometry(f"{self.display_res}x{self.display_res}+120+90")

        self.canvas = tk.Canvas(self.root, width=self.display_res, height=self.display_res,
                                bg="#ffffff", highlightthickness=0, bd=0)
        self.canvas.pack()

        self.renderer = AvatarRenderer(render_res=1400, display_res=self.display_res)
        self.memory = MemoryEngine()
        self.audio = AudioSyncEngine()
        self.attention = AttentionEngine()
        self.behavior = BehaviorEngine()
        self.emotion = EmotionEngine()
        self.motion = MotionEngine()
        self.space_controller = OccupancyVisibilityController(
            screen_width=float(self.root.winfo_screenwidth() or self.display_res),
            screen_height=float(self.root.winfo_screenheight() or self.display_res),
            avatar_size=float(self.display_res),
        ) if OccupancyVisibilityController is not None else None
        self.llm = LLMController()
        self.llm.load()

        self.perception = PerceptionSnapshot()
        self.plan: Optional[LLMPlan] = None
        self.is_speaking = False
        self.start_time = time.time()
        self.last_frame_t = self.start_time
        self.last_mouse_t = self.start_time
        self.last_activity_t = self.start_time
        self.last_click_t = 0.0
        self.last_input_text = ""
        self.input_queue: "queue.Queue[str]" = queue.Queue()
        self.audio_thread_stop = threading.Event()
        self.speech_cancel = threading.Event()
        self.voice_input_bridge = None
        self.photo = None
        self.drag = {"x": 0, "y": 0}
        self.blink_amount = 0.0
        self.mouse_x = self.display_res // 2
        self.mouse_y = self.display_res // 2
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.mouse_speed = 0.0
        self.focus = True
        self.silence_start_t = self.start_time
        self.turn_index = 0
        self.window_active = True
        self.last_memory_consolidate_t = self.start_time
        self.window_x = 120.0
        self.window_y = 90.0
        self.window_alpha = 1.0
        self.window_topmost = True
        self.window_layer_mode = "respect_space"

        self._setup_window()
        self._bind_events()
        self._start_console_thread()
        self._maybe_load_tts()
        self._setup_voice_bridge()

    def _setup_voice_bridge(self) -> None:
        try:
            if install_voice_input_bridge is None:
                return
            config = VoiceInputConfig(
                model_name="base",
                language="pt",
                vad_rms_threshold=0.02,
                vad_start_frames=2,
                vad_stop_frames=10,
                min_utterance_seconds=0.25,
                silence_flush_s=0.45,
                blocksize=512,
                sample_rate=16000,
                channels=1,
                device=None,
            )
            self.voice_input_bridge = install_voice_input_bridge(
                self,
                config=config,
                on_speech_start=self._on_user_speech_start,
                on_speech_end=self._on_user_speech_end,
            )
            print("Voice bridge ativo.")
        except Exception as e:
            self.voice_input_bridge = None
            print(f"Voice bridge desativado: {e}")

    def _current_window_geometry(self) -> Tuple[int, int]:
        try:
            geom = self.root.geometry()
            size_part, pos_part = geom.split('+', 1)
            _w, _h = map(int, size_part.split('x'))
            x, y = map(int, pos_part.split('+'))
            return x, y
        except Exception:
            return 0, 0

    def _apply_space_dynamics(self, motion_data: Dict[str, Any], dt: float) -> Dict[str, Any]:
        controller = getattr(self, 'space_controller', None)
        if controller is None or SpaceOccupancyInput is None:
            return motion_data

        x, y = self._current_window_geometry()
        avatar_scale = float(getattr(self.motion.body, 'scale_x', 1.0) * self.display_res)
        alpha = float(motion_data.get('space_alpha', 1.0))
        now = time.time()
        idle_seconds = max(0.0, now - max(self.last_mouse_t, self.last_activity_t))
        if not self.focus:
            idle_seconds += 2.0
        fullscreen_pressure = float(motion_data.get('space_fullscreen_pressure', 1.0 if (not self.window_active or not self.focus) else 0.0))
        interaction_intensity = float(motion_data.get('space_interaction_intensity', 0.0))
        inp = SpaceOccupancyInput(
            avatar_x=float(x),
            avatar_y=float(y),
            avatar_scale=avatar_scale,
            avatar_alpha=alpha,
            mouse_x=float(self.mouse_x or self.display_res * 0.5),
            mouse_y=float(self.mouse_y or self.display_res * 0.5),
            mouse_dx=float(self.mouse_dx),
            mouse_dy=float(self.mouse_dy),
            mouse_speed=float(self.mouse_speed),
            mouse_pressure=clamp(1.0 - self.perception.mouse_distance_to_face / 420.0, 0.0, 1.0),
            focus=bool(self.focus),
            click_pressure=1.0 if self.perception.click else 0.0,
            fullscreen_pressure=fullscreen_pressure,
            silence_pressure=clamp(self.perception.silence_s / 18.0, 0.0, 1.0),
            workspace_density=clamp(0.22 + 0.40 * (1.0 - self.focus) + 0.20 * (1.0 if self.perception.input_active else 0.0) + 0.12 * (1.0 if self.is_speaking else 0.0), 0.0, 1.0),
            occupancy_pressure=clamp(0.25 + 0.40 * (1.0 - self.focus) + 0.15 * (1.0 if self.is_speaking else 0.0) + 0.12 * clamp(self.perception.silence_s / 12.0, 0.0, 1.0), 0.0, 1.0),
            idle_seconds=idle_seconds,
            time_since_input=idle_seconds,
            screen_width=float(self.root.winfo_screenwidth() or self.display_res),
            screen_height=float(self.root.winfo_screenheight() or self.display_res),
            safe_margin=24.0,
            float_bias=clamp(self.motion.body.head_x if hasattr(self.motion.body, 'head_x') else 0.0, -1.0, 1.0),
            attention_pressure=clamp(self.perception.mouse_distance_to_center / max(1.0, self.display_res), 0.0, 1.0),
            front_bias=0.72 if idle_seconds > 5.0 else 0.35,
            mouse_hot_radius=260.0,
            cursor_locked=bool(motion_data.get('space_cursor_locked', False)),
            heatmap=motion_data.get('space_heatmap'),
            occupied_rects=motion_data.get('space_occupied_rects'),
            active_window_rect=motion_data.get('space_active_window_rect'),
            dragged_window_rect=motion_data.get('space_dragged_window_rect'),
            fullscreen_window_rect=motion_data.get('space_fullscreen_window_rect'),
            z_order_hint=float(motion_data.get('space_target_z', 0.0)),
            interaction_intensity=interaction_intensity,
            window_velocity_x=float(motion_data.get('space_window_velocity_x', self.mouse_dx)),
            window_velocity_y=float(motion_data.get('space_window_velocity_y', self.mouse_dy)),
            talking=bool(self.is_speaking or motion_data.get('space_talking', False)),
            window_count=int(motion_data.get('space_window_count', 0)),
            drag_active=bool(motion_data.get('space_drag_active', False)),
            allow_return_to_last_safe=bool(motion_data.get('space_allow_return_to_last_safe', True)),
            hold_lock_seconds=float(motion_data.get('space_hold_lock_seconds', 0.85)),
            preferred_slot_index=int(motion_data.get('space_preferred_slot_index', -1)),
            return_bias=float(motion_data.get('space_return_bias', 0.35)),
            occupied_rect_bias=float(motion_data.get('space_occupied_rect_bias', 0.60)),
            workspace_focus_bias=float(motion_data.get('space_workspace_focus_bias', 0.25)),
        )
        state = controller.step(inp, dt=dt)
        self.window_x = exp_smooth(getattr(self, 'window_x', state.target_x), state.target_x, dt, 0.18 if state.layer_mode != 'dodge_window' else 0.10)
        self.window_y = exp_smooth(getattr(self, 'window_y', state.target_y), state.target_y, dt, 0.18 if state.layer_mode != 'dodge_window' else 0.10)
        self.window_alpha = exp_smooth(getattr(self, 'window_alpha', state.target_alpha), state.target_alpha, dt, 0.14 if state.target_alpha < getattr(self, 'window_alpha', 1.0) else 0.22)
        self.window_topmost = bool(state.topmost)
        self.window_layer_mode = str(state.layer_mode)
        try:
            self.root.geometry(f"{self.display_res}x{self.display_res}+{int(round(self.window_x))}+{int(round(self.window_y))}")
        except Exception:
            pass
        try:
            self.root.attributes('-alpha', float(clamp(self.window_alpha, 0.08, 1.0)))
        except Exception:
            pass
        try:
            self.root.attributes('-topmost', bool(self.window_topmost))
        except Exception:
            pass
        motion_data = dict(motion_data)
        motion_data['space_target_x'] = state.target_x
        motion_data['space_target_y'] = state.target_y
        motion_data['space_alpha'] = state.target_alpha
        motion_data['space_scale'] = state.target_scale
        motion_data['space_reason'] = state.reason
        motion_data['space_slide_x'] = state.slide_x
        motion_data['space_slide_y'] = state.slide_y
        motion_data['space_dodge_strength'] = state.dodge_strength
        motion_data['space_visibility_score'] = state.visibility_score
        motion_data['space_overlap_score'] = state.overlap_score
        motion_data['space_anchor_x'] = state.anchor_x
        motion_data['space_anchor_y'] = state.anchor_y
        motion_data['space_layer_mode'] = state.layer_mode
        motion_data['space_topmost'] = state.topmost
        motion_data['space_target_z'] = state.target_z
        motion_data['space_current_slot_id'] = state.current_slot_id
        motion_data['space_locked_slot_id'] = state.locked_slot_id
        motion_data['space_last_safe_slot_id'] = state.last_safe_slot_id
        motion_data['space_last_safe_x'] = state.last_safe_x
        motion_data['space_last_safe_y'] = state.last_safe_y
        motion_data['space_motion_curvature'] = state.motion_curvature
        motion_data['space_fullscreen_pressure'] = fullscreen_pressure
        motion_data['space_interaction_intensity'] = interaction_intensity
        motion_data['space_window_velocity_x'] = float(motion_data.get('space_window_velocity_x', self.mouse_dx))
        motion_data['space_window_velocity_y'] = float(motion_data.get('space_window_velocity_y', self.mouse_dy))
        return motion_data

    def _setup_window(self) -> None:
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", self.window_alpha)
        except Exception:
            pass
        self.root.after(100, lambda: self.root.attributes("-topmost", False))

    def _bind_events(self) -> None:
        self.root.bind("<Motion>", self._on_motion)
        self.root.bind("<Button-1>", self._on_click)
        self.root.bind("<Double-Button-1>", self._on_double_click)
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<ButtonPress-3>", self._drag_start)
        self.root.bind("<B3-Motion>", self._drag_move)

    def _drag_start(self, event) -> None:
        self.drag["x"] = event.x_root
        self.drag["y"] = event.y_root

    def _drag_move(self, event) -> None:
        dx = event.x_root - self.drag["x"]
        dy = event.y_root - self.drag["y"]
        geom = self.root.geometry()
        size_part, pos_part = geom.split("+", 1)
        w, h = map(int, size_part.split("x"))
        x, y = map(int, pos_part.split("+"))
        self.root.geometry(f"{w}x{h}+{x + dx}+{y + dy}")
        self.drag["x"] = event.x_root
        self.drag["y"] = event.y_root

    def _on_motion(self, event) -> None:
        nx = clamp(event.x, 0, self.display_res)
        ny = clamp(event.y, 0, self.display_res)
        now = time.time()
        dt = max(1e-4, now - self.last_mouse_t)
        self.mouse_dx = nx - self.mouse_x
        self.mouse_dy = ny - self.mouse_y
        self.mouse_speed = math.hypot(self.mouse_dx, self.mouse_dy) / dt
        self.mouse_x = nx
        self.mouse_y = ny
        self.last_mouse_t = now
        self.last_activity_t = now

    def _on_click(self, _event) -> None:
        now = time.time()
        self.last_click_t = now
        self.last_activity_t = now

    def _on_double_click(self, _event) -> None:
        now = time.time()
        self.last_click_t = now
        self.last_activity_t = now

    def _on_focus_in(self, _event) -> None:
        self.focus = True

    def _on_focus_out(self, _event) -> None:
        self.focus = False

    def _start_console_thread(self) -> None:
        def loop() -> None:
            while True:
                try:
                    text = input("Você: ").strip()
                    if not text:
                        continue
                    if text.lower() in {"/quit", "/exit"}:
                        self.root.after(0, self.close)
                        break
                    self.input_queue.put(text)
                except Exception:
                    break
        threading.Thread(target=loop, daemon=True).start()

    def _maybe_load_tts(self) -> None:
        self.tts = None
        if LocalTTSManager is None:
            return
        try:
            self.tts = LocalTTSManager()
        except Exception:
            self.tts = None

    def _distance_to_face(self, x: int, y: int) -> float:
        fx = self.display_res // 2
        fy = self.display_res // 2 - 20
        return math.hypot(x - fx, y - fy)

    def _update_perception(self, t: float, dt: float, user_text: str = "") -> None:
        if self.mouse_x is None:
            self.mouse_x = self.display_res // 2
        if self.mouse_y is None:
            self.mouse_y = self.display_res // 2

        click_recent = (t - self.last_click_t) < 0.18
        double_click_recent = (t - self.last_click_t) < 0.08

        if self.audio.voiced or self.is_speaking:
            self.silence_start_t = t
        silence_s = max(0.0, t - self.silence_start_t)
        self.perception = PerceptionSnapshot(
            t=t,
            dt=dt,
            mouse_x=int(self.mouse_x),
            mouse_y=int(self.mouse_y),
            mouse_dx=float(self.mouse_dx),
            mouse_dy=float(self.mouse_dy),
            mouse_speed=float(self.mouse_speed),
            mouse_distance_to_center=math.hypot(self.mouse_x - self.display_res / 2,
                                                self.mouse_y - self.display_res / 2),
            mouse_distance_to_face=self._distance_to_face(self.mouse_x, self.mouse_y),
            click=click_recent,
            double_click=double_click_recent,
            focus=self.focus,
            silence_s=silence_s,
            audio_rms=self.audio.energy,
            audio_peak=self.audio.peak,
            audio_attack=self.audio.attack,
            audio_release=self.audio.release,
            audio_voiced=self.audio.voiced,
            user_text=user_text,
            user_text_len=len(user_text or ""),
            speaking=self.is_speaking,
            input_active=bool(user_text),
            window_active=self.window_active,
            turn_index=self.turn_index,
        )


    def _on_user_speech_start(self) -> None:
        self.last_activity_t = time.time()
        if self.is_speaking:
            self._interrupt_speaking("user_voice")

    def _on_user_speech_end(self) -> None:
        self.last_activity_t = time.time()

    def _interrupt_speaking(self, reason: str = "barge_in") -> None:
        self.speech_cancel.set()
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass
        self.is_speaking = False
        self.audio.speaking = False
        self.audio.proxy_active = False
        self.audio.reset()
        if self.plan is not None:
            self.memory.session_notes.append(f"assistant_interrupt:{reason}:{text_summary(self.plan.text, 60)}")

    def _begin_response(self, user_text: str) -> None:
        now = time.time()
        t = now - self.start_time
        self.turn_index += 1
        self.last_activity_t = now
        self.speech_cancel.clear()
        self.memory.remember_turn("user", user_text, emotion=self.emotion.state.emotion,
                                  state=self.behavior.current_state, t=t)
        self.plan = self.llm.build_plan(user_text, self.memory, self.emotion.state, self.perception)
        self.memory.remember_plan(self.plan, t=t)
        self.behavior.force_target_state = self.plan.state
        self.is_speaking = True
        self.audio.speaking = True
        self.audio.reset()

        safe_text = _assistant_reply_text(getattr(self.plan, 'text', ''), 'Entendi.', 2) or 'Entendi.'
        self.plan.text = safe_text
        duration = max(1.2, len(safe_text) * 0.05 + self.plan.pause_ms / 1000.0)
        self.audio.start_proxy(safe_text, duration)

        if self.tts is not None:
            try:
                audio_gen = self.tts.synthesize(safe_text)
            except Exception:
                audio_gen = None
            if audio_gen is not None and sd is not None:
                threading.Thread(target=self._play_audio_stream, args=(audio_gen,), daemon=True).start()

        self.root.after(int(max(duration * 2.0, duration + 2.5) * 1000), self._stop_speaking)

    def _play_audio_stream(self, audio_gen) -> None:
        try:
            for chunk in audio_gen:
                if self.speech_cancel.is_set():
                    break
                if chunk is None:
                    continue
                self.audio.feed(chunk)
                if sd is not None:
                    try:
                        sd.play(chunk, samplerate=24000)
                        sd.wait()
                    except Exception:
                        break
                if self.speech_cancel.is_set():
                    break
        except Exception:
            pass

    def _stop_speaking(self) -> None:
        self.speech_cancel.set()
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass
        self.is_speaking = False
        self.audio.speaking = False
        self.audio.proxy_active = False
        self.audio.reset()
        if self.plan is not None:
            self.memory.session_notes.append(f"assistant_spoke:{text_summary(self.plan.text, 90)}")

    def _update_logic(self, user_text: str = "") -> Dict[str, Any]:
        now = time.time()
        dt = max(1e-4, now - self.last_frame_t)
        self.last_frame_t = now

        self._update_perception(now - self.start_time, dt, user_text=user_text)

        if now - self.last_memory_consolidate_t > 5.0:
            self.memory.consolidate(now - self.start_time)
            self.last_memory_consolidate_t = now

        active_plan = self.plan if self.is_speaking else None
        self.attention.target = self.attention.choose_target(self.perception, self.emotion.state,
                                                             self.memory, active_plan)
        self.attention.target_point = self.attention.target_point_for(self.attention.target, self.perception)

        state_name = self.behavior.update(self.perception, self.emotion.state, self.memory, active_plan)
        self.emotion.update(self.perception, self.memory, active_plan, state_name)
        self.audio.update(dt, self.is_speaking)
        motion_data = self.motion.update(self.perception, self.emotion.state, self.behavior,
                                         self.attention, active_plan)
        self.blink_amount = float(motion_data.get("blink_amount", 0.0))

        self.attention.update_blink(dt, self.emotion.state, self.perception)
        self.motion.body.gaze_target = self.attention.target

        motion_data = self._apply_space_dynamics(motion_data, dt)
        return motion_data

    def _draw(self) -> None:
        t = time.time() - self.start_time
        pending_user_text = ""
        if not self.input_queue.empty():
            try:
                txt = self.input_queue.get_nowait()
                if txt:
                    self.last_input_text = txt
                    pending_user_text = txt
                    self._begin_response(txt)
            except Exception:
                pass

        motion_data = self._update_logic(user_text=pending_user_text)
        frame = self.renderer.render(t, self.perception, self.emotion.state,
                                     self.motion.body, self.motion.face, self.audio,
                                     self.plan if self.is_speaking else None,
                                     motion_data=motion_data,
                                     blink_amount=self.blink_amount)
        display = frame.resize((self.display_res, self.display_res), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(self.display_res // 2, self.display_res // 2,
                                 anchor="center", image=self.photo)

        self.root.after(int(1000 / max(30, min(60, getattr(self, "target_fps", 60)))), self._draw)

    def run(self) -> None:
        self._draw()
        self.root.mainloop()

    def close(self) -> None:
        try:
            self.audio_thread_stop.set()
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    app = AvatarApp()
    app.run()



# ==============================================================================
class StreamingVoiceEvent:
    def __init__(self, kind: str, text: str = '', payload: Optional[Dict[str, Any]] = None):
        self.kind = kind
        self.text = text
        self.payload = payload or {}


class StreamingConversationHub:
    """Compatível com o arranque streaming do AvatarAppV2."""

    def __init__(self, app: Any, model: Any = None, tts: Any = None, *, max_queue: int = 8):
        self.app = app
        self.model = model
        self.tts = tts
        self.user_text_in: queue.Queue[str] = queue.Queue(maxsize=max_queue)
        self.stop_event = threading.Event()
        self.current_cancel = threading.Event()
        self.output_events: queue.Queue[StreamingVoiceEvent] = queue.Queue(maxsize=max_queue)
        self._started = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.current_cancel.set()
        try:
            self.user_text_in.put_nowait('__STOP__')
        except Exception:
            pass

    def submit(self, text: str) -> None:
        text = ' '.join((text or '').strip().split())
        if not text:
            return
        try:
            self.user_text_in.put_nowait(text)
        except queue.Full:
            try:
                _ = self.user_text_in.get_nowait()
                self.user_text_in.put_nowait(text)
            except Exception:
                pass

    def cancel_current(self) -> None:
        self.current_cancel.set()
        try:
            if hasattr(self.app, 'speech_cancel'):
                self.app.speech_cancel.set()
        except Exception:
            pass
        try:
            if hasattr(self.app, 'audio'):
                self.app.audio.speaking = False
                self.app.audio.reset()
        except Exception:
            pass
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass

    def stream_events(self) -> Iterable[str]:
        while not self.stop_event.is_set():
            try:
                item = self.user_text_in.get(timeout=0.05)
            except queue.Empty:
                continue
            if item == '__STOP__':
                break
            yield item

    def _model_stream(self, prompt: str) -> Iterable[str]:
        if self.model is None:
            return
        try:
            if hasattr(self.model, 'stream_generate'):
                for token in self.model.stream_generate(prompt, max_tokens=48, temperature=0.45):
                    token = _stream_chunk_sanitize(token)
                    if token:
                        yield token
                return
            if hasattr(self.model, 'generate_stream'):
                for token in self.model.generate_stream(prompt, max_tokens=48, temperature=0.45):
                    token = _stream_chunk_sanitize(token)
                    if token:
                        yield token
                return
            if hasattr(self.model, 'stream'):
                for token in self.model.stream(prompt, max_tokens=48, temperature=0.45):
                    token = _stream_chunk_sanitize(token)
                    if token:
                        yield token
                return
        except Exception:
            logger.exception('Falha no stream do LLM, usando fallback')
        finally:
            try:
                gc.collect()
            except Exception:
                pass

    def _run_turn(self, user_text: str) -> None:
        app = self.app
        try:
            if app is not None and hasattr(app, '_begin_response'):
                app._begin_response(user_text)
            elif app is not None and hasattr(app, 'input_queue'):
                app.input_queue.put_nowait(user_text)
        except Exception:
            logger.exception('Falha ao processar turno streaming')
        finally:
            try:
                gc.collect()
            except Exception:
                pass

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.user_text_in.get(timeout=0.05)
            except queue.Empty:
                continue
            if item == '__STOP__':
                break
            if self.current_cancel.is_set():
                self.current_cancel.clear()
            self._run_turn(item)
# Compatibility patches for streaming/full-duplex runtime
# ==============================================================================

def _compat_hub_start(self) -> None:
    self._started = True


def _compat_hub_stop(self) -> None:
    try:
        self.stop_event.set()
    except Exception:
        pass
    try:
        self.current_cancel.set()
    except Exception:
        pass
    try:
        self.user_text_in.put_nowait('__STOP__')
    except Exception:
        pass


def _compat_hub_submit(self, text: str) -> None:
    text = ' '.join((text or '').strip().split())
    if not text:
        return
    try:
        self.user_text_in.put_nowait(text)
    except queue.Full:
        try:
            _ = self.user_text_in.get_nowait()
            self.user_text_in.put_nowait(text)
        except Exception:
            pass


def _compat_hub_cancel_current(self) -> None:
    try:
        self.current_cancel.set()
    except Exception:
        pass
    try:
        app = getattr(self, 'app', None)
        if app is not None and hasattr(app, 'speech_cancel'):
            app.speech_cancel.set()
    except Exception:
        pass
    try:
        app = getattr(self, 'app', None)
        if app is not None and hasattr(app, 'audio'):
            app.audio.speaking = False
            app.audio.reset()
    except Exception:
        pass
    try:
        if sd is not None:
            sd.stop()
    except Exception:
        pass


def _compat_hub_stream_events(self) -> Iterator[str]:
    while not getattr(self, 'stop_event').is_set():
        try:
            item = self.user_text_in.get(timeout=0.05)
        except queue.Empty:
            continue
        if item == '__STOP__':
            break
        yield item


def _compat_on_user_speech_start(self) -> None:
    try:
        self.last_activity_t = time.time()
    except Exception:
        pass
    try:
        if getattr(self, 'is_speaking', False):
            self._interrupt_speaking('user_voice')
    except Exception:
        pass
    try:
        if getattr(self, 'streaming_hub', None) is not None:
            self.streaming_hub.cancel_current()
    except Exception:
        pass


def _compat_on_user_speech_end(self) -> None:
    try:
        self.last_activity_t = time.time()
    except Exception:
        pass


def _compat_on_user_barge_in(self) -> None:
    try:
        if getattr(self, 'streaming_hub', None) is not None:
            self.streaming_hub.cancel_current()
    except Exception:
        pass
    try:
        if hasattr(self, 'audio'):
            self.audio.speaking = False
            self.audio.reset()
    except Exception:
        pass

try:
    StreamingConversationHub.start = _compat_hub_start  # type: ignore[assignment]
    StreamingConversationHub.stop = _compat_hub_stop  # type: ignore[assignment]
    StreamingConversationHub.submit = _compat_hub_submit  # type: ignore[assignment]
    StreamingConversationHub.cancel_current = _compat_hub_cancel_current  # type: ignore[assignment]
    StreamingConversationHub.stream_events = _compat_hub_stream_events  # type: ignore[assignment]
    AvatarAppV2._on_user_speech_start = _compat_on_user_speech_start  # type: ignore[assignment]
    AvatarAppV2._on_user_speech_end = _compat_on_user_speech_end  # type: ignore[assignment]
    AvatarAppV2._on_user_barge_in = _compat_on_user_barge_in  # type: ignore[assignment]
except Exception:
    pass


# ==============================================================================
#  TTS / viseme runtime patch (pre-main, preserves existing API)
# ==============================================================================

def _tts_runtime_split_sentences(text: str) -> list[str]:
    try:
        return [s for s in re.split(r"(?<=[.!?\n])\s+", " ".join((text or "").strip().split())) if s.strip()]
    except Exception:
        t = " ".join((text or "").strip().split())
        return [t] if t else []


def _tts_runtime_safe_clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


# -----------------------------
# AudioSyncEngine viseme patch
# -----------------------------
_audio_sync_init_orig = AudioSyncEngine.__init__
_audio_sync_reset_orig = AudioSyncEngine.reset
_audio_sync_update_orig = AudioSyncEngine.update
_audio_sync_mouth_open_orig = AudioSyncEngine.mouth_open


def _audio_sync_init_viseme(self) -> None:
    self.viseme_label = "silence"
    self.viseme_open = 0.0
    self.viseme_round = 0.0
    self.viseme_spread = 0.0
    self.viseme_close = 1.0
    self.viseme_jaw = 0.0
    self.viseme_energy = 0.0
    self.viseme_active = False
    self.viseme_time = 0.0
    self.viseme_decay = 0.0
    self.viseme_timestamp = 0.0


def _audio_sync_init(self) -> None:
    _audio_sync_init_orig(self)
    _audio_sync_init_viseme(self)


def _audio_sync_register_viseme_listener(self, listener):
    self._viseme_listener = listener


def _audio_sync_feed_viseme(self, frame) -> None:
    try:
        if frame is None:
            self.viseme_active = False
            self.viseme_close = exp_smooth(self.viseme_close, 1.0, 0.016, 0.10)
            self.viseme_open = exp_smooth(self.viseme_open, 0.0, 0.016, 0.10)
            self.viseme_round = exp_smooth(self.viseme_round, 0.0, 0.016, 0.10)
            self.viseme_spread = exp_smooth(self.viseme_spread, 0.0, 0.016, 0.10)
            self.viseme_jaw = exp_smooth(self.viseme_jaw, 0.0, 0.016, 0.10)
            return

        if isinstance(frame, dict):
            label = str(frame.get("label", frame.get("viseme", "silence")))
            openness = float(frame.get("openness", frame.get("mouth_open", 0.0)))
            rounding = float(frame.get("rounding", frame.get("round", 0.0)))
            closure = float(frame.get("closure", frame.get("mouth_close", 1.0)))
            spread = float(frame.get("lip_spread", frame.get("spread", 0.0)))
            jaw = float(frame.get("jaw", frame.get("jaw_drop", openness)))
            energy = float(frame.get("energy", frame.get("audio_energy", 0.0)))
            timestamp = float(frame.get("timestamp", time.time()))
        else:
            label = str(getattr(frame, "label", getattr(frame, "viseme", "silence")))
            openness = float(getattr(frame, "openness", getattr(frame, "mouth_open", 0.0)))
            rounding = float(getattr(frame, "rounding", getattr(frame, "round", 0.0)))
            closure = float(getattr(frame, "closure", getattr(frame, "mouth_close", 1.0)))
            spread = float(getattr(frame, "lip_spread", getattr(frame, "spread", 0.0)))
            jaw = float(getattr(frame, "jaw", getattr(frame, "jaw_drop", openness)))
            energy = float(getattr(frame, "energy", getattr(frame, "audio_energy", 0.0)))
            timestamp = float(getattr(frame, "timestamp", time.time()))

        self.viseme_label = label
        self.viseme_open = exp_smooth(self.viseme_open, _tts_runtime_safe_clamp(openness, 0.0, 1.0), 0.016, 0.05)
        self.viseme_round = exp_smooth(self.viseme_round, _tts_runtime_safe_clamp(rounding, 0.0, 1.0), 0.016, 0.06)
        self.viseme_spread = exp_smooth(self.viseme_spread, _tts_runtime_safe_clamp(spread, 0.0, 1.0), 0.016, 0.06)
        self.viseme_close = exp_smooth(self.viseme_close, _tts_runtime_safe_clamp(closure, 0.0, 1.0), 0.016, 0.06)
        self.viseme_jaw = exp_smooth(self.viseme_jaw, _tts_runtime_safe_clamp(jaw, 0.0, 1.0), 0.016, 0.05)
        self.viseme_energy = exp_smooth(self.viseme_energy, _tts_runtime_safe_clamp(energy, 0.0, 1.0), 0.016, 0.07)
        self.viseme_timestamp = timestamp
        self.viseme_active = True
        self.viseme_time = 0.0
        self.viseme_decay = 1.0

        cb = getattr(self, "_viseme_listener", None)
        if cb is not None:
            try:
                cb(frame)
            except Exception:
                pass
    except Exception:
        logger.exception("Falha ao processar visema")


def _audio_sync_reset(self) -> None:
    _audio_sync_reset_orig(self)
    _audio_sync_init_viseme(self)


def _audio_sync_update(self, dt: float, is_speaking: bool) -> None:
    _audio_sync_update_orig(self, dt, is_speaking)
    self.viseme_time += dt
    if not self.viseme_active:
        self.viseme_open = exp_smooth(self.viseme_open, 0.0, dt, 0.11)
        self.viseme_round = exp_smooth(self.viseme_round, 0.0, dt, 0.11)
        self.viseme_spread = exp_smooth(self.viseme_spread, 0.0, dt, 0.11)
        self.viseme_close = exp_smooth(self.viseme_close, 1.0, dt, 0.11)
        self.viseme_jaw = exp_smooth(self.viseme_jaw, 0.0, dt, 0.11)
        self.viseme_energy = exp_smooth(self.viseme_energy, 0.0, dt, 0.11)
    else:
        self.viseme_decay = exp_smooth(self.viseme_decay, 0.0, dt, 0.22)
        if self.viseme_decay < 0.02:
            self.viseme_active = False


def _audio_sync_mouth_open(self, state: EmotionState, plan: Optional[LLMPlan] = None) -> float:
    base = _audio_sync_mouth_open_orig(self, state, plan)
    viseme_drive = (
        0.84 * getattr(self, "viseme_open", 0.0)
        + 0.26 * getattr(self, "viseme_jaw", 0.0)
        + 0.18 * getattr(self, "viseme_energy", 0.0)
        - 0.42 * getattr(self, "viseme_close", 1.0)
        + 0.12 * getattr(self, "viseme_spread", 0.0)
        - 0.10 * getattr(self, "viseme_round", 0.0)
    )
    if getattr(self, "speaking", False) or getattr(self, "proxy_active", False):
        base = max(base, 0.07 + 0.48 * self.energy + 0.18 * self.attack + 0.22 * viseme_drive)
    else:
        base = max(base * 0.55, 0.02 + state.energy * 0.05 + 0.08 * viseme_drive)
    if state.current_state in {"shy", "hide", "retreat"}:
        base *= 0.86
    if state.current_state in {"surprised", "reacting"}:
        base *= 1.08
    if state.emotion in {"happy", "confident"}:
        base *= 1.03
    return clamp(base, 0.0, 1.0)


def _audio_sync_register_viseme_api(self):
    if not hasattr(self, "_viseme_listener"):
        self._viseme_listener = None


AudioSyncEngine.__init__ = _audio_sync_init  # type: ignore[assignment]
AudioSyncEngine.reset = _audio_sync_reset  # type: ignore[assignment]
AudioSyncEngine.update = _audio_sync_update  # type: ignore[assignment]
AudioSyncEngine.mouth_open = _audio_sync_mouth_open  # type: ignore[assignment]
AudioSyncEngine.feed_viseme = _audio_sync_feed_viseme  # type: ignore[assignment]
AudioSyncEngine.register_viseme_listener = _audio_sync_register_viseme_listener  # type: ignore[assignment]


# -----------------------------
# AvatarApp TTS wiring patch
# -----------------------------
_avatarapp_maybe_load_tts_orig = AvatarApp._maybe_load_tts
_avatarapp_play_audio_stream_orig = AvatarApp._play_audio_stream


def _avatarapp_maybe_load_tts(self) -> None:
    _avatarapp_maybe_load_tts_orig(self)
    tts = getattr(self, "tts", None)
    audio = getattr(self, "audio", None)
    if tts is None or audio is None:
        return
    try:
        if hasattr(tts, "register_audio_listener"):
            tts.register_audio_listener(getattr(audio, "feed", None))
        if hasattr(tts, "register_viseme_listener") and hasattr(audio, "feed_viseme"):
            tts.register_viseme_listener(getattr(audio, "feed_viseme", None))
        if hasattr(tts, "register_echo_reference_sink") and hasattr(audio, "feed_playback_chunk"):
            tts.register_echo_reference_sink(getattr(audio, "feed_playback_chunk", None))
    except Exception:
        logger.exception("Falha ao ligar TTS ao sincronizador de áudio")


def _avatarapp_play_audio_stream(self, audio_gen) -> None:
    try:
        tts = getattr(self, "tts", None)
        if tts is not None and hasattr(tts, "play_stream"):
            try:
                tts.play_stream(audio_gen, sample_rate=getattr(tts, "sample_rate", 24000), stop_event=self.speech_cancel)
            except Exception:
                logger.debug("play_stream falhou; usando fallback manual", exc_info=True)
            finally:
                if not self.speech_cancel.is_set():
                    try:
                        self.root.after(0, self._stop_speaking)
                    except Exception:
                        self._stop_speaking()
            return

        for chunk in audio_gen:
            if self.speech_cancel.is_set():
                break
            if chunk is None:
                continue
            if hasattr(self.audio, "feed"):
                self.audio.feed(chunk)
            if sd is not None:
                try:
                    sd.play(chunk, samplerate=24000)
                    sd.wait()
                except Exception:
                    break
            if self.speech_cancel.is_set():
                break
    except Exception:
        logger.exception("Falha ao reproduzir stream de áudio")
    finally:
        if not self.speech_cancel.is_set():
            try:
                self.root.after(0, self._stop_speaking)
            except Exception:
                try:
                    self._stop_speaking()
                except Exception:
                    pass


AvatarApp._maybe_load_tts = _avatarapp_maybe_load_tts  # type: ignore[assignment]
AvatarApp._play_audio_stream = _avatarapp_play_audio_stream  # type: ignore[assignment]



if __name__ == "__main__":
    _AVATAR_MAIN_DEFERRED = True

# ======================================================================
#  Legacy / alternative engine stack (renamed to V2 to avoid collisions)
# ======================================================================

import json
import math
import queue
import random
import threading
import time
import pickle
import logging
import os
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageTk
try:
    from voice_input_bridge import VoiceInputConfig, install_voice_input_bridge
    print("Loaded: Voice Input Bridge")
except Exception as e:
    print(f"Skipped Voice Input Bridge: {e}")
try:
    from occupancy_visibility_transformer import (
        OccupancyVisibilityController,
        SpaceOccupancyInput,
        SpaceOccupancyState,
    )
except Exception:
    OccupancyVisibilityController = None
    SpaceOccupancyInput = None
    SpaceOccupancyState = None

try:
    from animation_fluidity_transformers import (
        AnimationFluidityTransformer,
        AnimationFluidityInput,
    )
except Exception:
    AnimationFluidityTransformer = None
    AnimationFluidityInput = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from model_manager import ModelManager
except Exception:
    ModelManager = None

try:
    from tts_local import LocalTTSManager
except Exception:
    LocalTTSManager = None

# ----------------------------------------------------------------------
#  CONSTANTES E UTILITÁRIOS
# ----------------------------------------------------------------------

STATE_NAMES = [
    "idle", "listening", "thinking", "speaking", "reacting",
    "curious", "surprised", "shy", "hide", "peek", "retreat",
    "lean_in", "observe", "rest"
]

EMOTIONS = [
    "neutral", "curious", "happy", "calm", "focused",
    "surprised", "shy", "confident", "tired", "annoyed"
]

GESTURES = [
    "none", "nod", "tilt_left", "tilt_right", "lean_in", "retreat",
    "peek", "micro_shrug", "soft_smile", "glance_away", "present"
]

GAZE_TARGETS = ["user", "mouse", "away", "internal", "edge", "text"]


def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def exp_smooth(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-dt / tau)
    return lerp(current, target, k)


def soft_noise(seed: float, t: float, freq: float = 1.0) -> float:
    return (math.sin(t * freq + seed) * 0.5 +
            math.sin(t * freq * 1.71 + seed * 1.37) * 0.32 +
            math.sin(t * freq * 2.63 + seed * 0.73) * 0.18)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def text_summary(text: str, max_len: int = 120) -> str:
    t = " ".join((text or "").strip().split())
    return t[:max_len]


# ----------------------------------------------------------------------
#  DATACLASSES PARA ESTADOS E PERCEPÇÃO
# ----------------------------------------------------------------------

@dataclass
class PerceptionSnapshotV2:
    t: float = 0.0
    dt: float = 0.016
    mouse_x: int = 0
    mouse_y: int = 0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_speed: float = 0.0
    mouse_distance_to_face: float = 0.0
    click: bool = False
    double_click: bool = False
    focus: bool = True
    silence_s: float = 0.0
    audio_rms: float = 0.0
    audio_peak: float = 0.0
    audio_attack: float = 0.0
    audio_voiced: bool = False
    user_text: str = ""
    user_text_len: int = 0
    speaking: bool = False
    window_active: bool = True
    turn_index: int = 0


@dataclass
class InternalState:
    """Estado interno contínuo do avatar."""
    valence: float = 0.0
    arousal: float = 0.35
    dominance: float = 0.4
    attention: float = 0.48
    curiosity: float = 0.32
    confidence: float = 0.52
    tension: float = 0.12
    fatigue: float = 0.08
    social_drive: float = 0.28
    energy: float = 0.45
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.valence, self.arousal, self.dominance, self.attention,
            self.curiosity, self.confidence, self.tension, self.fatigue,
            self.social_drive, self.energy
        ], dtype=torch.float32)
    
    def from_tensor(self, t: torch.Tensor):
        self.valence = float(t[0])
        self.arousal = float(t[1])
        self.dominance = float(t[2])
        self.attention = float(t[3])
        self.curiosity = float(t[4])
        self.confidence = float(t[5])
        self.tension = float(t[6])
        self.fatigue = float(t[7])
        self.social_drive = float(t[8])
        self.energy = float(t[9])


@dataclass
class MemoryTurnV2:
    t: float
    speaker: str
    text: str
    emotion: str = "neutral"
    intention: str = "none"
    state: str = "idle"
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None


@dataclass
class TransformerOutput:
    text: str = ""
    emotion: str = "neutral"
    intensity: float = 0.35
    intention: str = "respond"
    attention_target: str = "user"
    gaze_target: str = "user"
    gesture: str = "none"
    gesture_strength: float = 0.35
    body_posture: Dict[str, float] = field(default_factory=dict)
    motion_speed: float = 0.45
    pause_ms: int = 120
    silence_response: bool = False
    confidence: float = 0.55
    curiosity: float = 0.5
    tension: float = 0.2
    memory_update: Dict[str, Any] = field(default_factory=dict)
    state_transition: str = "speaking"


# ----------------------------------------------------------------------
#  TRANSFORMER ARCHITECTURE
# ----------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class AvatarTransformer(nn.Module):
    """
    Transformer que atua como o cérebro decisor do avatar.
    Recebe embedding de contexto e produz ações estruturadas.
    """
    def __init__(self, 
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 vocab_size: int = 50000,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.state_embedding = nn.Linear(10, d_model)  # 10 dimensões do InternalState
        self.stimulus_embedding = nn.Linear(12, d_model)  # 12 estímulos de percepção
        self.memory_embedding = nn.Linear(128, d_model)  # embedding de memória consolidada
        
        # Posicional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.text_head = nn.Linear(d_model, vocab_size)
        self.emotion_head = nn.Linear(d_model, len(EMOTIONS))
        self.intensity_head = nn.Linear(d_model, 1)
        self.intention_head = nn.Linear(d_model, 8)  # 8 intenções
        self.gaze_head = nn.Linear(d_model, len(GAZE_TARGETS))
        self.gesture_head = nn.Linear(d_model, len(GESTURES))
        self.gesture_strength_head = nn.Linear(d_model, 1)
        self.motion_speed_head = nn.Linear(d_model, 1)
        self.pause_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)
        self.curiosity_head = nn.Linear(d_model, 1)
        self.tension_head = nn.Linear(d_model, 1)
        self.state_transition_head = nn.Linear(d_model, len(STATE_NAMES))
        
        # Memory projection
        self.memory_projection = nn.Linear(d_model, 128)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                token_ids: torch.Tensor,
                state: torch.Tensor,
                stimulus: torch.Tensor,
                memory: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len] tokens do texto do usuário
            state: [batch, 10] estado interno atual
            stimulus: [batch, 12] vetor de estímulos
            memory: [batch, 128] embedding de memória consolidada
            attention_mask: [batch, seq_len] máscara de atenção
        
        Returns:
            Dicionário com todas as saídas estruturadas
        """
        batch_size = token_ids.shape[0]
        seq_len = token_ids.shape[1]
        
        # Embeddings
        token_emb = self.token_embedding(token_ids)  # [batch, seq_len, d_model]
        state_emb = self.state_embedding(state).unsqueeze(1)  # [batch, 1, d_model]
        stimulus_emb = self.stimulus_embedding(stimulus).unsqueeze(1)  # [batch, 1, d_model]
        memory_emb = self.memory_embedding(memory).unsqueeze(1)  # [batch, 1, d_model]
        
        # Concatenar todas as fontes de informação
        combined = torch.cat([token_emb, state_emb, stimulus_emb, memory_emb], dim=1)
        
        # Adicionar posicional encoding
        combined = self.pos_encoding(combined)
        combined = self.dropout(combined)
        
        # Passar pelo transformer
        for layer in self.layers:
            combined = layer(combined, attention_mask)
        
        # Extrair o token [CLS] (último token combinado)
        cls_embedding = combined[:, -1, :]  # [batch, d_model]
        
        # Gerar saídas
        outputs = {
            'text_logits': self.text_head(cls_embedding),
            'emotion_logits': self.emotion_head(cls_embedding),
            'intensity': torch.sigmoid(self.intensity_head(cls_embedding)).squeeze(-1),
            'intention_logits': self.intention_head(cls_embedding),
            'gaze_logits': self.gaze_head(cls_embedding),
            'gesture_logits': self.gesture_head(cls_embedding),
            'gesture_strength': torch.sigmoid(self.gesture_strength_head(cls_embedding)).squeeze(-1),
            'motion_speed': torch.sigmoid(self.motion_speed_head(cls_embedding)).squeeze(-1),
            'pause_ms': torch.sigmoid(self.pause_head(cls_embedding)).squeeze(-1) * 500,  # 0-500ms
            'confidence': torch.sigmoid(self.confidence_head(cls_embedding)).squeeze(-1),
            'curiosity': torch.sigmoid(self.curiosity_head(cls_embedding)).squeeze(-1),
            'tension': torch.sigmoid(self.tension_head(cls_embedding)).squeeze(-1),
            'state_transition_logits': self.state_transition_head(cls_embedding),
            'memory_update': self.memory_projection(cls_embedding)
        }
        
        return outputs
    
    def generate_action(self, 
                        token_ids: torch.Tensor,
                        state: torch.Tensor,
                        stimulus: torch.Tensor,
                        memory: torch.Tensor,
                        temperature: float = 0.7) -> TransformerOutput:
        """
        Gera uma ação estruturada a partir das entradas.
        """
        with torch.no_grad():
            outputs = self.forward(token_ids, state, stimulus, memory)
        
        # Decodificar saídas
        emotion_idx = torch.argmax(outputs['emotion_logits'], dim=-1).item()
        intention_idx = torch.argmax(outputs['intention_logits'], dim=-1).item()
        gaze_idx = torch.argmax(outputs['gaze_logits'], dim=-1).item()
        gesture_idx = torch.argmax(outputs['gesture_logits'], dim=-1).item()
        state_idx = torch.argmax(outputs['state_transition_logits'], dim=-1).item()
        
        # Mapeamento de intenções (simplificado)
        intentions = ["respond", "ask", "explain", "empathize", "react", "observe", "wait", "initiate"]
        
        return TransformerOutput(
            text="",  # Será gerado pelo LLM
            emotion=EMOTIONS[emotion_idx],
            intensity=float(outputs['intensity'].item()),
            intention=intentions[intention_idx] if intention_idx < len(intentions) else "respond",
            attention_target=GAZE_TARGETS[gaze_idx],
            gaze_target=GAZE_TARGETS[gaze_idx],
            gesture=GESTURES[gesture_idx],
            gesture_strength=float(outputs['gesture_strength'].item()),
            body_posture={},
            motion_speed=float(outputs['motion_speed'].item()),
            pause_ms=int(outputs['pause_ms'].item()),
            silence_response=False,
            confidence=float(outputs['confidence'].item()),
            curiosity=float(outputs['curiosity'].item()),
            tension=float(outputs['tension'].item()),
            memory_update={},
            state_transition=STATE_NAMES[state_idx]
        )


# ----------------------------------------------------------------------
#  PERCEPTION ENGINE
# ----------------------------------------------------------------------

class PerceptionEngine:
    """Processa estímulos sensoriais em vetor de características."""
    
    def __init__(self):
        self.last_mouse_pos = (0, 0)
        self.last_mouse_time = 0.0
        
    def process(self, perception: PerceptionSnapshotV2) -> torch.Tensor:
        """
        Converte percepção bruta em vetor de características.
        Dimensão: 12
        """
        # Normalizar distâncias (0-1)
        mouse_dist_norm = min(1.0, perception.mouse_distance_to_face / 500.0)
        mouse_speed_norm = min(1.0, perception.mouse_speed / 1000.0)
        
        features = torch.tensor([
            mouse_dist_norm,                    # 0
            mouse_speed_norm,                   # 1
            1.0 if perception.click else 0.0,   # 2
            1.0 if perception.double_click else 0.0,  # 3
            perception.audio_rms,               # 4
            perception.audio_peak,              # 5
            perception.audio_attack,            # 6
            min(1.0, perception.silence_s / 30.0),  # 7
            1.0 if perception.audio_voiced else 0.0, # 8
            1.0 if perception.speaking else 0.0,     # 9
            min(1.0, perception.user_text_len / 100.0),  # 10
            1.0 if perception.focus else 0.0     # 11
        ], dtype=torch.float32)
        
        return features.unsqueeze(0)  # [batch=1, 12]


# ----------------------------------------------------------------------
#  MEMORY ENGINE
# ----------------------------------------------------------------------

class MemoryEngineV2:
    """Memória em camadas com consolidação e embeddings."""
    
    def __init__(self, embedding_dim: int = 128, short_limit: int = 24, long_limit: int = 200):
        self.short_term: Deque[MemoryTurnV2] = deque(maxlen=short_limit)
        self.long_term: List[MemoryTurnV2] = []
        self.long_limit = long_limit
        self.embedding_dim = embedding_dim
        self.topics: Counter[str] = Counter()
        self.user_preferences: Dict[str, Any] = {}
        self.session_notes: List[str] = []
        self.last_user_text: str = ""
        self.last_assistant_text: str = ""
        
        # Embedding model placeholder (usaria um modelo real)
        self.embedding_model = None
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding para texto (placeholder - implementar com modelo real)."""
        # Simulação: hash simples para embedding
        if not text:
            return np.zeros(self.embedding_dim)
        
        # Gerar embedding determinístico baseado no texto
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(self.embedding_dim) * 0.1
        return emb / np.linalg.norm(emb)
    
    def remember_turn(self, speaker: str, text: str, emotion: str = "neutral",
                      intention: str = "none", state: str = "idle",
                      importance: float = 0.5, t: float = 0.0) -> None:
        embedding = self._get_embedding(text)
        turn = MemoryTurnV2(
            t=t, speaker=speaker, text=text, emotion=emotion,
            intention=intention, state=state, importance=importance,
            embedding=embedding
        )
        self.short_term.append(turn)
        
        if speaker == "user":
            self.last_user_text = text
        else:
            self.last_assistant_text = text
            
        topic = self._extract_topic(text)
        if topic:
            self.topics[topic] += 1
    
    def _extract_topic(self, text: str) -> str:
        if not text:
            return ""
        words = [w.strip(".,!?;:()[]{}\"'\"").lower() for w in text.split()]
        words = [w for w in words if len(w) > 3]
        if not words:
            return ""
        stop = {"que", "para", "com", "uma", "isso", "esta", "esse", "sobre", "como"}
        filtered = [w for w in words if w not in stop]
        return filtered[0] if filtered else words[0]
    
    def consolidate(self, current_time: float, importance_threshold: float = 0.65) -> None:
        """Consolida memórias importantes para longo prazo."""
        for turn in list(self.short_term):
            if turn.importance >= importance_threshold:
                self.long_term.append(turn)
        
        if len(self.long_term) > self.long_limit:
            self.long_term = self.long_term[-self.long_limit:]
    
    def get_consolidated_embedding(self) -> torch.Tensor:
        """Gera embedding consolidado da memória."""
        if not self.long_term and not self.short_term:
            return torch.zeros(1, self.embedding_dim)
        
        # Média dos embeddings das memórias mais importantes
        all_memories = list(self.short_term) + self.long_term
        important_memories = sorted(all_memories, key=lambda x: x.importance, reverse=True)[:10]
        
        if not important_memories:
            return torch.zeros(1, self.embedding_dim)
        
        embeddings = [m.embedding for m in important_memories if m.embedding is not None]
        if not embeddings:
            return torch.zeros(1, self.embedding_dim)
        
        avg_embedding = np.mean(embeddings, axis=0)
        return torch.from_numpy(avg_embedding).float().unsqueeze(0)
    
    def get_context(self) -> str:
        """Retorna resumo de contexto para LLM."""
        last_user = text_summary(self.last_user_text, 90)
        last_assistant = text_summary(self.last_assistant_text, 90)
        topics = ", ".join([f"{k}" for k, _ in self.topics.most_common(3)])
        return f"user={last_user} | assistant={last_assistant} | topics={topics}"


# ----------------------------------------------------------------------
#  EMOTION ENGINE
# ----------------------------------------------------------------------

class EmotionEngineV2:
    """Sistema emocional contínuo com inércia e recuperação."""
    
    def __init__(self):
        self.state = InternalState()
        self.emotion_label = "neutral"
        self._inertia = 0.15
        
    def update(self, stimulus: torch.Tensor, plan: Optional[TransformerOutput] = None,
               dt: float = 0.016) -> InternalState:
        """Atualiza estado emocional baseado em estímulos e plano."""
        
        # Extrair estímulos
        mouse_dist = float(stimulus[0, 0])
        audio_energy = float(stimulus[0, 4])
        silence = float(stimulus[0, 7])
        speaking = float(stimulus[0, 9]) > 0.5
        
        # Atualizar alvos baseados em estímulos
        target_valence = 0.0
        target_arousal = 0.35
        target_curiosity = 0.32
        target_confidence = 0.52
        target_tension = 0.12
        
        # Mouse próximo aumenta curiosidade e tensão
        if mouse_dist < 0.3:
            target_curiosity += 0.2
            target_tension += 0.1
        
        # Áudio aumenta arousal
        if audio_energy > 0.3:
            target_arousal += audio_energy * 0.4
        
        # Silêncio prolongado diminui arousal e aumenta fadiga
        if silence > 0.5:
            target_arousal -= 0.15
            self.state.fatigue += 0.01 * dt * 30
        
        # Falando aumenta confiança
        if speaking:
            target_confidence += 0.1
        
        # Plano pode modificar alvos
        if plan:
            if plan.emotion == "happy":
                target_valence += 0.3
                target_arousal += 0.2
            elif plan.emotion == "curious":
                target_curiosity += 0.25
            elif plan.emotion == "confident":
                target_confidence += 0.2
            elif plan.emotion == "shy":
                target_confidence -= 0.15
                target_tension += 0.1
            
            target_curiosity += plan.curiosity * 0.3
            target_tension += plan.tension * 0.2
        
        # Aplicar suavização
        tau = 0.2
        self.state.valence = exp_smooth(self.state.valence, clamp(target_valence, -1, 1), dt, tau)
        self.state.arousal = exp_smooth(self.state.arousal, clamp(target_arousal, 0, 1), dt, tau)
        self.state.curiosity = exp_smooth(self.state.curiosity, clamp(target_curiosity, 0, 1), dt, tau)
        self.state.confidence = exp_smooth(self.state.confidence, clamp(target_confidence, 0, 1), dt, tau)
        self.state.tension = exp_smooth(self.state.tension, clamp(target_tension, 0, 1), dt, tau)
        self.state.fatigue = exp_smooth(self.state.fatigue, clamp(self.state.fatigue, 0, 1), dt, tau * 1.5)
        
        # Energia
        target_energy = 0.45 + self.state.arousal * 0.3 - self.state.fatigue * 0.2
        self.state.energy = exp_smooth(self.state.energy, clamp(target_energy, 0, 1), dt, tau)
        
        # Determinar label de emoção
        self.emotion_label = self._get_emotion_label()
        
        return self.state
    
    def _get_emotion_label(self) -> str:
        """Converte estado contínuo para label discreta."""
        if self.state.fatigue > 0.7:
            return "tired"
        if self.state.tension > 0.7:
            return "annoyed"
        if self.state.arousal > 0.7 and self.state.valence < -0.1:
            return "surprised"
        if self.state.confidence > 0.7 and self.state.valence > 0.15:
            return "confident"
        if self.state.curiosity > 0.6 and self.state.arousal > 0.4:
            return "curious"
        if self.state.valence > 0.2 and self.state.arousal > 0.4:
            return "happy"
        if self.state.arousal < 0.25 and abs(self.state.valence) < 0.15:
            return "calm"
        if self.state.confidence > 0.55 and self.state.arousal > 0.4:
            return "focused"
        if self.state.valence < -0.2:
            return "shy"
        return "neutral"


# ----------------------------------------------------------------------
#  ATTENTION ENGINE
# ----------------------------------------------------------------------

class AttentionEngineV2:
    """Sistema de atenção com saliência e foco dinâmico."""
    
    def __init__(self):
        self.target = "user"
        self.target_point = (0.0, 0.0)
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.blink_timer = 0.0
        self.next_blink = random.uniform(2.0, 4.8)
        self.blink_amount = 0.0
        self.saccade_seed = random.random() * 100.0
        
        # Saliencia dos estímulos
        self.salience = {
            "mouse": 0.0, "click": 0.0, "speech": 0.0, "silence": 0.0, "text": 0.0
        }
        self.salience_decay = 0.95
        
    def update(self, perception: PerceptionSnapshotV2, state: InternalState,
               plan: Optional[TransformerOutput], dt: float) -> str:
        """Atualiza alvo de atenção."""
        
        # Atualizar saliência
        if perception.mouse_distance_to_face < 300:
            self.salience["mouse"] = 1.0
        else:
            self.salience["mouse"] *= self.salience_decay ** dt
        
        if perception.click or perception.double_click:
            self.salience["click"] = 1.0
        else:
            self.salience["click"] *= self.salience_decay ** dt
        
        if perception.audio_voiced or perception.user_text_len > 0:
            self.salience["speech"] = 1.0
        else:
            self.salience["speech"] *= self.salience_decay ** dt
        
        if perception.silence_s > 5.0:
            self.salience["silence"] = min(1.0, self.salience["silence"] + 0.2 * dt)
        else:
            self.salience["silence"] *= self.salience_decay ** dt
        
        # Decidir alvo
        if plan and plan.attention_target:
            self.target = plan.attention_target
        elif self.salience["click"] > 0.5:
            self.target = "user"
        elif self.salience["speech"] > 0.4:
            self.target = "user"
        elif self.salience["mouse"] > 0.6 and state.curiosity > 0.4:
            self.target = "mouse"
        elif self.salience["silence"] > 0.7 and state.curiosity > 0.5:
            self.target = "internal"
        elif state.curiosity > 0.6:
            self.target = "mouse"
        elif state.attention < 0.3:
            self.target = "away"
        else:
            self.target = "user"
        
        return self.target
    
    def target_point_for(self, target: str, perception: PerceptionSnapshotV2) -> Tuple[float, float]:
        """Retorna ponto de olhar para o alvo."""
        w, h = 1024.0, 1024.0
        if target == "user":
            return w / 2, h / 2 - 55
        if target == "mouse":
            return float(perception.mouse_x), float(perception.mouse_y)
        if target == "away":
            return w * 0.70, h * 0.34
        if target == "internal":
            return w * 0.42, h * 0.30
        return w / 2, h / 2
    
    def update_gaze(self, dt: float, target_point: Tuple[float, float],
                    face_center: Tuple[float, float]) -> Tuple[float, float]:
        """Atualiza posição do olhar com micro-sacadas."""
        tx, ty = target_point
        cx, cy = face_center
        dx = tx - cx
        dy = ty - cy
        
        # Micro-sacadas
        sacc = soft_noise(self.saccade_seed, time.time(), 2.5)
        dx += sacc * 8.0
        dy += sacc * 5.0
        
        self.gaze_x = exp_smooth(self.gaze_x, clamp(dx / 30.0, -1.0, 1.0), dt, 0.08)
        self.gaze_y = exp_smooth(self.gaze_y, clamp(dy / 28.0, -1.0, 1.0), dt, 0.08)
        
        return self.gaze_x, self.gaze_y
    
    def update_blink(self, dt: float, state: InternalState) -> float:
        """Atualiza estado de piscar."""
        self.blink_timer += dt
        if self.blink_amount <= 0.0 and self.blink_timer >= self.next_blink:
            self.blink_amount = 1.0
            self.blink_timer = 0.0
            base = 2.5 if state.arousal > 0.6 else 3.5
            self.next_blink = random.uniform(base, base + 2.0)
        
        if self.blink_amount > 0.0:
            self.blink_amount = max(0.0, self.blink_amount - dt * 12.0)
        
        return self.blink_amount


# ----------------------------------------------------------------------
#  BEHAVIOR ENGINE
# ----------------------------------------------------------------------

class BehaviorEngineV2:
    """Máquina de estados com transições suaves."""
    
    def __init__(self):
        self.current_state = "idle"
        self.state_enter_t = 0.0
        self.state_dwell = 0.0
        self.force_target_state: Optional[str] = None
        
        self.state_profiles = {
            "idle": {"speed": 0.30, "attention": 0.45, "energy": 0.35},
            "listening": {"speed": 0.38, "attention": 0.78, "energy": 0.42},
            "thinking": {"speed": 0.22, "attention": 0.38, "energy": 0.28},
            "speaking": {"speed": 0.78, "attention": 0.88, "energy": 0.76},
            "reacting": {"speed": 0.95, "attention": 0.92, "energy": 0.90},
            "curious": {"speed": 0.42, "attention": 0.82, "energy": 0.52},
            "surprised": {"speed": 0.96, "attention": 1.00, "energy": 0.95},
            "shy": {"speed": 0.25, "attention": 0.48, "energy": 0.33},
            "hide": {"speed": 0.18, "attention": 0.28, "energy": 0.22},
            "peek": {"speed": 0.34, "attention": 0.70, "energy": 0.41},
            "retreat": {"speed": 0.28, "attention": 0.36, "energy": 0.24},
            "lean_in": {"speed": 0.55, "attention": 0.86, "energy": 0.66},
            "observe": {"speed": 0.22, "attention": 0.50, "energy": 0.29},
            "rest": {"speed": 0.14, "attention": 0.22, "energy": 0.16},
        }
        
    def update(self, perception: PerceptionSnapshotV2, state: InternalState,
               plan: Optional[TransformerOutput], t: float) -> str:
        """Atualiza estado atual baseado em percepção e plano."""
        
        if self.force_target_state:
            self.current_state = self.force_target_state
            self.force_target_state = None
            self.state_enter_t = t
        
        elif plan and plan.state_transition:
            if plan.state_transition != self.current_state and t - self.state_enter_t > 0.3:
                self.current_state = plan.state_transition
                self.state_enter_t = t
        
        else:
            # Decisão baseada em heurística
            if perception.speaking:
                candidate = "speaking"
            elif perception.audio_voiced:
                candidate = "listening"
            elif perception.mouse_distance_to_face < 150 and state.confidence < 0.55:
                candidate = "retreat"
            elif perception.mouse_distance_to_face < 200 and state.curiosity > 0.48:
                candidate = "curious"
            elif perception.silence_s > 8.0:
                candidate = "observe"
            elif perception.silence_s > 14.0:
                candidate = "rest"
            else:
                candidate = "idle"
            
            if candidate != self.current_state and t - self.state_enter_t > 0.5:
                self.current_state = candidate
                self.state_enter_t = t
        
        self.state_dwell = t - self.state_enter_t
        return self.current_state
    
    def state_profile(self) -> Dict[str, float]:
        return self.state_profiles.get(self.current_state, self.state_profiles["idle"])


# ----------------------------------------------------------------------
#  MOTION ENGINE
# ----------------------------------------------------------------------

class MotionEngineV2:
    """Gera movimento contínuo e expressões faciais."""
    
    def __init__(self):
        self.seed = random.random() * 200.0
        self.body_lean_x = 0.0
        self.body_lean_y = 0.0
        self.head_x = 0.0
        self.head_y = 0.0
        self.eye_open = 1.0
        self.mouth_open = 0.0
        self.brow_left = 0.0
        self.brow_right = 0.0
        self.mouth_curve = 0.0
        
    def update(self, perception: PerceptionSnapshotV2, state: InternalState,
               behavior: BehaviorEngineV2, attention: AttentionEngineV2,
               plan: Optional[TransformerOutput], dt: float) -> Dict[str, Any]:
        """Atualiza todos os parâmetros de movimento."""
        
        profile = behavior.state_profile()
        t = perception.t
        
        # Respiração contínua
        breath = 0.5 + 0.5 * math.sin(t * 1.05 + self.seed)
        
        # Micro-movimentos
        micro_x = soft_noise(self.seed, t, 0.3) * 2.0
        micro_y = soft_noise(self.seed + 1, t, 0.35) * 1.5
        
        # Baseado em estado
        speed_factor = profile["speed"] * (0.6 + 0.4 * state.energy)
        
        # Movimento corporal
        target_lean_x = micro_x * 5.0 * state.curiosity
        target_lean_y = micro_y * 4.0 - state.fatigue * 3.0 + (breath - 0.5) * 6.0
        
        if behavior.current_state == "lean_in":
            target_lean_y -= 8.0
        elif behavior.current_state in {"retreat", "hide"}:
            target_lean_y += 6.0
            target_lean_x -= 4.0
        
        self.body_lean_x = exp_smooth(self.body_lean_x, target_lean_x, dt, 0.12)
        self.body_lean_y = exp_smooth(self.body_lean_y, target_lean_y, dt, 0.12)
        
        # Cabeça
        target_head_x = micro_x * 4.0 + (perception.mouse_x - 512) / 512.0 * 8.0 * attention.salience["mouse"]
        target_head_y = micro_y * 3.0 - state.fatigue * 2.0
        
        self.head_x = exp_smooth(self.head_x, target_head_x, dt, 0.1)
        self.head_y = exp_smooth(self.head_y, target_head_y, dt, 0.1)
        
        # Expressões faciais baseadas em emoção
        if state.valence > 0.2:
            self.mouth_curve = 0.3 + state.valence * 0.4
            self.brow_left = -0.1
            self.brow_right = -0.1
        elif state.valence < -0.2:
            self.mouth_curve = -0.1
            self.brow_left = 0.15
            self.brow_right = 0.15
        else:
            self.mouth_curve = 0.1
            self.brow_left = 0.0
            self.brow_right = 0.0
        
        # Abertura dos olhos
        target_eye_open = 1.0 - state.fatigue * 0.3 + state.arousal * 0.2
        target_eye_open = clamp(target_eye_open, 0.6, 1.3)
        self.eye_open = exp_smooth(self.eye_open, target_eye_open, dt, 0.15)
        
        # Abertura da boca (baseada em áudio)
        if perception.speaking:
            target_mouth = perception.audio_rms * 0.8 + perception.audio_attack * 0.3
            target_mouth = clamp(target_mouth, 0.05, 0.85)
        else:
            target_mouth = 0.02 + state.energy * 0.05
        
        self.mouth_open = exp_smooth(self.mouth_open, target_mouth, dt, 0.08)
        
        return {
            "body_lean_x": self.body_lean_x,
            "body_lean_y": self.body_lean_y,
            "head_x": self.head_x,
            "head_y": self.head_y,
            "eye_open": self.eye_open,
            "mouth_open": self.mouth_open,
            "brow_left": self.brow_left,
            "brow_right": self.brow_right,
            "mouth_curve": self.mouth_curve,
            "breath": breath
        }


# ----------------------------------------------------------------------
#  AUDIO SYNC ENGINE
# ----------------------------------------------------------------------

class AudioSyncEngineV2:
    """Sincronização de áudio em tempo real."""
    
    def __init__(self):
        self.energy = 0.0
        self.peak = 0.0
        self.attack = 0.0
        self.voiced = False
        self.speaking = False
        self.proxy_active = False
        self.proxy_time = 0.0
        self.proxy_duration = 0.0
        
    def feed(self, chunk: Optional[np.ndarray]) -> None:
        """Processa chunk de áudio."""
        if chunk is None:
            self.voiced = False
            return
        
        data = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return
        
        rms = float(np.sqrt(np.mean(np.square(data))))
        peak = float(np.max(np.abs(data)))
        
        self.energy = exp_smooth(self.energy, rms, 0.016, 0.08)
        self.peak = peak
        self.attack = max(0.0, (rms - self.energy) * 5.0)
        self.voiced = rms > 0.02 or peak > 0.05
        
    def start_proxy(self, text: str, duration: float = 2.0) -> None:
        """Inicia proxy de áudio (quando não há TTS)."""
        self.proxy_active = True
        self.proxy_time = 0.0
        self.proxy_duration = duration
        
    def update(self, dt: float, is_speaking: bool) -> None:
        """Atualiza estado de áudio."""
        self.speaking = is_speaking
        
        if self.proxy_active:
            self.proxy_time += dt
            if self.proxy_time >= self.proxy_duration:
                self.proxy_active = False
    
    def mouth_open(self, state: InternalState, plan: Optional[TransformerOutput]) -> float:
        """Calcula abertura da boca baseada em áudio e estado."""
        if self.speaking or self.proxy_active:
            base = self.energy * 0.7 + self.attack * 0.3
            base = clamp(base, 0.05, 0.85)
        else:
            base = 0.02 + state.energy * 0.05
        
        if plan and plan.state_transition == "speaking":
            base *= 1.2
        
        return clamp(base, 0.0, 1.0)


# ----------------------------------------------------------------------
#  LLM CONTROLLER
# ----------------------------------------------------------------------

class LLMControllerV2:
    """Controlador LLM para geração de texto estruturado."""
    
    def __init__(self):
        self.model = None
        self.available = False
        self.persona = (
            "Você é um avatar digital vivo, com presença, intenção e continuidade psicológica. "
            "Responda de forma natural, com emoção apropriada ao contexto. "
            "Suas respostas devem ser concisas e vivas."
        )
        
    def load(self) -> None:
        """Carrega modelo LLM."""
        if ModelManager is not None:
            try:
                self.model = ModelManager(models_folder="models")
                self.available = True
            except Exception:
                self.available = False
    
    def generate(self, user_text: str, context: str, state: InternalState,
                 transformer_plan: TransformerOutput) -> str:
        """Gera texto baseado no plano do transformer."""
        
        if self.available and self.model:
            try:
                prompt = f"{self.persona}\n\nContexto: {context}\n\n"
                prompt += f"Estado emocional: {transformer_plan.emotion} (intensidade {transformer_plan.intensity:.2f})\n"
                prompt += f"Intenção: {transformer_plan.intention}\n\n"
                prompt += f"Usuário: {user_text}\n\nAvatar:"
                
                result = self.model.generate(prompt, max_tokens=80, temperature=0.8)
                return _assistant_reply_text(str(result), transformer_plan.text or 'Entendi.', 2)
            except Exception:
                pass
        
        # Fallback baseado no estado
        if transformer_plan.emotion == "happy":
            return f"Que bom! {transformer_plan.text or 'Estou feliz com isso!'}"
        elif transformer_plan.emotion == "curious":
            return f"Interessante! {transformer_plan.text or 'Me conta mais sobre isso.'}"
        elif transformer_plan.emotion == "confident":
            return transformer_plan.text or "Entendi. Vou ajudar com isso."
        elif transformer_plan.emotion == "shy":
            return transformer_plan.text or "Ah, entendi..."
        else:
            return transformer_plan.text or "Entendi. Vou refletir sobre isso."


# ----------------------------------------------------------------------
#  RENDERER
# ----------------------------------------------------------------------

class AvatarRendererV2:
    """Renderizador visual do avatar."""
    
    def __init__(self, render_res: int = 1400, display_res: int = 900):
        self.render_res = render_res
        self.display_res = display_res
        self.cx = render_res // 2
        self.cy = render_res // 2 + 45
        self.head_radius = int(render_res * 0.22)
        
    def render(self, t: float, perception: PerceptionSnapshotV2, state: InternalState,
               motion: Dict[str, Any], attention: AttentionEngineV2,
               audio: AudioSyncEngineV2, plan: Optional[TransformerOutput]) -> Image.Image:
        """Renderiza o avatar com expressões atuais."""
        
        img = Image.new("RGBA", (self.render_res, self.render_res), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        motion_meta = motion.get("motion_meta", {}) if isinstance(motion, dict) else {}
        fluidity = motion.get("fluidity", {}) if isinstance(motion, dict) else {}
        render_scale = float(motion_meta.get("body_scale", fluidity.get("scale", 1.0)))
        render_alpha = float(motion_meta.get("alpha", fluidity.get("alpha", 1.0)))
        place_x = float(motion_meta.get("placement_x", 0.0))
        place_y = float(motion_meta.get("placement_y", 0.0))

        # Corpo
        torso_cx = self.cx + motion["body_lean_x"] * 40 + place_x * 34
        torso_cy = self.cy + 255 + motion["body_lean_y"] * 24 + place_y * 26
        torso_w = 210 * render_scale
        torso_h = 245 * render_scale
        draw.ellipse((torso_cx - torso_w, torso_cy - torso_h, torso_cx + torso_w, torso_cy + torso_h),
                     fill=(18, 72, 112, int(184 * clamp(render_alpha, 0.0, 1.0))))

        # Cabeça
        head_x = self.cx + motion["head_x"] * 12 + place_x * 22
        head_y = self.cy - 15 + motion["head_y"] * 10 + place_y * 18
        radius = self.head_radius * render_scale
        
        # Gradiente da cabeça
        for i in range(58, 0, -1):
            r = radius * (i / 58) ** 0.92
            color = (20 + int(14 * (1 - i/58)),
                     88 + int(78 * (1 - i/58)),
                     180 + int(38 * (1 - i/58)))
            draw.ellipse((head_x - r, head_y - r, head_x + r, head_y + r),
                         fill=color)
        
        # Olhos
        eye_y = head_y - int(radius * 0.22)
        left_eye_x = head_x - int(radius * 0.34)
        right_eye_x = head_x + int(radius * 0.34)
        eye_h = 50 * state.arousal * motion["eye_open"]
        
        for ex in [left_eye_x, right_eye_x]:
            draw.ellipse((ex - 58, eye_y - eye_h, ex + 58, eye_y + eye_h),
                         fill=(240, 245, 255))
            
            # Íris com direção do olhar
            iris_cx = ex + attention.gaze_x * 12
            iris_cy = eye_y + attention.gaze_y * 10
            draw.ellipse((iris_cx - 30, iris_cy - 30, iris_cx + 30, iris_cy + 30),
                         fill=(12, 70, 150))
            draw.ellipse((iris_cx - 10, iris_cy - 10, iris_cx + 10, iris_cy + 10),
                         fill=(0, 0, 0))
        
        # Sobrancelhas
        brow_lift = -20 * motion["brow_left"]
        self._draw_brow(draw, left_eye_x, eye_y - 34, brow_lift, -5)
        self._draw_brow(draw, right_eye_x, eye_y - 34, -20 * motion["brow_right"], 5)
        
        # Boca
        mouth_y = head_y + 102
        mouth_w = 82
        curve = motion["mouth_curve"]
        open_amt = motion["mouth_open"]
        
        pts = []
        for i in range(-9, 10):
            x = head_x + i * 8
            u = i / 9.0
            arch = math.sin((u + 1.0) * math.pi * 0.5)
            yy = mouth_y + curve * 18 * arch - open_amt * 8 * (1 - abs(u))
            pts.append((x, yy))
        
        draw.line(pts, fill=(112, 52, 64), width=8)
        
        # Piscar
        if attention.blink_amount > 0.01:
            lid_h = eye_h * attention.blink_amount * 1.55
            for ex in [left_eye_x, right_eye_x]:
                draw.rounded_rectangle((ex - 64, eye_y - lid_h, ex + 64, eye_y + lid_h),
                                       radius=22, fill=(80, 112, 145))
        
        # Redimensionar para display
        img = img.resize((self.display_res, self.display_res), Image.Resampling.LANCZOS)
        if render_alpha < 0.999:
            mask = img.getchannel('A')
            mask = mask.point(lambda px: int(px * clamp(render_alpha, 0.0, 1.0)))
            img.putalpha(mask)
        return img
    
    def _draw_brow(self, draw: ImageDraw.Draw, bx: float, by: float, lift: float, tilt: float):
        pts = []
        for i in range(18):
            u = i / 17.0
            x = bx - 40 + 80 * u
            arch = math.sin(u * math.pi)
            y = by + lift + tilt * (u - 0.5) - arch * 12
            pts.append((x, y))
        draw.line(pts, fill=(22, 44, 70), width=10)


# ----------------------------------------------------------------------
#  STATE MANAGER
# ----------------------------------------------------------------------

class StateManager:
    """Gerencia persistência de estado entre sessões."""
    
    def __init__(self, save_path: str = "avatar_state.pkl"):
        self.save_path = save_path
        self.last_save = 0.0
        
    def save(self, memory: MemoryEngineV2, state: InternalState, t: float) -> None:
        """Salva estado atual em disco."""
        if t - self.last_save > 60.0:  # Salvar a cada minuto
            try:
                data = {
                    'memory_short': list(memory.short_term),
                    'memory_long': memory.long_term,
                    'topics': dict(memory.topics),
                    'preferences': memory.user_preferences,
                    'state': state,
                    'timestamp': t
                }
                with open(self.save_path, 'wb') as f:
                    pickle.dump(data, f)
                self.last_save = t
            except Exception:
                pass
    
    def load(self, memory: MemoryEngineV2, state: InternalState) -> bool:
        """Carrega estado anterior do disco."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                memory.short_term.extend(data.get('memory_short', []))
                memory.long_term.extend(data.get('memory_long', []))
                memory.topics.update(data.get('topics', {}))
                memory.user_preferences.update(data.get('preferences', {}))
                loaded_state = data.get('state')
                if loaded_state:
                    state.valence = loaded_state.valence
                    state.arousal = loaded_state.arousal
                    state.curiosity = loaded_state.curiosity
                    state.confidence = loaded_state.confidence
                return True
            except Exception:
                pass
        return False


# ----------------------------------------------------------------------
#  AVATAR APP - INTEGRAÇÃO PRINCIPAL
# ----------------------------------------------------------------------

class AvatarAppV2:
    """Aplicação principal do avatar."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Avatar Vivo - Transformer Brain")
        self.root.overrideredirect(True)
        self.root.configure(bg="#ffffff")
        try:
            self.root.attributes("-transparentcolor", "#ffffff")
        except Exception:
            pass
        
        self.display_res = 900
        self.root.geometry(f"{self.display_res}x{self.display_res}+120+90")
        
        self.canvas = tk.Canvas(self.root, width=self.display_res, height=self.display_res,
                                bg="#ffffff", highlightthickness=0, bd=0)
        self.canvas.pack()
        
        # Módulos
        self.perception = PerceptionEngine()
        self.memory = MemoryEngineV2()
        self.emotion = EmotionEngineV2()
        self.attention = AttentionEngineV2()
        self.behavior = BehaviorEngineV2()
        self.motion = MotionEngineV2()
        self.audio = AudioSyncEngineV2()
        self.llm = LLMControllerV2()
        self.renderer = AvatarRendererV2()
        self.state_manager = StateManager()
        
        # Transformer
        self.transformer = AvatarTransformer(
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024
        )
        
        # Estado
        self.perception_data = PerceptionSnapshotV2()
        self.plan: Optional[TransformerOutput] = None
        self.is_speaking = False
        self.start_time = time.time()
        self.last_frame_t = self.start_time
        self.input_queue: queue.Queue = queue.Queue()
        
        # UI
        self.photo = None
        self.drag = {"x": 0, "y": 0}
        self.mouse_x = self.display_res // 2
        self.mouse_y = self.display_res // 2
        
        # Carregar estado salvo
        self.state_manager.load(self.memory, self.emotion.state)
        self.llm.load()
        
        self._setup_window()
        self._bind_events()
        self._start_console_thread()
        self._setup_voice_bridge()
        
    def _setup_window(self) -> None:
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))
    
    def _setup_voice_bridge(self) -> None:
        try:
            if install_voice_input_bridge is None:
                return
            config = VoiceInputConfig(
                model_name="base",
                language="pt",
                vad_rms_threshold=0.02,
                vad_start_frames=2,
                vad_stop_frames=10,
                min_utterance_seconds=0.25,
                silence_flush_s=0.45,
                blocksize=512,
                sample_rate=16000,
                channels=1,
                device=1,
            )
            self.voice_input_bridge = install_voice_input_bridge(self, config=config)
            print("Voice bridge ativo.")
        except Exception as e:
            self.voice_input_bridge = None
            print(f"Voice bridge desativado: {e}")
    
    def _bind_events(self):
        self.root.bind("<Motion>", self._on_motion)
        self.root.bind("<Button-1>", self._on_click)
        self.root.bind("<Double-Button-1>", self._on_double_click)
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<ButtonPress-3>", self._drag_start)
        self.root.bind("<B3-Motion>", self._drag_move)
    
    def _drag_start(self, event):
        self.drag["x"] = event.x_root
        self.drag["y"] = event.y_root
    
    def _drag_move(self, event):
        dx = event.x_root - self.drag["x"]
        dy = event.y_root - self.drag["y"]
        geom = self.root.geometry()
        size_part, pos_part = geom.split("+", 1)
        w, h = map(int, size_part.split("x"))
        x, y = map(int, pos_part.split("+"))
        self.root.geometry(f"{w}x{h}+{x + dx}+{y + dy}")
        self.drag["x"] = event.x_root
        self.drag["y"] = event.y_root
    
    def _on_motion(self, event):
        self.mouse_x = clamp(event.x, 0, self.display_res)
        self.mouse_y = clamp(event.y, 0, self.display_res)
    
    def _on_click(self, _event):
        self.perception_data.click = True
    
    def _on_double_click(self, _event):
        self.perception_data.double_click = True
    
    def _on_focus_in(self, _event):
        self.perception_data.focus = True
    
    def _on_focus_out(self, _event):
        self.perception_data.focus = False
    
    def _start_console_thread(self):
        def loop():
            while True:
                try:
                    text = input("Você: ").strip()
                    if text:
                        if text.lower() in {"/quit", "/exit"}:
                            self.root.after(0, self.close)
                            break
                        self.input_queue.put(text)
                except Exception:
                    break
        threading.Thread(target=loop, daemon=True).start()
    
    def _distance_to_face(self) -> float:
        fx, fy = self.display_res // 2, self.display_res // 2 - 20
        return math.hypot(self.mouse_x - fx, self.mouse_y - fy)
    
    def _update_perception(self, t: float, dt: float, user_text: str = ""):
        click_recent = (t - self.last_click_time) < 0.18 if hasattr(self, 'last_click_time') else False
        double_recent = (t - self.last_click_time) < 0.08 if hasattr(self, 'last_click_time') else False
        
        if self.audio.voiced or self.is_speaking:
            self.silence_start = t
        silence_s = max(0.0, t - self.silence_start) if hasattr(self, 'silence_start') else 0.0
        
        self.perception_data = PerceptionSnapshotV2(
            t=t, dt=dt,
            mouse_x=self.mouse_x, mouse_y=self.mouse_y,
            mouse_distance_to_face=self._distance_to_face(),
            click=click_recent, double_click=double_recent,
            focus=self.perception_data.focus,
            silence_s=silence_s,
            audio_rms=self.audio.energy,
            audio_peak=self.audio.peak,
            audio_attack=self.audio.attack,
            audio_voiced=self.audio.voiced,
            user_text=user_text, user_text_len=len(user_text),
            speaking=self.is_speaking,
            turn_index=getattr(self, 'turn_index', 0)
        )
    # Exemplo de implementação no bloco principal ou no init da sua classe
def setup_voice(app_instance):
    config_denise = VoiceInputConfig(
        device=1,               # O ID que deu [OK] no teste
        vad_rms_threshold=0.02, # Ajustado para 0.02 para ser seguro e não pegar ruído de fundo
        language="pt",         # Força o reconhecimento em Português
        model_name="base",
        vad_start_frames=2,
        vad_stop_frames=10,
        min_utterance_seconds=0.25,
        silence_flush_s=0.45,
        blocksize=512,
        sample_rate=16000,
        channels=1,
    )
    try:
        # Esta função cria a thread, vincula ao input_queue e inicia o áudio
        voice_bridge = install_voice_input_bridge(app_instance, config=config_denise)
        print("Sistema de voz iniciado com sucesso.")
        return voice_bridge
    except Exception as e:
        print(f"Erro ao instalar Voice Bridge: {e}")
        return None
# ----------------------------------------------------------------------
#  Advanced Transformers & Micro-Expressions Integration
# ----------------------------------------------------------------------

# 1. Advanced Animation (Suavização e Gestão de Espaço Avançada)
try:
    from advanced_animation_transformers import install_advanced_animation_transformers
    install_advanced_animation_transformers(globals())
    print("Loaded: Advanced Animation Transformers")
except Exception as e:
    print(f"Skipped Advanced Animation Transformers: {e}")

# 2. Facial Micro Expressions (Pestanejo emocional, sorrisos assimétricos, sacadas)
try:
    from facial_micro_expression_transformers import install_facial_micro_expression_transformers
    install_facial_micro_expression_transformers(globals())
    print("Loaded: Facial Micro Expressions")
except Exception as e:
    print(f"Skipped Facial Micro Expressions: {e}")

# 3. Axial Rotation (Movimento esférico dos olhos e rotação da cabeça em 3D)
try:
    from axial_rotation_transformers import (
        install_axial_rotation_pipeline,
        install_axial_rotation_core,
        install_axial_rotation_viseme_hints
    )
    # Patch para a Versão 1 (AvatarApp / MotionEngine)
    install_axial_rotation_pipeline(MotionEngine, AvatarRenderer, globals())
    
    # Patch para a Versão 2 (AvatarAppV2 / MotionEngineV2)
    install_axial_rotation_pipeline(MotionEngineV2, AvatarRendererV2, globals())
    
    # Patch do core e visemas
    install_axial_rotation_core(globals())
    install_axial_rotation_viseme_hints(globals())
    print("Loaded: Axial Rotation Transformers")
except Exception as e:
    print(f"Skipped Axial Rotation Transformers: {e}")

# ======================================================================
#  Live Stream Orchestrator patch for low-RAM continuous voice
# ======================================================================

def _light_text(t: str, max_len: int = 220) -> str:
    return text_summary(t, max_len) if 'text_summary' in globals() else ' '.join((t or '').split())[:max_len]


def _limit_short_reply(text: str, max_sentences: int = 2) -> str:
    s = ' '.join((text or '').strip().split())
    if not s:
        return s
    parts = re.split(r'(?<=[.!?\n])\s+', s)
    kept = [p.strip() for p in parts if p.strip()][:max_sentences]
    if not kept:
        kept = [s[:160]]
    out = ' '.join(kept).strip()
    return out if out else s[:160]


def _extract_json_payload(text: str) -> str:
    s = ' '.join((text or '').strip().split())
    if not s:
        return s
    if s.startswith('```'):
        s = strip_json_fences(s)
    if s.startswith('{') and s.endswith('}'):
        return s
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s


def _final_assistant_text(text: str, fallback: str = '') -> str:
    raw = ' '.join((text or '').strip().split())
    if not raw:
        return fallback or ''
    raw = strip_json_fences(raw) if 'strip_json_fences' in globals() else raw

    markers = (
        '### Assistente:', '### Assistant:', 'Assistant:', 'Assistente:',
        'Avatar:', 'Response:', 'Resposta final:', 'Resposta:', 'Sistema:',
        'System:', 'Contexto:', 'Context:', 'Estado emocional:', 'Emotion:',
        'Usuário:', 'Usuario:', 'User:', 'Logs:', 'Log:', 'Prompt:'
    )
    for marker in markers:
        pos = raw.rfind(marker)
        if pos != -1:
            tail = raw[pos + len(marker):].strip()
            if tail:
                raw = tail

    cleaned_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith((
            'contexto:', 'context:', 'estado emocional:', 'emotion:',
            'usuário:', 'usuario:', 'user:', 'assistant:', 'assistente:',
            'system:', 'sistema:', 'logs:', 'log:', 'prompt:',
            'análise:', 'analise:', 'analysis:', 'reasoning:', 'thought:',
            'memória:', 'memoria:', 'json:'
        )):
            continue
        if s.startswith('```'):
            continue
        cleaned_lines.append(s)

    cleaned = ' '.join(cleaned_lines) if cleaned_lines else raw
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned.startswith('{') and cleaned.endswith('}'):
        return fallback or ''
    if cleaned.startswith('{"') or cleaned.startswith("{'"):
        return fallback or ''
    return cleaned[:420]



def _assistant_output_looks_like_leak(text: str) -> bool:
    low = ' '.join((text or '').split()).lower()
    if not low:
        return False
    suspicious = (
        'contexto atual', 'estado emocional', 'entrada do usuário', 'entrada do utilizador',
        'retorne apenas json', 'responda apenas json', 'responda somente json',
        'você é um avatar', 'voce é um avatar', 'você é uma assistente', 'voce é uma assistente',
        'assistant:', 'assistente:', 'system:', 'sistema:', 'prompt:', 'logs:', 'log:',
        'analysis:', 'reasoning:', 'thought:'
    )
    return any(marker in low for marker in suspicious)


def _assistant_reply_text(text: str, fallback: str = '', max_sentences: int = 2) -> str:
    cleaned = _limit_short_reply(_final_assistant_text(text, fallback), max_sentences)
    if not cleaned:
        return _limit_short_reply(fallback, max_sentences) if fallback else ''
    if _assistant_output_looks_like_leak(cleaned):
        return _limit_short_reply(fallback, max_sentences) if fallback else ''
    return cleaned


def _stream_chunk_sanitize(token: str) -> str:
    token = _clean_text(token) if '_clean_text' in globals() else ' '.join((token or '').split())
    if not token:
        return ''
    if _assistant_output_looks_like_leak(token):
        return ''
    token = token.replace('```', '')
    token = re.sub(r'^(?:assistant|assistente|avatar|response|resposta|system|sistema|prompt|logs?|analysis|reasoning|thought)\s*[:\-]\s*', '', token, flags=re.I)
    return token.strip()
def _audio_reset_v2(self):
    self.energy = 0.0
    self.peak = 0.0
    self.attack = 0.0
    self.voiced = False
    self.speaking = False
    self.proxy_active = False
    self.proxy_time = 0.0
    self.proxy_duration = 0.0
    self.viseme_label = 'silence'
    self.viseme_openness = 0.0
    self.viseme_rounding = 0.0
    self.viseme_closure = 1.0
    self.viseme_lip_spread = 0.0
    self.viseme_jaw = 0.0
    self.viseme_energy = 0.0
    self.viseme_stamp = 0.0


def _audio_feed_viseme_v2(self, frame):
    try:
        label = str(getattr(frame, 'label', 'silence'))
        openness = float(getattr(frame, 'openness', 0.0))
        rounding = float(getattr(frame, 'rounding', 0.0))
        closure = float(getattr(frame, 'closure', 1.0))
        lip_spread = float(getattr(frame, 'lip_spread', 0.0))
        jaw = float(getattr(frame, 'jaw', openness))
        energy = float(getattr(frame, 'energy', openness))
        self.viseme_label = label
        self.viseme_openness = clamp(openness, 0.0, 1.0)
        self.viseme_rounding = clamp(rounding, 0.0, 1.0)
        self.viseme_closure = clamp(closure, 0.0, 1.0)
        self.viseme_lip_spread = clamp(lip_spread, 0.0, 1.0)
        self.viseme_jaw = clamp(jaw, 0.0, 1.0)
        self.viseme_energy = clamp(energy, 0.0, 1.0)
        self.viseme_stamp = float(getattr(frame, 'timestamp', time.time()))
    except Exception:
        pass


def _audio_mouth_open_v2(self, state, plan):
    viseme_open = float(getattr(self, 'viseme_openness', 0.0))
    viseme_close = float(getattr(self, 'viseme_closure', 1.0))
    viseme_round = float(getattr(self, 'viseme_rounding', 0.0))
    viseme_jaw = float(getattr(self, 'viseme_jaw', 0.0))
    if self.speaking or self.proxy_active:
        base = (self.energy * 0.45) + (self.attack * 0.20) + (viseme_open * 0.30) + (viseme_jaw * 0.15)
        base *= (1.0 - 0.22 * viseme_close)
        if viseme_round > 0.5:
            base *= 0.92
        base = clamp(base, 0.03, 0.90)
    else:
        base = 0.02 + state.energy * 0.05 + viseme_open * 0.04
    if plan and getattr(plan, 'state_transition', None) == 'speaking':
        base *= 1.08
    return clamp(base, 0.0, 1.0)


def _tts_patch_config_v2(self):
    return dict(
        models_folder=getattr(self, 'models_folder', None),
        default_speaker='serena',
        sample_rate=24000,
        chunk_seconds=0.20,
        max_chars_per_chunk=96,
        max_chars_per_sentence=160,
        max_threads=1,
        preload=True,
        warmup=False,
        allow_cuda=False,
        low_memory_mode=True,
        keep_model_ready=True,
        language='portuguese',
        pre_roll_sentences=2,
        start_after_sentences=1,
        silence_fallback_s=0.10,
    )


def _maybe_load_tts_v2(self):
    self.tts = None
    if LocalTTSManager is None:
        return
    try:
        cfg = _tts_patch_config_v2(self)
        self.tts = LocalTTSManager(**cfg)
        try:
            if hasattr(self.tts, 'register_audio_listener'):
                self.tts.register_audio_listener(self.audio.feed)
        except Exception:
            pass
        try:
            if hasattr(self.tts, 'register_viseme_listener'):
                self.tts.register_viseme_listener(self.audio.feed_viseme)
        except Exception:
            pass
    except Exception:
        self.tts = None


def _build_short_plan_prompt_v2(self, user_text: str, context: str, state, transformer_plan):
    prompt = f"{self.persona}\n\n"
    prompt += f"Contexto: {_light_text(context, 220)}\n\n"
    prompt += f"Estado emocional: {getattr(transformer_plan, 'emotion', 'neutral')} (intensidade {float(getattr(transformer_plan, 'intensity', 0.35)):.2f})\n"
    prompt += f"Intenção: {getattr(transformer_plan, 'intention', 'respond')}\n\n"
    prompt += f"Usuário: {_light_text(user_text, 160)}\n\n"
    prompt += "Responda em 1 ou 2 frases curtas, claras e naturais.\nAvatar:"
    return prompt


def _llm_generate_v2(self, user_text: str, context: str, state, transformer_plan):
    if self.available and self.model:
        try:
            prompt = _build_short_plan_prompt_v2(self, user_text, context, state, transformer_plan)
            result = self.model.generate(prompt, max_tokens=48, temperature=0.55)
            return _limit_short_reply(_final_assistant_text(str(result), getattr(transformer_plan, 'text', '') or 'Entendi.'), 2)
        except Exception:
            pass
    if getattr(transformer_plan, 'emotion', 'neutral') == 'happy':
        return _limit_short_reply(f"Que bom! {getattr(transformer_plan, 'text', '') or 'Estou feliz com isso.'}", 2)
    if getattr(transformer_plan, 'emotion', 'neutral') == 'curious':
        return _limit_short_reply(f"Interessante. {getattr(transformer_plan, 'text', '') or 'Me conta mais.'}", 2)
    if getattr(transformer_plan, 'emotion', 'neutral') == 'confident':
        return _limit_short_reply(getattr(transformer_plan, 'text', '') or 'Entendi. Vou ajudar com isso.', 2)
    if getattr(transformer_plan, 'emotion', 'neutral') == 'shy':
        return _limit_short_reply(getattr(transformer_plan, 'text', '') or 'Ah, entendi...', 2)
    return _limit_short_reply(getattr(transformer_plan, 'text', '') or 'Entendi. Vou refletir sobre isso.', 2)


def _build_plan_lowram_v2(self, user_text: str, memory, emotion, perception):
    fallback = self._fallback(user_text, memory, emotion, perception)
    if not self.available or self.model is None:
        return fallback
    try:
        context = memory.brief_context() if hasattr(memory, 'brief_context') else memory.context_digest()
        prompt = (
            f"{self.persona}\n\n"
            f"Contexto atual:\n{_light_text(context, 220)}\n\n"
            f"Estado emocional:\n{json.dumps(asdict(emotion), ensure_ascii=False)}\n\n"
            f"Entrada do usuário:\n{_light_text(user_text, 160)}\n\n"
            f"Responda com JSON curto."
        )
        messages = [
            {"role": "system", "content": self.persona},
            {"role": "user", "content": prompt},
        ]
        result = self.model.chat(messages, max_tokens=128, temperature=0.55)
        if isinstance(result, dict) and 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content']
        else:
            content = str(result)
        plan = self._parse_plan(content, fallback)
        plan.text = _assistant_reply_text(plan.text, fallback.text, 2) or fallback.text
        self.history.append({'user': user_text, 'assistant': plan.text})
        return plan
    except Exception:
        return fallback


try:
    AudioSyncEngineV2.reset = _audio_reset_v2  # type: ignore[assignment]
    AudioSyncEngineV2.feed_viseme = _audio_feed_viseme_v2  # type: ignore[assignment]
    AudioSyncEngineV2.mouth_open = _audio_mouth_open_v2  # type: ignore[assignment]
    AvatarAppV2._maybe_load_tts = _maybe_load_tts_v2  # type: ignore[assignment]
    LLMControllerV2.generate = _llm_generate_v2  # type: ignore[assignment]
    LLMControllerV2.build_plan = _build_plan_lowram_v2  # type: ignore[assignment]
except Exception:
    pass

# ==============================================================================
# Voice output guarantee patch
# ==============================================================================

_AVATAR_TTS_PATCH_MARKER = True


def _avatar_tts_cfg(self) -> dict:
    return dict(
        models_folder=getattr(self, 'models_folder', None),
        default_speaker='serena',
        sample_rate=16000,
        chunk_seconds=0.20,
        max_chars_per_chunk=96,
        max_chars_per_sentence=160,
        max_threads=1,
        preload=True,
        warmup=False,
        allow_cuda=False,
        low_memory_mode=True,
        keep_model_ready=True,
        prefer_int8_cpu=True,
        language='portuguese',
        pre_roll_sentences=2,
        start_after_sentences=1,
        silence_fallback_s=0.10,
    )


def _avatar_maybe_load_tts(self) -> None:
    self.tts = None
    if LocalTTSManager is None:
        return
    try:
        self.tts = LocalTTSManager(**_avatar_tts_cfg(self))
        if hasattr(self.tts, 'register_audio_listener'):
            self.tts.register_audio_listener(getattr(self.audio, 'feed', None))
        if hasattr(self.tts, 'register_viseme_listener') and hasattr(self.audio, 'feed_viseme'):
            self.tts.register_viseme_listener(getattr(self.audio, 'feed_viseme', None))
        if hasattr(self.tts, 'register_echo_reference_sink') and hasattr(self.audio, 'feed_playback_chunk'):
            self.tts.register_echo_reference_sink(getattr(self.audio, 'feed_playback_chunk', None))
    except Exception:
        logger.exception('Falha ao carregar o TTS')
        self.tts = None


def _avatar_begin_response(self, user_text: str) -> None:
    user_text = ' '.join((user_text or '').strip().split())
    if not user_text:
        return
    now = time.time()
    t = now - self.start_time
    self.turn_index += 1
    self.last_activity_t = now
    self.speech_cancel.clear()
    self.memory.remember_turn('user', user_text, emotion=self.emotion.state.emotion, state=self.behavior.current_state, t=t)
    self.plan = self.llm.build_plan(user_text, self.memory, self.emotion.state, self.perception)
    self.memory.remember_plan(self.plan, t=t)
    self.behavior.force_target_state = self.plan.state
    self.is_speaking = True
    self.audio.speaking = True
    self.audio.reset()
    text = _assistant_reply_text(getattr(self.plan, 'text', ''), 'Entendi.', 2) or 'Entendi.'
    self.plan.text = text
    duration = max(1.2, len(text) * 0.05 + getattr(self.plan, 'pause_ms', 120) / 1000.0)
    self.audio.start_proxy(text, duration)
    gen = None
    if self.tts is not None:
        try:
            if hasattr(self.tts, 'stream_synthesize'):
                gen = self.tts.stream_synthesize(text, voice=getattr(getattr(self.tts, 'cfg', None), 'default_speaker', None))
            else:
                gen = self.tts.synthesize(text)
        except Exception:
            gen = None
    if gen is not None:
        threading.Thread(target=self._play_audio_stream, args=(gen,), daemon=True, name='AvatarSpeech').start()
    else:
        self.root.after(int(max(duration * 1000, 1600)), self._stop_speaking)


def _avatar_play_audio_stream(self, audio_gen) -> None:
    try:
        tts = getattr(self, 'tts', None)
        if tts is not None and hasattr(tts, 'play_stream'):
            tts.play_stream(audio_gen, sample_rate=getattr(tts, 'sample_rate', 24000), stop_event=self.speech_cancel)
        else:
            for chunk in audio_gen:
                if self.speech_cancel.is_set():
                    break
                if chunk is None:
                    continue
                data = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if data.size == 0:
                    continue
                self.audio.feed(data)
                if sd is not None:
                    try:
                        sd.play(data, samplerate=24000)
                        sd.wait()
                    except Exception:
                        break
    except Exception:
        logger.exception('Falha na saída de voz')
    finally:
        if not self.speech_cancel.is_set():
            try:
                self.root.after(0, self._stop_speaking)
            except Exception:
                self._stop_speaking()


def _avatar_stop_speaking(self) -> None:
    self.speech_cancel.set()
    try:
        if sd is not None:
            sd.stop()
    except Exception:
        pass
    self.is_speaking = False
    self.audio.speaking = False
    self.audio.proxy_active = False
    if hasattr(self.audio, 'reset'):
        self.audio.reset()


try:
    AvatarApp._maybe_load_tts = _avatar_maybe_load_tts  # type: ignore[assignment]
    AvatarApp._begin_response = _avatar_begin_response  # type: ignore[assignment]
    AvatarApp._play_audio_stream = _avatar_play_audio_stream  # type: ignore[assignment]
    AvatarApp._stop_speaking = _avatar_stop_speaking  # type: ignore[assignment]
except Exception:
    pass

try:
    AvatarAppV2._maybe_load_tts = _maybe_load_tts_v2  # type: ignore[assignment]
except Exception:
    pass

if __name__ == "__main__":
    main()
