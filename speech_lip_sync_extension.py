from __future__ import annotations

"""
speech_lip_sync_extension.py

Camada adicional para TTS + sincronização labial, sem alterar o runtime existente.

Integra:
- buffer de frases completas
- streaming contínuo de áudio
- listener de áudio via callback/hook
- geração heurística de visemas em português
- atualização suave de boca/face apenas por interface
- compatível com TTSManager / LocalTTSManager já existentes
"""

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # type: ignore

try:
    from tts_local import LocalTTSManager, TTSManager, TTSConfig  # type: ignore
except Exception:
    LocalTTSManager = None  # type: ignore
    TTSManager = None  # type: ignore
    TTSConfig = None  # type: ignore


_SENTENCE_END_RE = re.compile(r"(?<=[.!?\n])\s+")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


def clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def split_sentences(text: str) -> List[str]:
    t = clean_text(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENTENCE_END_RE.split(t) if p.strip()]
    if not parts:
        return [t if t[-1] in ".!?\n" else t + "."]
    return [p if p[-1] in ".!?\n" else p + "." for p in parts]


def extract_complete_sentences(text: str) -> Tuple[List[str], str]:
    t = clean_text(text)
    if not t:
        return [], ""
    complete: List[str] = []
    start = 0
    i = 0
    while i < len(t):
        if t[i] in ".!?\n":
            sent = clean_text(t[start:i + 1])
            if sent:
                complete.append(sent)
            i += 1
            while i < len(t) and t[i].isspace():
                i += 1
            start = i
            continue
        i += 1
    return complete, clean_text(t[start:])


def safe_audio_array(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(x, -1.0, 1.0)


def audio_rms(audio: np.ndarray) -> float:
    x = safe_audio_array(audio)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def audio_peak(audio: np.ndarray) -> float:
    x = safe_audio_array(audio)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


@dataclass(slots=True)
class VisemeFrame:
    t: float
    symbol: str
    intensity: float
    mouth_open: float
    mouth_width: float
    mouth_round: float
    jaw_drop: float
    lip_tension: float
    lip_corner_up: float
    lip_corner_down: float
    lips_together: float
    tongue_out: float = 0.0
    upper_lip_raise: float = 0.0
    lower_lip_depress: float = 0.0
    mouth_corner_stretch: float = 0.0
    blend: float = 0.0
    source_text: str = ""

    def as_face_update(self) -> Dict[str, float]:
        return {
            "mouth_open": float(self.mouth_open),
            "mouth_width": float(self.mouth_width),
            "mouth_round": float(self.mouth_round),
            "jaw_drop": float(self.jaw_drop),
            "lip_tension": float(self.lip_tension),
            "lip_corner_up": float(self.lip_corner_up),
            "lip_corner_down": float(self.lip_corner_down),
            "lips_together": float(self.lips_together),
            "tongue_out": float(self.tongue_out),
            "upper_lip_raise": float(self.upper_lip_raise),
            "lower_lip_depress": float(self.lower_lip_depress),
            "mouth_corner_stretch": float(self.mouth_corner_stretch),
        }


@dataclass(slots=True)
class SpeechLipSyncConfig:
    sample_rate: int = 24000
    chunk_seconds: float = 0.22
    max_threads: int = 1
    preload: bool = False
    warmup: bool = False
    allow_cuda: bool = True
    low_memory_mode: bool = True
    language: str = "pt"
    default_speaker: str = "serena"

    pre_roll_sentences: int = 2
    start_after_sentences: int = 1
    max_chars_per_chunk: int = 120
    max_chars_per_sentence: int = 220
    silence_fallback_s: float = 0.10

    playback_enabled: bool = True
    update_face: bool = True
    update_audio_engine: bool = True
    viseme_smoothing: float = 0.18
    viseme_min_hold_s: float = 0.06
    viseme_listener_queue_size: int = 32
    audio_listener_queue_size: int = 32


class PortugueseVisemeMapper:
    """
    Heurística leve para visemas em português.
    Não depende de ASR fonético e funciona diretamente com texto + energia de áudio.
    """

    VOWEL_GROUPS: Dict[str, str] = {
        "a": "A",
        "á": "A",
        "ã": "A_NASAL",
        "â": "A",
        "e": "E",
        "é": "E",
        "ê": "E",
        "i": "I",
        "í": "I",
        "o": "O",
        "ó": "O",
        "ô": "O",
        "õ": "O_NASAL",
        "u": "U",
        "ú": "U",
    }

    CLOSED_LIP = {"p", "b", "m"}
    DENTAL = {"f", "v"}
    ALVEOLAR = {"t", "d", "n", "l", "r", "s", "z", "ç", "x"}
    PALATAL = {"j", "g", "lh", "nh"}
    ROUNDED = {"o", "ó", "ô", "u", "ú", "õ"}

    def _classify_word(self, word: str) -> str:
        w = word.lower()
        if not w:
            return "SIL"
        if any(seq in w for seq in ("ão", "ãe", "õe", "am", "em", "im", "om", "um")):
            return "NASAL"
        if any(ch in w for ch in ("o", "ó", "ô", "u", "ú", "õ")):
            return "ROUND"
        if any(ch in w for ch in ("a", "á", "â")):
            return "OPEN"
        if any(ch in w for ch in ("e", "é", "ê", "i", "í")):
            return "FRONT"
        if any(ch in w for ch in ("p", "b", "m")):
            return "CLOSED"
        if any(ch in w for ch in ("f", "v")):
            return "TEETH"
        if any(ch in w for ch in ("t", "d", "n", "l", "r", "s", "z", "ç", "x")):
            return "MID"
        return "MID"

    def _base_frame(self, symbol: str, intensity: float, text: str, t: float) -> VisemeFrame:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        open_map = {
            "SIL": 0.02,
            "CLOSED": 0.05,
            "TEETH": 0.14,
            "MID": 0.18,
            "FRONT": 0.24,
            "ROUND": 0.30,
            "OPEN": 0.42,
            "NASAL": 0.22,
            "ROUND_NASAL": 0.28,
        }
        width_map = {
            "SIL": 0.92,
            "CLOSED": 0.86,
            "TEETH": 0.98,
            "MID": 1.00,
            "FRONT": 1.02,
            "ROUND": 0.92,
            "OPEN": 1.04,
            "NASAL": 0.95,
            "ROUND_NASAL": 0.92,
        }
        round_map = {
            "SIL": 0.00,
            "CLOSED": 0.02,
            "TEETH": 0.02,
            "MID": 0.04,
            "FRONT": 0.05,
            "ROUND": 0.24,
            "OPEN": 0.08,
            "NASAL": 0.10,
            "ROUND_NASAL": 0.20,
        }
        lip_up_map = {
            "SIL": 0.00,
            "CLOSED": 0.02,
            "TEETH": 0.05,
            "MID": 0.03,
            "FRONT": 0.03,
            "ROUND": 0.02,
            "OPEN": 0.04,
            "NASAL": 0.03,
            "ROUND_NASAL": 0.03,
        }

        mouth_open = open_map.get(symbol, 0.18) + 0.22 * intensity
        mouth_width = width_map.get(symbol, 1.0)
        mouth_round = round_map.get(symbol, 0.04)
        lip_corner_up = 0.04 * intensity
        lip_corner_down = 0.0
        lips_together = 0.0
        jaw_drop = mouth_open * 0.82
        lip_tension = 0.08 + 0.08 * intensity
        upper_lip_raise = lip_up_map.get(symbol, 0.03)
        lower_lip_depress = 0.02 * intensity
        mouth_corner_stretch = 0.03 * intensity
        tongue_out = 0.0

        if symbol == "SIL":
            lips_together = 0.85
            lip_tension = 0.02
            mouth_open = 0.03
            jaw_drop = 0.01
        elif symbol == "CLOSED":
            lips_together = 0.98
            mouth_open = 0.02
            jaw_drop = 0.0
            lip_tension = 0.12
        elif symbol == "TEETH":
            mouth_open = max(mouth_open, 0.12)
            lip_tension = 0.16
            upper_lip_raise = 0.08
            lower_lip_depress = 0.03
        elif symbol == "ROUND":
            mouth_round = 0.30
            mouth_open = max(mouth_open, 0.24)
        elif symbol == "OPEN":
            mouth_open = max(mouth_open, 0.36)
            jaw_drop = max(jaw_drop, 0.28)
        elif symbol == "NASAL":
            lips_together = 0.22
            mouth_round *= 0.85
            lip_tension = 0.12
        elif symbol == "FRONT":
            mouth_width = 1.04
            mouth_round = 0.05

        return VisemeFrame(
            t=t,
            symbol=symbol,
            intensity=intensity,
            mouth_open=float(np.clip(mouth_open, 0.0, 1.0)),
            mouth_width=float(np.clip(mouth_width, 0.7, 1.25)),
            mouth_round=float(np.clip(mouth_round, 0.0, 0.65)),
            jaw_drop=float(np.clip(jaw_drop, 0.0, 1.0)),
            lip_tension=float(np.clip(lip_tension, 0.0, 1.0)),
            lip_corner_up=float(np.clip(lip_corner_up, 0.0, 1.0)),
            lip_corner_down=float(np.clip(lip_corner_down, 0.0, 1.0)),
            lips_together=float(np.clip(lips_together, 0.0, 1.0)),
            tongue_out=float(np.clip(tongue_out, 0.0, 1.0)),
            upper_lip_raise=float(np.clip(upper_lip_raise, 0.0, 1.0)),
            lower_lip_depress=float(np.clip(lower_lip_depress, 0.0, 1.0)),
            mouth_corner_stretch=float(np.clip(mouth_corner_stretch, 0.0, 1.0)),
            blend=1.0,
            source_text=text,
        )

    def viseme_for_word(self, word: str, intensity: float = 0.5, t: float = 0.0) -> VisemeFrame:
        symbol = self._classify_word(word)
        if symbol == "ROUND":
            # refina em nasalização
            if any(seq in word.lower() for seq in ("ão", "am", "em", "im", "om", "um")):
                symbol = "ROUND_NASAL"
        return self._base_frame(symbol, intensity, word, t)

    def sequence_for_text(self, text: str, energy: float = 0.5, t0: float = 0.0) -> List[VisemeFrame]:
        words = _WORD_RE.findall(text or "")
        if not words:
            return [self._base_frame("SIL", energy, text, t0)]

        frames: List[VisemeFrame] = []
        dt = max(0.06, min(0.14, 0.10 - 0.03 * float(np.clip(energy, 0.0, 1.0))))
        t = t0
        for word in words:
            frames.append(self.viseme_for_word(word, intensity=energy, t=t))
            t += dt
        return frames


class SpeechSentenceBuffer:
    """
    Garante que apenas frases completas sejam emitidas para o TTS.
    Mantém um buffer de pré-rolagem para não falar texto incompleto.
    """

    def __init__(self, pre_roll_sentences: int = 2, start_after_sentences: int = 1):
        self.pre_roll_sentences = max(1, int(pre_roll_sentences))
        self.start_after_sentences = max(1, int(start_after_sentences))
        self.buffer = ""
        self.complete: List[str] = []
        self.started = False

    def push(self, text: str) -> List[str]:
        out: List[str] = []
        text = clean_text(text)
        if not text:
            return out

        if self.buffer:
            self.buffer = f"{self.buffer} {text}".strip()
        else:
            self.buffer = text

        complete, carry = extract_complete_sentences(self.buffer)
        self.complete.extend(complete)
        self.buffer = carry

        if not self.started and len(self.complete) >= self.start_after_sentences:
            self.started = True

        if self.started:
            while self.complete and len(out) < self.pre_roll_sentences:
                out.append(self.complete.pop(0))
        return out

    def flush(self) -> List[str]:
        out = [s for s in self.complete if s.strip()]
        self.complete.clear()
        tail = clean_text(self.buffer)
        self.buffer = ""
        if tail:
            out.extend(split_sentences(tail))
        return [s for s in out if s.strip()]


class SmoothVisemeState:
    """
    Mantém o estado da boca suave entre frames, sem mexer em _draw.
    """

    def __init__(self, smoothing: float = 0.18):
        self.smoothing = float(np.clip(smoothing, 0.02, 0.6))
        self.current: Dict[str, float] = {
            "mouth_open": 0.0,
            "mouth_width": 1.0,
            "mouth_round": 0.0,
            "jaw_drop": 0.0,
            "lip_tension": 0.0,
            "lip_corner_up": 0.0,
            "lip_corner_down": 0.0,
            "lips_together": 0.0,
            "tongue_out": 0.0,
            "upper_lip_raise": 0.0,
            "lower_lip_depress": 0.0,
            "mouth_corner_stretch": 0.0,
        }
        self.last_update = 0.0

    def reset(self) -> None:
        for k in self.current:
            self.current[k] = 0.0 if k != "mouth_width" else 1.0
        self.last_update = 0.0

    def step(self, target: Dict[str, float], dt: float) -> Dict[str, float]:
        alpha = 1.0 - float(np.exp(-max(1e-4, dt) / max(1e-4, self.smoothing)))
        for key, value in target.items():
            if key not in self.current:
                continue
            cur = self.current[key]
            self.current[key] = cur + (float(value) - cur) * alpha
        self.last_update += dt
        return dict(self.current)


class SpeechLipSyncPipeline:
    """
    Orquestra:
    texto -> buffer de frases -> TTS -> chunks de áudio -> visemas -> avatar
    """

    def __init__(
        self,
        tts: Any,
        *,
        config: Optional[SpeechLipSyncConfig] = None,
        audio_listener: Optional[Callable[[np.ndarray], None]] = None,
        viseme_listener: Optional[Callable[[VisemeFrame], None]] = None,
        playback_sink: Optional[Callable[[np.ndarray], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
    ):
        self.tts = tts
        self.config = config or SpeechLipSyncConfig()
        self.audio_listener = audio_listener
        self.viseme_listener = viseme_listener
        self.playback_sink = playback_sink
        self.on_complete = on_complete
        self._stop = threading.Event()
        self._text_queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=128)
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
        self._viseme_queue: "queue.Queue[VisemeFrame]" = queue.Queue(maxsize=self.config.viseme_listener_queue_size)
        self._worker = threading.Thread(target=self._run, daemon=True, name="SpeechLipSyncPipeline")
        self._sentence_buffer = SpeechSentenceBuffer(
            pre_roll_sentences=self.config.pre_roll_sentences,
            start_after_sentences=self.config.start_after_sentences,
        )
        self._mapper = PortugueseVisemeMapper()
        self._viseme_state = SmoothVisemeState(smoothing=self.config.viseme_smoothing)
        self._active_sentence: str = ""
        self._active_sentence_started = 0.0
        self._lock = threading.RLock()

        if hasattr(self.tts, "register_audio_listener"):
            try:
                self.tts.register_audio_listener(self._on_tts_audio_chunk)
            except Exception:
                pass
        if hasattr(self.tts, "register_echo_reference_sink"):
            try:
                self.tts.register_echo_reference_sink(self._on_tts_audio_chunk)
            except Exception:
                pass

    def start(self) -> None:
        if not self._worker.is_alive():
            self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._text_queue.put_nowait(None)
        except Exception:
            pass
        try:
            if hasattr(self.tts, "stop"):
                self.tts.stop()
        except Exception:
            pass
        try:
            if callable(self.on_complete):
                self.on_complete()
        except Exception:
            pass

    def enqueue_text(self, text: str) -> None:
        text = clean_text(text)
        if not text:
            return
        try:
            self._text_queue.put_nowait(text)
        except queue.Full:
            try:
                _ = self._text_queue.get_nowait()
                self._text_queue.put_nowait(text)
            except Exception:
                pass

    def clear(self) -> None:
        self._viseme_state.reset()
        self._active_sentence = ""
        self._active_sentence_started = 0.0

    def _emit_viseme(self, frame: VisemeFrame) -> None:
        try:
            self._viseme_queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._viseme_queue.get_nowait()
                self._viseme_queue.put_nowait(frame)
            except Exception:
                pass
        if self.viseme_listener is not None:
            try:
                self.viseme_listener(frame)
            except Exception:
                pass

    def _on_tts_audio_chunk(self, chunk: np.ndarray) -> None:
        try:
            data = safe_audio_array(chunk)
            if data.size == 0:
                return
            if self.audio_listener is not None:
                self.audio_listener(data)
            try:
                self._audio_queue.put_nowait(data)
            except queue.Full:
                try:
                    _ = self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(data)
                except Exception:
                    pass
            # Se houver um avatar com audio engine, ele pode ser atualizado pelo adaptador externo.
        except Exception:
            pass

    def _apply_sentence_visemes(self, sentence: str, chunk_iter: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        frames = self._mapper.sequence_for_text(sentence, energy=0.55, t0=time.time())
        frame_iter = iter(frames)
        current_frame = next(frame_iter, None)
        last_emit_t = 0.0

        for chunk in chunk_iter:
            if self._stop.is_set():
                break
            data = safe_audio_array(chunk)
            if data.size == 0:
                continue

            now = time.time()
            rms = audio_rms(data)
            peak = audio_peak(data)
            energy = float(np.clip(max(rms * 2.8, peak * 1.2), 0.0, 1.0))

            # Atualiza frame alvo
            if current_frame is None:
                current_frame = self._mapper.viseme_for_word(sentence, intensity=energy, t=now)

            target = current_frame.as_face_update()
            target["mouth_open"] = float(np.clip(max(target["mouth_open"], 0.06 + energy * 0.82), 0.0, 1.0))
            target["jaw_drop"] = float(np.clip(max(target["jaw_drop"], 0.05 + energy * 0.70), 0.0, 1.0))
            target["mouth_round"] = float(np.clip(target["mouth_round"] * (0.85 + 0.25 * energy), 0.0, 1.0))
            target["lips_together"] = float(np.clip(target["lips_together"] * (1.0 - 0.75 * energy), 0.0, 1.0))

            smooth = self._viseme_state.step(target, dt=max(0.016, now - last_emit_t if last_emit_t else 0.022))
            last_emit_t = now

            viseme = VisemeFrame(
                t=now,
                symbol=current_frame.symbol,
                intensity=energy,
                mouth_open=smooth["mouth_open"],
                mouth_width=smooth["mouth_width"],
                mouth_round=smooth["mouth_round"],
                jaw_drop=smooth["jaw_drop"],
                lip_tension=smooth["lip_tension"],
                lip_corner_up=smooth["lip_corner_up"],
                lip_corner_down=smooth["lip_corner_down"],
                lips_together=smooth["lips_together"],
                tongue_out=smooth["tongue_out"],
                upper_lip_raise=smooth["upper_lip_raise"],
                lower_lip_depress=smooth["lower_lip_depress"],
                mouth_corner_stretch=smooth["mouth_corner_stretch"],
                blend=1.0,
                source_text=sentence,
            )
            self._emit_viseme(viseme)
            yield data

            # Troca de visema baseada em palavras, mas sem cortar o fluxo.
            if current_frame is not None:
                current_frame = next(frame_iter, current_frame)

    def _tts_stream_sentence(self, sentence: str) -> Iterator[np.ndarray]:
        if hasattr(self.tts, "stream_synthesize"):
            yield from self.tts.stream_synthesize(sentence, voice=self.config.default_speaker)
            return
        if hasattr(self.tts, "synthesize"):
            yield from self.tts.synthesize(sentence, voice=self.config.default_speaker)
            return
        raise RuntimeError("TTS atual não expõe stream_synthesize/synthesize")

    def _play_chunk(self, chunk: np.ndarray) -> None:
        if not self.config.playback_enabled:
            return
        if self.playback_sink is not None:
            self.playback_sink(chunk)
            return
        if sd is None:
            return
        try:
            sd.play(chunk, samplerate=self.config.sample_rate)
            sd.wait()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._text_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break

            item = clean_text(item)
            if not item:
                continue

            for sentence in self._sentence_buffer.push(item):
                if self._stop.is_set():
                    break
                self._active_sentence = sentence
                self._active_sentence_started = time.time()
                try:
                    stream = self._tts_stream_sentence(sentence)
                except Exception:
                    continue
                for chunk in self._apply_sentence_visemes(sentence, stream):
                    if self._stop.is_set():
                        break
                    self._play_chunk(chunk)

        # Flush final
        for sentence in self._sentence_buffer.flush():
            if self._stop.is_set():
                break
            try:
                stream = self._tts_stream_sentence(sentence)
            except Exception:
                continue
            for chunk in self._apply_sentence_visemes(sentence, stream):
                if self._stop.is_set():
                    break
                self._play_chunk(chunk)
        try:
            if not self._stop.is_set() and callable(self.on_complete):
                self.on_complete()
        except Exception:
            pass

    def poll_viseme(self, timeout: float = 0.0) -> Optional[VisemeFrame]:
        try:
            return self._viseme_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def poll_audio(self, timeout: float = 0.0) -> Optional[np.ndarray]:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class AvatarSpeechAdapter:
    """
    Liga o pipeline ao avatar apenas por interface, sem tocar no _draw.
    Atualiza:
    - audio engine (feed / speaking / proxy)
    - face pose (mouth_open, mouth_round, etc.) se existir
    """

    def __init__(
        self,
        app: Any,
        *,
        config: Optional[SpeechLipSyncConfig] = None,
    ):
        self.app = app
        self.config = config or SpeechLipSyncConfig()
        self._lock = threading.RLock()
        self._last_face_update = 0.0
        self._last_audio_update = 0.0

    def feed_audio(self, chunk: np.ndarray) -> None:
        if not self.config.update_audio_engine:
            return
        try:
            audio = getattr(self.app, "audio", None)
            if audio is not None and hasattr(audio, "feed"):
                audio.feed(chunk)
                self._last_audio_update = time.time()
        except Exception:
            pass

    def set_speaking(self, speaking: bool) -> None:
        try:
            if hasattr(self.app, "is_speaking"):
                self.app.is_speaking = bool(speaking)
        except Exception:
            pass
        try:
            audio = getattr(self.app, "audio", None)
            if audio is not None and hasattr(audio, "speaking"):
                audio.speaking = bool(speaking)
        except Exception:
            pass

    def start_proxy(self, text: str, duration: float = 2.0) -> None:
        try:
            audio = getattr(self.app, "audio", None)
            if audio is not None and hasattr(audio, "start_proxy"):
                audio.start_proxy(text, duration)
        except Exception:
            pass

    def reset_audio(self) -> None:
        try:
            audio = getattr(self.app, "audio", None)
            if audio is not None and hasattr(audio, "reset"):
                audio.reset()
        except Exception:
            pass

    def apply_viseme(self, frame: VisemeFrame) -> None:
        if not self.config.update_face:
            return
        with self._lock:
            self._last_face_update = time.time()

            candidates = [
                getattr(self.app, "motion", None),
                getattr(self.app, "face", None),
                getattr(self.app, "avatar_face", None),
            ]
            for obj in candidates:
                if obj is None:
                    continue
                face = getattr(obj, "face", obj)
                try:
                    current = getattr(face, "mouth_open", 0.0)
                    face.mouth_open = float(current + (frame.mouth_open - current) * 0.55)
                except Exception:
                    pass
                try:
                    current = getattr(face, "mouth_width", 1.0)
                    face.mouth_width = float(current + (frame.mouth_width - current) * 0.45)
                except Exception:
                    pass
                try:
                    current = getattr(face, "mouth_round", 0.0)
                    face.mouth_round = float(current + (frame.mouth_round - current) * 0.48)
                except Exception:
                    pass
                try:
                    current = getattr(face, "jaw_drop", 0.0)
                    face.jaw_drop = float(current + (frame.jaw_drop - current) * 0.50)
                except Exception:
                    pass
                try:
                    current = getattr(face, "lip_tension", 0.0)
                    face.lip_tension = float(current + (frame.lip_tension - current) * 0.42)
                except Exception:
                    pass
                try:
                    current = getattr(face, "lip_corner_up", 0.0)
                    face.lip_corner_up = float(current + (frame.lip_corner_up - current) * 0.42)
                except Exception:
                    pass
                try:
                    current = getattr(face, "lip_corner_down", 0.0)
                    face.lip_corner_down = float(current + (frame.lip_corner_down - current) * 0.42)
                except Exception:
                    pass
                try:
                    current = getattr(face, "lips_together", 0.0)
                    face.lips_together = float(current + (frame.lips_together - current) * 0.45)
                except Exception:
                    pass
                try:
                    current = getattr(face, "upper_lip_raise", 0.0)
                    face.upper_lip_raise = float(current + (frame.upper_lip_raise - current) * 0.40)
                except Exception:
                    pass
                try:
                    current = getattr(face, "lower_lip_depress", 0.0)
                    face.lower_lip_depress = float(current + (frame.lower_lip_depress - current) * 0.40)
                except Exception:
                    pass
                try:
                    current = getattr(face, "mouth_corner_stretch", 0.0)
                    face.mouth_corner_stretch = float(current + (frame.mouth_corner_stretch - current) * 0.40)
                except Exception:
                    pass

    def bind(self, pipeline: SpeechLipSyncPipeline) -> None:
        pipeline.audio_listener = self.feed_audio
        pipeline.viseme_listener = self.apply_viseme

    def speak_text(self, pipeline: SpeechLipSyncPipeline, text: str, *, proxy_duration: float = 2.0) -> None:
        text = clean_text(text)
        if not text:
            return
        self.set_speaking(True)
        self.start_proxy(text, duration=proxy_duration)
        pipeline.enqueue_text(text)

    def stop(self) -> None:
        self.set_speaking(False)
        self.reset_audio()


def create_local_tts_manager_safe(**kwargs: Any) -> Any:
    """
    Cria um TTS com perfil leve para máquinas pequenas.
    Não altera o TTS existente; apenas fornece defaults seguros.
    """
    merged = dict(
        sample_rate=24000,
        chunk_seconds=0.22,
        max_threads=1,
        preload=False,
        warmup=False,
        allow_cuda=True,
        low_memory_mode=True,
        language="pt",
        default_speaker="serena",
    )
    merged.update(kwargs)
    if TTSManager is not None:
        try:
            return TTSManager(**merged)
        except Exception:
            pass
    if LocalTTSManager is not None:
        try:
            return LocalTTSManager(**merged)
        except Exception:
            pass
    raise RuntimeError("Nenhum TTS local disponível")


def install_speech_lip_sync(
    app: Any,
    *,
    tts: Optional[Any] = None,
    config: Optional[SpeechLipSyncConfig] = None,
    on_viseme: Optional[Callable[[VisemeFrame], None]] = None,
    on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None,
) -> Tuple[SpeechLipSyncPipeline, AvatarSpeechAdapter]:
    """
    Instala a camada adicional de fala no app.

    Espera que o app tenha, quando possível:
    - app.audio com feed/start_proxy/reset/speaking
    - app.motion.face ou app.face para receber visemas
    """
    cfg = config or SpeechLipSyncConfig()

    if tts is None:
        tts = create_local_tts_manager_safe(
            sample_rate=cfg.sample_rate,
            chunk_seconds=cfg.chunk_seconds,
            max_threads=cfg.max_threads,
            preload=cfg.preload,
            warmup=cfg.warmup,
            allow_cuda=cfg.allow_cuda,
            low_memory_mode=cfg.low_memory_mode,
            language=cfg.language,
            default_speaker=cfg.default_speaker,
        )

    adapter = AvatarSpeechAdapter(app, config=cfg)
    pipeline = SpeechLipSyncPipeline(
        tts,
        config=cfg,
        audio_listener=on_audio_chunk or adapter.feed_audio,
        viseme_listener=on_viseme or adapter.apply_viseme,
        on_complete=adapter.stop,
    )
    adapter.bind(pipeline)
    pipeline.start()
    setattr(app, "speech_lip_sync", pipeline)
    setattr(app, "speech_lip_sync_adapter", adapter)
    return pipeline, adapter


__all__ = [
    "SpeechLipSyncConfig",
    "SpeechLipSyncPipeline",
    "AvatarSpeechAdapter",
    "PortugueseVisemeMapper",
    "VisemeFrame",
    "SmoothVisemeState",
    "SpeechSentenceBuffer",
    "clean_text",
    "split_sentences",
    "extract_complete_sentences",
    "install_speech_lip_sync",
    "create_local_tts_manager_safe",
]
