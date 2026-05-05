from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterator, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except Exception:
    sd = None  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

try:
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore


class TextSink(Protocol):
    def put_nowait(self, item: str) -> None: ...


@dataclass(slots=True)
class VoiceInputConfig:
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 256
    device: Optional[int | str] = None

    vad_rms_threshold: float = 0.025
    vad_start_frames: int = 2
    vad_stop_frames: int = 5
    vad_release_frames: int = 4
    vad_noise_floor_decay: float = 0.985

    max_utterance_seconds: float = 18.0
    min_utterance_seconds: float = 0.18
    min_transcript_chars: int = 2
    silence_flush_s: float = 0.28
    partial_flush_s: float = 0.0

    language: str = "pt"
    model_name: str = "base"
    whisper_model_dir: Optional[str] = None
    prefer_faster_whisper: bool = True

    auto_restart: bool = True
    queue_maxsize: int = 128
    wake_words: Tuple[str, ...] = ("denise", "assistente", "olá", "ola")

    duplex_mode: bool = True
    aggressive_vad: bool = True
    aec_enabled: bool = True
    aec_tail_s: float = 1.25
    echo_attenuation: float = 0.82
    barge_in_enabled: bool = True
    barge_in_threshold: float = 0.014
    stream_partial_transcripts: bool = False
    max_interruption_latency_s: float = 0.15


@dataclass(slots=True)
class VoiceUtterance:
    started_at: float
    ended_at: float = 0.0
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    transcript: str = ""
    language: str = "pt"
    confidence: float = 0.0
    backend: str = "unknown"
    partial: bool = False

    @property
    def duration(self) -> float:
        end = self.ended_at or time.time()
        return max(0.0, end - self.started_at)


@dataclass(slots=True)
class StreamingVoiceEvent:
    kind: str
    text: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    confidence: float = 0.0
    backend: str = "unknown"
    audio_rms: float = 0.0
    audio_peak: float = 0.0
    sample_rate: int = 16000


class WhisperASRBackend:
    def __init__(self, config: VoiceInputConfig):
        self.config = config
        self.backend_name = "none"
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return whisper is not None or WhisperModel is not None or sr is not None

    def _load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            model_path = self.config.whisper_model_dir or self.config.model_name
            if WhisperModel is not None and self.config.prefer_faster_whisper:
                device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                self._model = WhisperModel(model_path, device=device, compute_type=compute_type)
                self.backend_name = "faster_whisper"
                return self._model
            if whisper is not None:
                self._model = whisper.load_model(model_path)
                self.backend_name = "whisper"
                return self._model
            if sr is not None:
                self._model = "speech_recognition"
                self.backend_name = "speech_recognition"
                return self._model
            raise RuntimeError("Nenhum backend ASR disponível")

    def transcribe(self, audio: np.ndarray) -> Tuple[str, float, str]:
        model = self._load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return "", 0.0, self.backend_name

        if self.backend_name == "faster_whisper":
            segments, info = model.transcribe(
                audio,
                language=self.config.language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            conf = float(getattr(info, "language_probability", 0.0) or 0.0)
            return text, conf, self.backend_name

        if self.backend_name == "whisper":
            result = model.transcribe(
                audio,
                language=self.config.language,
                fp16=bool(torch is not None and torch.cuda.is_available()),
                temperature=0.0,
            )
            return str(result.get("text", "")).strip(), 0.5, self.backend_name

        return "", 0.0, self.backend_name


class VoiceInputBridge:
    def __init__(
        self,
        config: Optional[VoiceInputConfig] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
    ):
        self.config = config or VoiceInputConfig()
        self.on_text = on_text
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_barge_in = on_barge_in

        self._backend = WhisperASRBackend(self.config)
        self._stream = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._history: Deque[VoiceUtterance] = deque(maxlen=32)

        self._speech_active = False
        self._speech_start = 0.0
        self._last_voice_frame = 0.0
        self._voice_frames = 0
        self._silence_frames = 0
        self._last_transcript = ""
        self._noise_floor = float(self.config.vad_rms_threshold)
        self._playback_energy = 0.0
        self._speaking = False
        self._last_emit_t = 0.0

        self._queue: queue.Queue[VoiceUtterance] = queue.Queue(maxsize=self.config.queue_maxsize)

    @property
    def available(self) -> bool:
        return sd is not None and self._backend.available

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_transcript(self) -> str:
        return self._last_transcript

    @property
    def history(self):
        return list(self._history)

    def attach_queue(self, q: TextSink) -> None:
        self.on_text = q.put_nowait

    def attach_text_sink(self, sink: Callable[[str], None]) -> None:
        self.on_text = sink

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = bool(speaking)

    def feed_playback_chunk(self, chunk: Optional[np.ndarray]) -> None:
        if chunk is None:
            self._playback_energy = max(0.0, self._playback_energy * 0.85)
            return
        data = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(data))))
        self._playback_energy = max(self._playback_energy * self.config.echo_attenuation, rms)
        self._last_emit_t = time.time()

    def cancel_current_utterance(self) -> None:
        with self._audio_lock:
            self._buffer.clear()
            self._speech_active = False
            self._voice_frames = 0
            self._silence_frames = 0
            self._speech_start = 0.0
        if self.on_barge_in:
            try:
                self.on_barge_in()
            except Exception:
                logger.exception("Falha no callback de barge-in")

    def start(self) -> None:
        if self._running:
            return
        if sd is None:
            raise RuntimeError("sounddevice não está instalado")
        if not self._backend.available:
            raise RuntimeError("Nenhum backend ASR disponível")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="VoiceInputBridge")
        self._thread.start()
        logger.info("VoiceInputBridge iniciado")

    def stop(self) -> None:
        self._running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    def _fire_async(self, cb: Optional[Callable[[], None]]) -> None:
        if cb is None:
            return
        try:
            threading.Thread(target=cb, daemon=True).start()
        except Exception:
            try:
                cb()
            except Exception:
                pass

    def _push_history(self, utt: VoiceUtterance) -> None:
        self._history.append(utt)
        try:
            self._queue.put_nowait(utt)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(utt)
            except Exception:
                pass

    def _emit_text(self, text: str) -> None:
        text = " ".join((text or "").strip().split())
        if len(text) < self.config.min_transcript_chars:
            return
        if text == self._last_transcript:
            return
        self._last_transcript = text
        if self.on_text:
            try:
                self.on_text(text)
            except Exception:
                logger.exception("Falha ao entregar texto transcrito")

    def _effective_vad_threshold(self) -> float:
        threshold = float(self.config.vad_rms_threshold)
        if self.config.aec_enabled and self._playback_energy > 0.0:
            threshold = max(threshold, self._playback_energy * 0.95)
        threshold = max(threshold, self._noise_floor * 1.08)
        return threshold

    def _callback(self, indata, frames, time_info, status):  # type: ignore[override]
        if not self._running:
            return
        try:
            data = np.asarray(indata, dtype=np.float32).reshape(-1)
            if data.size == 0:
                return

            rms = float(np.sqrt(np.mean(np.square(data))))
            peak = float(np.max(np.abs(data)))
            now = time.time()
            voiced = rms >= self._effective_vad_threshold()
            speech_started = False
            speech_ended = False

            with self._audio_lock:
                if voiced:
                    self._noise_floor = self._noise_floor * self.config.vad_noise_floor_decay + rms * (1.0 - self.config.vad_noise_floor_decay)
                    if not self._speech_active:
                        self._voice_frames += 1
                        if self._voice_frames >= self.config.vad_start_frames:
                            self._speech_active = True
                            self._speech_start = now
                            self._buffer.clear()
                            self._last_voice_frame = now
                            speech_started = True
                    else:
                        self._last_voice_frame = now
                    self._silence_frames = 0
                    self._buffer.append(data.copy())

                    if self.config.stream_partial_transcripts and (now - self._last_emit_t) >= max(0.12, self.config.partial_flush_s or 0.12):
                        self._last_emit_t = now
                        self._safe_emit_partial()
                else:
                    self._voice_frames = 0
                    self._noise_floor = self._noise_floor * self.config.vad_noise_floor_decay + rms * (1.0 - self.config.vad_noise_floor_decay)
                    if self._speech_active:
                        self._silence_frames += 1
                        if self._silence_frames >= self.config.vad_stop_frames or (now - self._speech_start) >= self.config.max_utterance_seconds:
                            self._flush_locked(now)
                            self._speech_active = False
                            self._silence_frames = 0
                            self._buffer.clear()
                            speech_ended = True

                if self.config.barge_in_enabled and self._speaking and voiced and rms >= self.config.barge_in_threshold:
                    self._speech_active = False
                    self._buffer.clear()
                    self._voice_frames = 0
                    self._silence_frames = 0
                    self._fire_async(self.on_barge_in)

            if speech_started:
                self._fire_async(self.on_speech_start)
            if speech_ended:
                self._fire_async(self.on_speech_end)

        except Exception:
            logger.exception("Erro no callback do microfone")

    def _safe_emit_partial(self) -> None:
        if not self.config.stream_partial_transcripts:
            return
        try:
            audio = np.concatenate(self._buffer, axis=0).astype(np.float32, copy=False)
            if audio.size < int(self.config.sample_rate * self.config.min_utterance_seconds):
                return
            transcript, conf, backend = self._backend.transcribe(audio)
        except Exception:
            return
        utt = VoiceUtterance(
            started_at=self._speech_start or time.time(),
            ended_at=time.time(),
            audio=audio,
            transcript=(transcript or "").strip(),
            language=self.config.language,
            confidence=float(conf),
            backend=backend,
            partial=True,
        )
        if utt.transcript:
            self._emit_text(utt.transcript)

    def _flush_locked(self, now: float) -> Optional[VoiceUtterance]:
        if not self._buffer:
            return None
        audio = np.concatenate(self._buffer, axis=0).astype(np.float32, copy=False)
        self._buffer.clear()
        if audio.size < int(self.config.sample_rate * self.config.min_utterance_seconds):
            return None
        try:
            transcript, conf, backend = self._backend.transcribe(audio)
        except Exception:
            logger.exception("Falha na transcrição de áudio")
            return None
        utt = VoiceUtterance(
            started_at=self._speech_start or now,
            ended_at=now,
            audio=audio,
            transcript=(transcript or "").strip(),
            language=self.config.language,
            confidence=float(conf),
            backend=backend,
        )
        self._push_history(utt)
        if utt.transcript:
            self._emit_text(utt.transcript)
        return utt

    def flush(self) -> Optional[VoiceUtterance]:
        with self._audio_lock:
            now = time.time()
            utt = self._flush_locked(now)
            self._speech_active = False
            self._voice_frames = 0
            self._silence_frames = 0
        return utt

    def poll_utterance(self, timeout: float = 0.0) -> Optional[VoiceUtterance]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice não está disponível")
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=self.config.blocksize,
                device=self.config.device,
                callback=self._callback,
            )
            self._stream.start()
            while self._running:
                time.sleep(0.03)
                with self._audio_lock:
                    if self._speech_active and self._last_voice_frame and (time.time() - self._last_voice_frame) >= self.config.silence_flush_s:
                        self._flush_locked(time.time())
                        self._speech_active = False
                        self._buffer.clear()
        except Exception:
            logger.exception("VoiceInputBridge falhou ao iniciar")
            self._running = False
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None


class StreamingVoiceInputBridge(VoiceInputBridge):
    def __init__(
        self,
        config: Optional[VoiceInputConfig] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
    ):
        super().__init__(
            config=config,
            on_text=on_text,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            on_barge_in=on_barge_in,
        )
        self.event_queue: "queue.Queue[StreamingVoiceEvent]" = queue.Queue(maxsize=self.config.queue_maxsize)
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.config.queue_maxsize)
        self.partial_text_queue: "queue.Queue[str]" = queue.Queue(maxsize=self.config.queue_maxsize)

    def _safe_put(self, q: "queue.Queue[Any]", item: Any) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                _ = q.get_nowait()
                q.put_nowait(item)
            except Exception:
                pass

    def _emit_event(self, event: StreamingVoiceEvent) -> None:
        self._safe_put(self.event_queue, event)

    def _callback(self, indata, frames, time_info, status):  # type: ignore[override]
        if not self._running:
            return
        try:
            data = np.asarray(indata, dtype=np.float32).reshape(-1)
            if data.size == 0:
                return
            now = time.time()
            rms = float(np.sqrt(np.mean(np.square(data))))
            peak = float(np.max(np.abs(data)))
            self._safe_put(self.audio_queue, data.copy())
            self._emit_event(StreamingVoiceEvent(
                kind="audio_frame",
                started_at=now,
                ended_at=now,
                audio_rms=rms,
                audio_peak=peak,
                sample_rate=self.config.sample_rate,
            ))
            super()._callback(indata, frames, time_info, status)
        except Exception:
            logger.exception("Erro no callback streaming do microfone")

    def _flush_locked(self, now: float) -> Optional[VoiceUtterance]:
        utt = super()._flush_locked(now)
        if utt is not None:
            self._emit_event(StreamingVoiceEvent(
                kind="utterance",
                text=utt.transcript,
                started_at=utt.started_at,
                ended_at=utt.ended_at,
                confidence=utt.confidence,
                backend=utt.backend,
                sample_rate=self.config.sample_rate,
            ))
        return utt

    def stream_events(self) -> Iterator[StreamingVoiceEvent]:
        while self._running:
            try:
                yield self.event_queue.get(timeout=0.1)
            except queue.Empty:
                continue


def install_voice_input_bridge(
    app: Any,
    *,
    config: Optional[VoiceInputConfig] = None,
    on_text: Optional[Callable[[str], None]] = None,
    on_speech_start: Optional[Callable[[], None]] = None,
    on_speech_end: Optional[Callable[[], None]] = None,
    on_barge_in: Optional[Callable[[], None]] = None,
) -> VoiceInputBridge:
    """Liga a entrada de voz ao app, escolhendo o modo streaming quando disponível."""
    cfg = config or VoiceInputConfig()
    use_streaming = bool(
        getattr(app, "conversation_bus", None) is not None
        or getattr(app, "audio_in_queue", None) is not None
        or getattr(app, "stream_mode", False)
        or getattr(app, "full_duplex", False)
    )

    if use_streaming:
        bridge: VoiceInputBridge = StreamingVoiceInputBridge(
            config=cfg,
            on_text=on_text,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            on_barge_in=on_barge_in,
        )
        if hasattr(app, "audio_in_queue"):
            bridge.audio_queue = getattr(app, "audio_in_queue")
        if hasattr(app, "voice_event_queue"):
            bridge.event_queue = getattr(app, "voice_event_queue")
        if hasattr(app, "partial_text_queue"):
            bridge.partial_text_queue = getattr(app, "partial_text_queue")
    else:
        bridge = VoiceInputBridge(
            config=cfg,
            on_text=on_text,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            on_barge_in=on_barge_in,
        )

    if bridge.on_text is None and hasattr(app, "input_queue"):
        bridge.attach_queue(app.input_queue)
    elif bridge.on_text is None and hasattr(app, "on_text") and callable(getattr(app, "on_text")):
        bridge.on_text = getattr(app, "on_text")
    elif bridge.on_text is None:
        raise AttributeError("O objeto app não tem input_queue nem on_text")

    bridge.start()
    setattr(app, "voice_input_bridge", bridge)
    return bridge


__all__ = [
    "VoiceInputConfig",
    "VoiceUtterance",
    "StreamingVoiceEvent",
    "VoiceInputBridge",
    "StreamingVoiceInputBridge",
    "WhisperASRBackend",
    "install_voice_input_bridge",
]

# ==============================================================================
# Bridge hardening patch
# ==============================================================================

_VOICE_PATCH_MARKER = True

_VoiceInputBridge_attach_queue_orig = VoiceInputBridge.attach_queue
_VoiceInputBridge_start_orig = VoiceInputBridge.start


def _vib_attach_queue(self, q: TextSink) -> None:
    self.on_text = q.put_nowait


def _vib_start(self) -> None:
    if self._running:
        return
    if sd is None:
        raise RuntimeError('sounddevice não está instalado')
    if not self._backend.available:
        raise RuntimeError('Nenhum backend ASR disponível')
    self._running = True
    self._thread = threading.Thread(target=self._run, daemon=True, name='VoiceInputBridge')
    self._thread.start()


try:
    VoiceInputBridge.attach_queue = _vib_attach_queue  # type: ignore[assignment]
    VoiceInputBridge.start = _vib_start  # type: ignore[assignment]
except Exception:
    pass
