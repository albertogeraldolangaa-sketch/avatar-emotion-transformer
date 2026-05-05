
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional


import numpy as np

import avatar_runtime_merged as runtime


def _safe_put(q, item) -> None:
    try:
        q.put_nowait(item)
    except Exception:
        try:
            _ = q.get_nowait()
            q.put_nowait(item)
        except Exception:
            pass


def _patch_conversation_hub() -> None:
    hub_cls = getattr(runtime, "StreamingConversationHub", None)
    if hub_cls is None:
        return

    if not hasattr(hub_cls, "submit"):
        def submit(self, text: str) -> None:
            text = " ".join((text or "").strip().split())
            if not text:
                return
            try:
                self.hub_queue.put_nowait(text)
            except Exception:
                try:
                    _ = self.hub_queue.get_nowait()
                    self.hub_queue.put_nowait(text)
                except Exception:
                    pass
        hub_cls.submit = submit  # type: ignore[assignment]

    if not hasattr(hub_cls, "start"):
        def start(self) -> None:
            if getattr(self, "_started", False):
                return
            self._started = True
            self._thread.start()
        hub_cls.start = start  # type: ignore[assignment]

    if not hasattr(hub_cls, "stop"):
        def stop(self) -> None:
            self.stop_event.set()
            self.current_cancel.set()
            try:
                self.hub_queue.put_nowait("__STOP__")
            except Exception:
                pass
        hub_cls.stop = stop  # type: ignore[assignment]

    def _speak_fragment(self, fragment: str) -> None:
        fragment = " ".join((fragment or "").strip().split())
        if not fragment:
            return

        tts = getattr(self.app, "tts", None)
        audio = getattr(self.app, "audio", None)

        if tts is None:
            if audio is not None and hasattr(audio, "start_proxy"):
                try:
                    audio.start_proxy(fragment, duration=max(0.8, len(fragment) * 0.035))
                except Exception:
                    pass
            return

        gen = None
        try:
            if hasattr(tts, "stream_synthesize"):
                gen = tts.stream_synthesize(fragment, voice=getattr(getattr(tts, "cfg", None), "default_speaker", None))
            elif hasattr(tts, "synthesize"):
                gen = tts.synthesize(fragment)
            elif hasattr(tts, "synthesize_stream"):
                gen = tts.synthesize_stream(fragment)
        except Exception:
            gen = None

        if gen is None:
            if audio is not None and hasattr(audio, "start_proxy"):
                try:
                    audio.start_proxy(fragment, duration=max(0.8, len(fragment) * 0.035))
                except Exception:
                    pass
            return

        sr = int(getattr(tts, "_sample_rate", getattr(getattr(tts, "cfg", None), "sample_rate", 24000)) or 24000)

        for chunk in gen:
            if self.current_cancel.is_set() or self.stop_event.is_set():
                break
            if chunk is None:
                continue
            data = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if data.size == 0:
                continue
            try:
                if audio is not None and hasattr(audio, "feed"):
                    try:
                        audio.feed(data)
                    except TypeError:
                        audio.feed(data, sample_rate=sr)
                        
                # --- A SOLUÇÃO: Usar o buffer contínuo (Sink) em vez de sd.play/sd.wait ---
                sink = getattr(self.app, "_stream_audio_sink", None)
                if sink is not None:
                    sink.push(data)
                elif getattr(runtime, "sd", None) is not None:
                    # Fallback (evite cair aqui)
                    runtime.sd.play(data, samplerate=sr)
                    runtime.sd.wait()
                # ---------------------------------------------------------------------------
                    
            except Exception:
                if audio is not None and hasattr(audio, "start_proxy"):
                    try:
                        audio.start_proxy(fragment, duration=max(0.8, len(fragment) * 0.03))
                    except Exception:
                        pass
                break
    hub_cls._speak_fragment = _speak_fragment  # type: ignore[assignment]


def _patch_avatar_app() -> None:
    app_cls = getattr(runtime, "AvatarAppV2", None)
    if app_cls is None:
        app_cls = getattr(runtime, "AvatarApp", None)
    if app_cls is None:
        return

    def _begin_response(self, user_text: str) -> None:
        user_text = " ".join((user_text or "").strip().split())
        if not user_text:
            return
        self.last_input_text = user_text
        hub = getattr(self, "streaming_hub", None)
        if hub is not None and hasattr(hub, "submit"):
            hub.submit(user_text)
            return
        # fallback: fila normal
        if hasattr(self, "input_queue"):
            _safe_put(self.input_queue, user_text)

    def _draw(self) -> None:
        t = time.time() - getattr(self, "start_time", time.time())
        pending_user_text = ""

        try:
            while hasattr(self, "input_queue") and not self.input_queue.empty():
                txt = self.input_queue.get_nowait()
                txt = " ".join((str(txt) or "").strip().split())
                if not txt:
                    continue
                pending_user_text = txt
                self.last_input_text = txt
                hub = getattr(self, "streaming_hub", None)
                if hub is not None and hasattr(hub, "submit"):
                    hub.submit(txt)
                elif hasattr(self, "_begin_response"):
                    self._begin_response(txt)
        except Exception:
            pass

        try:
            motion_data = self._update_logic(user_text=pending_user_text)
        except Exception:
            motion_data = {}

        frame = None
        try:
            if hasattr(self, "renderer") and hasattr(self.renderer, "render"):
                try:
                    frame = self.renderer.render(
                        t,
                        getattr(self, "perception", getattr(self, "perception_data", None)),
                        self.emotion.state,
                        self.motion.body if hasattr(self.motion, "body") else motion_data,
                        self.motion.face if hasattr(self.motion, "face") else motion_data,
                        self.audio,
                        self.plan if getattr(self, "is_speaking", False) else None,
                        motion_data=motion_data,
                        blink_amount=getattr(self, "blink_amount", 0.0),
                    )
                except TypeError:
                    frame = self.renderer.render(
                        t,
                        getattr(self, "perception_data", getattr(self, "perception", None)),
                        self.emotion.state,
                        motion_data,
                        self.attention,
                        self.audio,
                        self.plan if getattr(self, "is_speaking", False) else None,
                    )
        except Exception:
            frame = None

        if frame is not None:
            try:
                display = frame.resize((self.display_res, self.display_res), runtime.Image.Resampling.LANCZOS)
                self.photo = runtime.ImageTk.PhotoImage(display)
                self.canvas.delete("all")
                self.canvas.create_image(self.display_res // 2, self.display_res // 2, anchor="center", image=self.photo)
            except Exception:
                pass

        try:
            self.root.after(max(1, int(1000 / max(1, int(getattr(self, "target_fps", 60))))), self._draw)
        except Exception:
            pass

    def _stop_speaking(self) -> None:
        try:
            hub = getattr(self, "streaming_hub", None)
            if hub is not None and hasattr(hub, "cancel_current"):
                hub.cancel_current()
        except Exception:
            pass
        try:
            self.is_speaking = False
            if hasattr(self, "audio"):
                self.audio.speaking = False
                if hasattr(self.audio, "reset"):
                    self.audio.reset()
        except Exception:
            pass

    def _setup_voice_bridge(self) -> None:
        try:
            if getattr(runtime, "install_voice_input_bridge", None) is None:
                return
            cfg_cls = getattr(runtime, "VoiceInputConfig", None)
            if cfg_cls is None:
                return
            config = cfg_cls(
                model_name="base",
                language="pt",
                vad_rms_threshold=0.02,
                vad_start_frames=2,
                vad_stop_frames=8,
                min_utterance_seconds=0.22,
                silence_flush_s=0.30,
                blocksize=512,
                sample_rate=16000,
                channels=1,
                device=None,
                aggressive_vad=True,
                aec_enabled=True,
                duplex_mode=True,
                barge_in_enabled=True,
                stream_partial_transcripts=False,
            )
            bridge = runtime.install_voice_input_bridge(
                self,
                config=config,
                on_speech_start=getattr(self, "_on_user_speech_start", None),
                on_speech_end=getattr(self, "_on_user_speech_end", None),
                on_barge_in=getattr(self, "_on_user_barge_in", None),
            )
            self.voice_input_bridge = bridge
            if hasattr(bridge, "attach_queue") and hasattr(self, "input_queue"):
                bridge.attach_queue(self.input_queue)
        except Exception:
            self.voice_input_bridge = None

    def _on_user_speech_start(self) -> None:
        try:
            self.last_activity_t = time.time()
        except Exception:
            pass
        try:
            if getattr(self, "is_speaking", False) and hasattr(self, "_interrupt_speaking"):
                self._interrupt_speaking("user_voice")
        except Exception:
            pass

    def _on_user_speech_end(self) -> None:
        try:
            self.last_activity_t = time.time()
        except Exception:
            pass

    def _on_user_barge_in(self) -> None:
        try:
            if hasattr(self, "_interrupt_speaking"):
                self._interrupt_speaking("barge_in")
        except Exception:
            pass
        try:
            self.is_speaking = False
            if hasattr(self, "audio"):
                self.audio.speaking = False
                if hasattr(self.audio, "reset"):
                    self.audio.reset()
        except Exception:
            pass
        try:
            hub = getattr(self, "streaming_hub", None)
            if hub is not None and hasattr(hub, "cancel_current"):
                hub.cancel_current()
        except Exception:
            pass

    app_cls._begin_response = _begin_response  # type: ignore[assignment]
    app_cls._draw = _draw  # type: ignore[assignment]
    app_cls._stop_speaking = _stop_speaking  # type: ignore[assignment]
    app_cls._setup_voice_bridge = _setup_voice_bridge  # type: ignore[assignment]
    app_cls._on_user_speech_start = _on_user_speech_start  # type: ignore[assignment]
    app_cls._on_user_speech_end = _on_user_speech_end  # type: ignore[assignment]
    app_cls._on_user_barge_in = _on_user_barge_in  # type: ignore[assignment]


_patch_conversation_hub()
_patch_avatar_app()

if __name__ == "__main__":
    if hasattr(runtime, "main_streaming"):
        runtime.main_streaming()
    elif hasattr(runtime, "main"):
        runtime.main()
