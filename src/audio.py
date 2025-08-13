from io import BytesIO
import queue
import threading
from typing import Generator, Optional
import wave

import numpy as np
import sounddevice as sd


class AudioManager:
    _DEVICES = sd.query_devices()
    _DEF_IN, _DEF_OUT = sd.default.device

    DEFAULT_INPUT_INDEX: int = (
        _DEF_IN
        if isinstance(_DEF_IN, int) and _DEF_IN >= 0
        else next((i for i, d in enumerate(_DEVICES) if d.get("max_input_channels", 0) > 0), 0)
    )
    DEFAULT_OUTPUT_INDEX: int = (
        _DEF_OUT
        if isinstance(_DEF_OUT, int) and _DEF_OUT >= 0
        else next((i for i, d in enumerate(_DEVICES) if d.get("max_output_channels", 0) > 0), 0)
    )
    try:
        _SR_IN = int(sd.query_devices(DEFAULT_INPUT_INDEX, "input")["default_samplerate"])
    except Exception:
        _SR_IN = None
    try:
        _SR_OUT = int(sd.query_devices(DEFAULT_OUTPUT_INDEX, "output")["default_samplerate"])
    except Exception:
        _SR_OUT = None
    DEFAULT_SAMPLERATE: int = (
        min(_SR_IN, _SR_OUT) if (_SR_IN and _SR_OUT) else (_SR_IN or _SR_OUT or 16_000)
    )

    def __init__(
        self,
        samplerate: Optional[int] = None,
        channels: int = 1,
        dtype: str = "int16",
        blocksize: int = 0,
        latency: str | float | None = None,
    ) -> None:
        self.input_device_index = self.DEFAULT_INPUT_INDEX
        self.output_device_index = self.DEFAULT_OUTPUT_INDEX
        self.samplerate = int(samplerate or self.DEFAULT_SAMPLERATE)
        self.channels = int(channels)
        self.dtype = dtype
        self.blocksize = int(blocksize)
        self.latency = latency

    def record_seconds(self, seconds: float, *, channels: Optional[int] = None) -> bytes:
        ch = int(channels or self.channels)
        frames = int(self.samplerate * float(seconds))
        audio = sd.rec(
            frames,
            samplerate=self.samplerate,
            channels=ch,
            dtype=self.dtype,
            device=self.input_device_index,
        )
        sd.wait()
        arr = np.asarray(audio, dtype=np.int16, order="C")
        return self._to_wav(arr, self.samplerate, ch)

    def record_until(
        self,
        stop_event: threading.Event,
        *,
        channels: Optional[int] = None,
        poll_ms: int = 20,
        timeout: Optional[float] = None,
    ) -> bytes:
        ch = int(channels or self.channels)
        frames: list[np.ndarray] = []

        def cb(indata, frames_count, time_info, status):
            if status:
                print(status)
            frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=ch,
            dtype=self.dtype,
            device=self.input_device_index,
            blocksize=self.blocksize or 0,
            latency=self.latency,
            callback=cb,
        ):
            step = poll_ms / 1000.0
            elapsed = 0.0
            while not stop_event.wait(step):
                if timeout is not None:
                    elapsed += step
                    if elapsed >= timeout:
                        break

        audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, ch), np.int16)
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16, copy=False)
        return self._to_wav(audio, self.samplerate, ch)

    def stream_pcm16(
        self,
        stop_event: threading.Event,
        *,
        chunk_ms: int = 30,
        channels: Optional[int] = None,
    ) -> Generator[bytes, None, None]:

        ch = int(channels or self.channels)
        q: "queue.Queue[bytes]" = queue.Queue(maxsize=10)
        blocksize = max(1, int(self.samplerate * (chunk_ms / 1000.0)))

        def cb(indata, frames_count, time_info, status):
            if status:
                print(status)
            try:
                q.put_nowait(indata.tobytes())
            except queue.Full:
                print("queue full")

        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=ch,
            dtype="int16",
            device=self.input_device_index,
            blocksize=blocksize,
            latency=self.latency,
            callback=cb,
        )
        stream.start()
        try:
            while not stop_event.is_set():
                try:
                    yield q.get(timeout=0.1)
                except queue.Empty:
                    continue
        finally:
            stream.stop()
            stream.close()
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break

    def play_pcm16(
        self,
        pcm_bytes: bytes,
        *,
        samplerate: Optional[int] = None,
        channels: Optional[int] = None,
        stop_event: threading.Event | None = None,
        blocking: bool = True,
    ) -> None:
        if not pcm_bytes:
            return
        sr = int(samplerate or self.samplerate)
        ch = int(channels or self.channels)

        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        arr = arr.reshape((-1, ch)) if ch > 1 else arr.reshape((-1, 1))

        if stop_event is None:
            sd.play(arr, samplerate=sr, device=self.output_device_index, blocking=blocking)
            return

        finished = threading.Event()

        def _waiter():
            try:
                sd.play(arr, samplerate=sr, device=self.output_device_index, blocking=True)
            finally:
                finished.set()

        t = threading.Thread(target=_waiter, daemon=True)
        t.start()

        while not finished.is_set():
            if stop_event.is_set():
                sd.stop()  # simple abort
                break
            sd.sleep(20)
        t.join(timeout=1.0)

    def play_wav(
        self,
        wav_bytes: bytes,
        *,
        stop_event: threading.Event | None = None,
        blocking: bool = True,
    ) -> None:
        if not wav_bytes:
            return
        ch, sr, sw, raw = self._from_wav(wav_bytes)
        if sw != 2:
            raise ValueError("play_wav expects 16-bit PCM WAV")
        arr = np.frombuffer(raw, dtype=np.int16)
        arr = arr.reshape((-1, ch)) if ch > 1 else arr.reshape((-1, 1))
        self.play_pcm16(
            arr.tobytes(), samplerate=sr, channels=ch, stop_event=stop_event, blocking=blocking
        )

    @staticmethod
    def pcm16_to_wav(pcm: bytes, samplerate: int, channels: int = 1) -> bytes:
        bio = BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(samplerate)
            wf.writeframes(pcm)
        return bio.getvalue()

    @staticmethod
    def _to_wav(arr_int16: np.ndarray, sr: int, ch: int) -> bytes:
        bio = BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(arr_int16.tobytes())
        return bio.getvalue()

    @staticmethod
    def _from_wav(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
        with wave.open(BytesIO(wav_bytes), "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            raw = wf.readframes(n)
        return ch, sr, sw, raw
