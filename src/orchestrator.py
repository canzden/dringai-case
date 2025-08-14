import threading
import time

from pynput import keyboard
from pynput.keyboard import Key

from audio import AudioManager
from config import load_config
from llm import OpenAIClient
from logger import VoiceAssistantLogger
from stt import DeepgramSTT
from tts import ElevenLabsTTS


class HotkeyTap:
    """
    HotkeyTap provides hotkey-based tap detection with cooldown and exit functionality.
    """

    DEFAULT_COOLDOWN_SECONDS: float = 0.8
    TRIGGER_KEY: Key = Key.space
    EXIT_KEY: Key = Key.esc

    def __init__(
        self,
        trigger_key: Key = TRIGGER_KEY,
        exit_key: Key = EXIT_KEY,
        cooldown_s: float = DEFAULT_COOLDOWN_SECONDS,
    ):
        self.trigger_key = trigger_key
        self.exit_key = exit_key
        self.cooldown_s = cooldown_s
        self._cool_until = 0.0
        self._lock = threading.Lock()

    def _set_cooldown(self):
        if self.cooldown_s > 0:
            with self._lock:
                self._cool_until = time.monotonic() + self.cooldown_s

    def _in_cooldown(self) -> bool:
        with self._lock:
            return time.monotonic() < self._cool_until

    def wait_tap(self) -> bool:
        """
        Waits for a specific key tap or exit key press using a keyboard listener.

        The method listens for keyboard events and blocks until either the trigger key is pressed and released,
        or the exit key is pressed. If the exit key is pressed, the method returns False. Otherwise, it returns True
        after the trigger key is tapped.

        Returns:
            bool: True if the trigger key was tapped, False if the exit key was pressed.
        """
        pressed = threading.Event()
        exited = threading.Event()

        def on_press(key):
            if key == self.exit_key:
                exited.set()
                return False
            if key == self.trigger_key and not pressed.is_set():
                self._set_cooldown()
                pressed.set()

        def on_release(key):
            if key == self.trigger_key and pressed.is_set():
                return False

        with keyboard.Listener(
            on_press=on_press, on_release=on_release  # pyright: ignore[reportArgumentType]
        ) as L:
            L.join()

        return not exited.is_set()

    def stop_on_next_press(
        self, stop_event: threading.Event, shutdown: threading.Event
    ) -> threading.Thread:
        """
        Starts a background thread that listens for keyboard events and sets the provided events based on key presses.

        When the `trigger_key` is pressed and not in cooldown, sets the `stop_event`.
        When the `exit_key` is pressed, sets the `shutdown` event and stops the listener.
        Ignores other keys and enforces a cooldown period between trigger key presses.

        Args:
            stop_event (threading.Event): Event to set when the trigger key is pressed.
            shutdown (threading.Event): Event to set when the exit key is pressed.

        Returns:
            threading.Thread: The thread running the keyboard listener.
        """

        def on_press(key):
            if key == self.exit_key:
                shutdown.set()
                stop_event.set()
                return False
            if key != self.trigger_key:
                return
            if self._in_cooldown():
                return

            stop_event.set()
            self._set_cooldown()
            return False

        t = threading.Thread(
            target=lambda: keyboard.Listener(on_press=on_press).run(),
            daemon=True,  # pyright: ignore[reportArgumentType]
        )
        t.start()
        return t


class VoiceAgent:
    def __init__(
        self,
        *,
        audio: AudioManager,
        stt: DeepgramSTT,
        llm: OpenAIClient,
        tts: ElevenLabsTTS,
        keys: HotkeyTap,
        logger: VoiceAssistantLogger,
    ) -> None:
        self.audio = audio
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.keys = keys
        self.logger = logger
        self.shutdown = threading.Event()

    def run(self) -> None:
        print("Press SPACE to start (ESC to exit).")
        if not self.keys.wait_tap():
            print("Early exit.")
            return

        turn = 1
        try:
            while not self.shutdown.is_set():

                # record user voice
                print(f"\n[{turn}] Recording… (SPACE to stop, ESC to exit)")
                stop_rec = threading.Event()
                rec_listener = self.keys.stop_on_next_press(stop_rec, self.shutdown)
                wav = self.audio.record_until(stop_rec, channels=1, timeout=10_000.0)
                rec_listener.join(timeout=0.5)
                if self.shutdown.is_set():
                    break

                # speech to text
                try:
                    user_text = self.stt.transcribe_wav(wav)
                except Exception as e:
                    print(f"[STT error] {e}")
                    continue
                print(f"[user] {user_text!r}")
                if not user_text.strip():
                    continue

                # llm
                try:
                    reply = self.llm.reply_short(user_text)
                except Exception as e:
                    print(f"[LLM error] {e}")
                    continue
                print(f"[assistant] {reply!r}")
                if not reply.strip():
                    continue

                # tts
                print("Speaking… (SPACE to abort, ESC to exit)")
                stop_play = threading.Event()
                play_listener = self.keys.stop_on_next_press(stop_play, self.shutdown)

                try:
                    pcm = self.tts.generate_speech(reply)  # bytes (PCM16 @ 16kHz)
                    wav_bytes = AudioManager.pcm16_to_wav(pcm, 16_000)
                except Exception as e:
                    print(f"[TTS error] {e}")
                    play_listener.join(timeout=0.5)
                    continue

                self.audio.play_wav(wav_bytes, stop_event=stop_play, blocking=True)
                play_listener.join(timeout=0.5)
                if self.shutdown.is_set():
                    break

                self.logger.log(turn_id=turn, user_text=user_text, assistant_text=reply)
                turn += 1

        except KeyboardInterrupt:
            pass
        finally:
            print("\nGraceful shutdown.")


def build_agent() -> VoiceAgent:
    stt_cfg, openai_cfg, eleven_cfg = load_config()
    audio = AudioManager(channels=1)
    stt = DeepgramSTT(stt_cfg)
    llm = OpenAIClient(openai_cfg)
    tts = ElevenLabsTTS(eleven_cfg)
    keys = HotkeyTap()
    logger = VoiceAssistantLogger()
    return VoiceAgent(audio=audio, stt=stt, llm=llm, tts=tts, keys=keys, logger=logger)


if __name__ == "__main__":
    build_agent().run()
