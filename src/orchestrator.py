import threading
import time

from pynput import keyboard
from pynput.keyboard import Key

from audio import AudioManager
from config import load_config
from stt import DeepgramSTT


class HotkeyTap:
    """
    HotkeyTap provides hotkey-based tap detection with cooldown and exit functionality.
    """

    DEFAULT_COOLDOWN_SECONDS: float = 1.5
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


def run():
    keys = HotkeyTap(cooldown_s=0.8)
    audio = AudioManager(channels=1)

    stt_config, llm_config = load_config()
    tts = DeepgramSTT(stt_config)
    shutdown = threading.Event()

    print("press SPACE to init conversation")
    if not keys.wait_tap():
        print("early exit")
        return

    counter = 1
    try:
        while not shutdown.is_set():
            print(f"\n[{counter}] Recording… (SPACE to stop, ESC to exit)")
            stop_rec = threading.Event()
            rec_listener = keys.stop_on_next_press(stop_rec, shutdown)

            wav = audio.record_until(stop_rec, channels=1)
            rec_listener.join(timeout=1.0)
            if shutdown.is_set():
                break

            print(f"[{counter}] Playing… (SPACE to abort, ESC to exit)")
            # stop_play = threading.Event()
            # play_listener = keys.stop_on_next_press(stop_play, shutdown)

            # audio.play_wav(wav, stop_event=stop_play, blocking=True)

            # play_listener.join(timeout=1.0)
            print(tts.transcribe_wav(wav))

            if shutdown.is_set():
                break

            counter += 1
    except KeyboardInterrupt:
        pass
    finally:
        # TODO: destroy idle resources, sockets, io interfaces here
        print("gracefully shutdown")


if __name__ == "__main__":
    run()
