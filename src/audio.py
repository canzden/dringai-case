import threading
import time

from pynput import keyboard
from pynput.keyboard import Key


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
            on_press=on_press, on_release=on_release
        ) as L:  # pyright: ignore[reportArgumentType]
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
                return False
            if key != self.trigger_key:
                return
            if self._in_cooldown():
                return

            stop_event.set()
            self._set_cooldown()
            return False

        t = threading.Thread(
            target=lambda: keyboard.Listener(on_press=on_press).run(), daemon=True
        )  # pyright: ignore[reportArgumentType]
        t.start()
        return t
