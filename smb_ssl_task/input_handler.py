"""
Unified input handler for the SMB SSL task.

Uses pyglet's KeyStateHandler for real-time key state (not event-based).
This ensures held keys are reported on every frame, which is required
for the duration-aware MSP mode.
"""

from pyglet.window import key as _pyglet_keys

from smb_ssl_task.config import (
    GAMEPAD_ENABLED,
    GAMEPAD_BUTTON_A,
    GAMEPAD_BUTTON_B,
    GAMEPAD_DPAD_THRESHOLD,
    NES_BUTTONS,
)

# Map pyglet key constants -> NES button names
_PYGLET_KEY_TO_NES = {
    _pyglet_keys.RIGHT: "RIGHT",
    _pyglet_keys.LEFT: "LEFT",
    _pyglet_keys.UP: "UP",
    _pyglet_keys.DOWN: "DOWN",
    _pyglet_keys.X: "A",
    _pyglet_keys.Z: "B",
}


class InputHandler:
    """Reads keyboard + gamepad, outputs NES button state.

    Uses pyglet's KeyStateHandler to track which keys are currently
    held down (state-based, not event-based).  This is updated
    automatically by pyglet's event loop during ``win.flip()``.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window (needed to attach the pyglet key handler).
    gamepad_enabled : bool
        Whether to attempt gamepad initialization.
    """

    def __init__(self, win, gamepad_enabled=GAMEPAD_ENABLED):
        self._key_handler = _pyglet_keys.KeyStateHandler()
        # Attach to the pyglet window so key state is updated each frame
        pyglet_win = getattr(win, 'winHandle', None)
        if pyglet_win is not None:
            pyglet_win.push_handlers(self._key_handler)

        self._gamepad = None
        if gamepad_enabled:
            try:
                from psychopy.hardware.joystick import Joystick
                if Joystick.getNumJoysticks() > 0:
                    self._gamepad = Joystick(0)
            except Exception:
                pass  # No gamepad available, keyboard-only

    def get_nes_state(self):
        """Return set of currently pressed NES buttons.

        Combines keyboard and gamepad input.  Keyboard state is
        read from pyglet's KeyStateHandler which reports held keys
        on every frame (not just on the frame they were pressed).

        Returns
        -------
        set[str]
            e.g. {"RIGHT", "A"} or {"RIGHT", "A", "B"}
        """
        pressed = set()

        # --- Keyboard (state-based via pyglet) ---
        for pyglet_key, nes_button in _PYGLET_KEY_TO_NES.items():
            if self._key_handler[pyglet_key]:
                pressed.add(nes_button)

        # --- Gamepad ---
        if self._gamepad is not None:
            try:
                # D-pad / left stick -> directions
                x_axis = self._gamepad.getX()
                y_axis = self._gamepad.getY()

                if x_axis > GAMEPAD_DPAD_THRESHOLD:
                    pressed.add("RIGHT")
                elif x_axis < -GAMEPAD_DPAD_THRESHOLD:
                    pressed.add("LEFT")

                if y_axis > GAMEPAD_DPAD_THRESHOLD:
                    pressed.add("DOWN")
                elif y_axis < -GAMEPAD_DPAD_THRESHOLD:
                    pressed.add("UP")

                # Face buttons
                if self._gamepad.getButton(GAMEPAD_BUTTON_A):
                    pressed.add("A")
                if self._gamepad.getButton(GAMEPAD_BUTTON_B):
                    pressed.add("B")
            except Exception:
                pass  # Gamepad read error, ignore

        return pressed

    def get_action_array(self):
        """Return 9-element list for gym-retro env.step().

        Follows NES_BUTTONS order: [B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]

        Returns
        -------
        list[int]
            0 or 1 for each button position.
        """
        pressed = self.get_nes_state()
        action = []
        for button in NES_BUTTONS:
            if button is None:
                action.append(0)
            else:
                action.append(1 if button in pressed else 0)
        return action

    def check_escape(self):
        """Return True if escape key is currently held."""
        return self._key_handler[_pyglet_keys.ESCAPE]

    def clear(self):
        """No-op — state-based handler has no event buffer to clear."""
        pass
