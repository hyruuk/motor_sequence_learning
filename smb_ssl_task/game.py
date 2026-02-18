"""
Gameplay mode for the SMB SSL task.

Wraps stable-retro (gym-retro) for Super Mario Bros. gameplay
and renders frames into a PsychoPy window via raw OpenGL texturing.

PsychoPy 2025's ImageStim shader pipeline can fail on certain GL
contexts, so we bypass it entirely: upload observations as a GL
texture and draw a textured quad with the fixed-function pipeline,
saving/restoring all GL state so PsychoPy's own rendering is
unaffected.
"""

import ctypes
import gzip
import os

import numpy as np
import pyglet.gl as GL
from psychopy import core

from collections import defaultdict

from smb_ssl_task.config import (
    GAME_RENDER_SIZE,
    GAMEPLAY_MAX_DURATION,
    GAME_NAME,
    EMULATOR_FPS,
    NES_BUTTONS,
    SYMBOL_TO_BUTTONS,
    ACTION_COLOR_CORRECT,
    ACTION_COLOR_ERROR,
)
from smb_ssl_task.config import verbose
from smb_ssl_task.scenes import get_savestate_path, buttons_to_symbol

# Time between emulator steps (NES runs at 60fps)
_FRAME_INTERVAL = 1.0 / EMULATOR_FPS


class _GLTextureRenderer:
    """Renders a numpy RGB array as an OpenGL texture quad.

    Bypasses PsychoPy's shader pipeline and uses fixed-function GL.
    Saves and restores all GL state (including matrices and shader
    program) so PsychoPy is unaffected.
    """

    def __init__(self, win, display_size):
        self._win = win
        self._display_size = display_size  # (w, h) of the quad to draw
        self._tex_id = GL.GLuint()
        GL.glGenTextures(1, ctypes.byref(self._tex_id))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id.value)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_NEAREST)
        self._tex_w = 0
        self._tex_h = 0

    def update(self, observation):
        """Upload uint8 RGB observation to the GL texture."""
        h, w = observation.shape[:2]
        # Flip vertically: numpy is top-left origin, GL is bottom-left
        obs = np.ascontiguousarray(observation[::-1])
        data_ptr = obs.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id.value)
        if w != self._tex_w or h != self._tex_h:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data_ptr,
            )
            self._tex_w, self._tex_h = w, h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data_ptr,
            )

    def draw(self):
        """Draw the texture as a centered quad."""
        if self._tex_w == 0:
            return

        # --- Save current shader program ---
        prev_prog = GL.GLint()
        GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM, ctypes.byref(prev_prog))

        # --- Save all GL state + matrices ---
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()

        # --- Set up our own fixed-function pipeline ---
        try:
            GL.glUseProgram(0)
        except Exception:
            pass

        # Set up orthographic projection centered on origin (pixel units)
        win_w, win_h = self._win.size
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-win_w / 2, win_w / 2, -win_h / 2, win_h / 2, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id.value)

        dw = self._display_size[0] / 2
        dh = self._display_size[1] / 2

        GL.glColor4f(1, 1, 1, 1)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(-dw, -dh)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(dw, -dh)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(dw, dh)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(-dw, dh)
        GL.glEnd()

        # --- Restore matrices ---
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()

        # --- Restore all GL state ---
        GL.glPopAttrib()

        # --- Restore shader program (not covered by glPopAttrib) ---
        try:
            GL.glUseProgram(prev_prog.value)
        except Exception:
            pass

    def cleanup(self):
        """Delete the GL texture."""
        if self._tex_id.value:
            GL.glDeleteTextures(1, ctypes.byref(self._tex_id))
            self._tex_id = GL.GLuint()


class GameEngine:
    """Wraps stable-retro for SMB gameplay rendering in PsychoPy.

    Parameters
    ----------
    win : psychopy.visual.Window
    scenes_path : str
        Path to the mario.scenes dataset root.
    env : retro.RetroEnv, optional
        Pre-created retro environment.  Must be created *before* the
        PsychoPy window so that its GL context init doesn't invalidate
        PsychoPy's compiled shaders.  If not provided, one is created
        here (only safe if no PsychoPy window exists yet).
    game_name : str
        Retro game name (default from config).
    """

    def __init__(self, win, scenes_path, env=None, game_name=GAME_NAME):
        self.win = win
        self.scenes_path = scenes_path
        self.game_name = game_name

        if env is not None:
            self.env = env
        else:
            import stable_retro
            self.env = stable_retro.make(
                game=game_name,
                inttype=stable_retro.data.Integrations.ALL,
                use_restricted_actions=stable_retro.Actions.ALL,
                render_mode=None,
            )

        # Get a frame to determine observation size
        first_obs = self.env.reset()[0]
        obs_h, obs_w = first_obs.shape[:2]
        min_ratio = min(
            GAME_RENDER_SIZE[0] / obs_w,
            GAME_RENDER_SIZE[1] / obs_h,
        )
        display_w = int(min_ratio * obs_w)
        display_h = int(min_ratio * obs_h)

        # Raw GL renderer — bypasses PsychoPy's shader pipeline
        self._renderer = _GLTextureRenderer(win, (display_w, display_h))
        self._renderer.update(first_obs)
        self._current_info = {}
        # Death detection state (reset on each load_scene)
        self._prev_player_state = 0
        self._prev_lives = 0

    def load_scene(self, scene_id, scene_info, state_path=None):
        """Load a savestate for a scene and reset the environment.

        Parameters
        ----------
        scene_id : str
        scene_info : dict
            Scene dict with entry, exit, layout fields.
        state_path : str, optional
            Explicit path to a .state file. If not provided, searches
            for any matching state in the dataset.
        """
        if state_path is None:
            state_path = get_savestate_path(scene_id, scenes_path=self.scenes_path)
        if state_path is None:
            raise FileNotFoundError(
                f"No savestate found for scene '{scene_id}'"
            )

        # State files are gzip-compressed (same format as stable_retro's
        # own load_state).  Try gzip first, fall back to raw bytes.
        try:
            with gzip.open(state_path, "rb") as f:
                state_data = f.read()
        except gzip.BadGzipFile:
            with open(state_path, "rb") as f:
                state_data = f.read()

        if verbose():
            print(f"  [LOAD] {scene_id} | {os.path.basename(state_path)} "
                  f"({len(state_data)} bytes)")

        self.env.initial_state = state_data
        obs, _info = self.env.reset()
        self._renderer.update(obs)
        self._current_info = _info
        # Reset death detection state from the fresh reset info
        self._prev_player_state = _info.get("player_state", 0)
        self._prev_lives = _info.get("lives", 0)

    def step(self, nes_buttons):
        """Advance the game by one frame.

        Parameters
        ----------
        nes_buttons : list[int]
            9-element action array for env.step().

        Returns
        -------
        dict
            Info dict from the environment.
        """
        self._prev_player_state = self._current_info.get("player_state", 0)
        self._prev_lives = self._current_info.get("lives", 0)
        obs, reward, done, truncated, info = self.env.step(nes_buttons)
        self._renderer.update(obs)
        self._current_info = info
        return info

    def render(self):
        """Draw current game frame to the PsychoPy window."""
        self._renderer.draw()

    def get_player_x(self, info=None):
        """Get player x-position from info dict.

        Uses RAM values: x_hi * 256 + x_lo.
        """
        if info is None:
            info = self._current_info
        x_hi = info.get("xscrollHi", 0)
        x_lo = info.get("xscrollLo", 0)
        return x_hi * 256 + x_lo

    def is_scene_complete(self, scene_info, info=None):
        """Check if player has reached the scene exit point."""
        if info is None:
            info = self._current_info
        player_x = self.get_player_x(info)
        return player_x >= scene_info["exit"]

    def is_death(self, info=None):
        """Check if player has died.

        Detects two kinds of death (matching mario.scenes logic):
        - Enemy / timeout: ``player_state`` transitions to 11.
        - Fall: ``lives`` decrease without the dying animation.
        """
        if info is None:
            info = self._current_info
        player_state = info.get("player_state", 0)
        lives = info.get("lives", 0)
        # Dying animation transition
        if player_state == 11 and self._prev_player_state != 11:
            return True
        # Fall death (lives decrease without player_state 11)
        if lives < self._prev_lives:
            return True
        return False

    def close(self):
        """Close the retro environment and free GL resources."""
        self._renderer.cleanup()
        if self.env is not None:
            self.env.close()
            self.env = None


def execute_gameplay_trial(win, input_handler, engine, scene_info,
                           max_duration=GAMEPLAY_MAX_DURATION,
                           speed_factor=1.0):
    """Run one gameplay execution of a scene.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    engine : GameEngine
    scene_info : dict
        Scene dict with id, entry, exit, layout.
    max_duration : float
        Maximum seconds before timeout.
    speed_factor : float
        Emulator speed multiplier (1.0 = real-time, 0.5 = half speed).

    Returns
    -------
    dict or None
        None if escape pressed. Otherwise:
        - outcome: "completed", "death", or "timeout"
        - traversal_time: float
        - distance_reached: float (fraction of scene traversed)
    """
    scene_length = scene_info["exit"] - scene_info["entry"]
    scaled_interval = _FRAME_INTERVAL / speed_factor

    input_handler.clear()
    start_time = core.getTime()
    next_step_time = start_time  # When to step the emulator next

    while True:
        now = core.getTime()
        elapsed = now - start_time

        # Check timeout
        if elapsed >= max_duration:
            info = engine._current_info
            player_x = engine.get_player_x(info)
            distance = max(0.0, (player_x - scene_info["entry"]) / scene_length)
            return {
                "outcome": "timeout",
                "traversal_time": elapsed,
                "distance_reached": min(1.0, distance),
            }

        # Check escape
        if input_handler.check_escape():
            return None

        # Step the emulator; on faster displays, re-render last frame
        if now >= next_step_time:
            action = input_handler.get_action_array()
            info = engine.step(action)
            next_step_time += scaled_interval

            # Check completion
            if engine.is_scene_complete(scene_info, info):
                elapsed = core.getTime() - start_time
                return {
                    "outcome": "completed",
                    "traversal_time": elapsed,
                    "distance_reached": 1.0,
                }

            # Check death
            if engine.is_death(info):
                elapsed = core.getTime() - start_time
                player_x = engine.get_player_x(info)
                distance = max(0.0, (player_x - scene_info["entry"]) / scene_length)
                return {
                    "outcome": "death",
                    "traversal_time": elapsed,
                    "distance_reached": min(1.0, distance),
                }

        # Always render + flip at display refresh rate
        engine.render()
        win.flip()


def execute_gameplay_scan_trial(win, input_handler, engine, scene_info,
                                duration, speed_factor=1.0):
    """Fixed-time gameplay execution for scan sessions.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    engine : GameEngine
    scene_info : dict
    duration : float
        Fixed execution window in seconds.
    speed_factor : float
        Emulator speed multiplier (1.0 = real-time, 0.5 = half speed).

    Returns
    -------
    dict or None
    """
    scene_length = scene_info["exit"] - scene_info["entry"]
    scaled_interval = _FRAME_INTERVAL / speed_factor

    input_handler.clear()
    start_time = core.getTime()
    next_step_time = start_time
    outcome = "timeout"
    final_distance = 0.0
    game_active = True  # False after completion/death (keep rendering)

    while True:
        now = core.getTime()
        elapsed = now - start_time
        if elapsed >= duration:
            break

        if input_handler.check_escape():
            return None

        # Step emulator while game is active
        if game_active and now >= next_step_time:
            action = input_handler.get_action_array()
            info = engine.step(action)
            next_step_time += scaled_interval

            if engine.is_scene_complete(scene_info, info):
                outcome = "completed"
                final_distance = 1.0
                game_active = False

            elif engine.is_death(info):
                outcome = "death"
                player_x = engine.get_player_x(info)
                final_distance = max(
                    0.0, (player_x - scene_info["entry"]) / scene_length
                )
                game_active = False

            else:
                player_x = engine.get_player_x(info)
                final_distance = max(
                    0.0, (player_x - scene_info["entry"]) / scene_length
                )

        # Always render + flip (shows last frame after completion/death)
        engine.render()
        win.flip()

    traversal_time = min(core.getTime() - start_time, duration)

    return {
        "outcome": outcome,
        "traversal_time": traversal_time,
        "distance_reached": min(1.0, final_distance),
    }


def _symbol_to_action_array(symbol):
    """Convert an action symbol to a 9-element action array for env.step().

    Parameters
    ----------
    symbol : str
        Action symbol (e.g. "rR", "RJ", "_").

    Returns
    -------
    list[int]
        9-element list matching NES_BUTTONS order.
    """
    buttons = SYMBOL_TO_BUTTONS.get(symbol, set())
    action = []
    for btn in NES_BUTTONS:
        if btn is None:
            action.append(0)
        else:
            action.append(1 if btn in buttons else 0)
    return action


def replay_bk2_preview(win, input_handler, engine, seq_display,
                       action_sequence, speed_factor=1.0):
    """Replay the compressed action sequence through the emulator.

    The bar fills green progressively as the emulator replays the scene.
    No player input is accepted (preview only).

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
        Used only for escape checking.
    engine : GameEngine
    seq_display : ActionSequenceDisplay
    action_sequence : list[tuple[str, int]]
        Compressed action sequence: (symbol, duration_frames) tuples.
    speed_factor : float
        Playback speed multiplier (1.0 = real-time, 0.5 = half speed).

    Returns
    -------
    dict or None
        None if escape pressed. Otherwise {"exit_x": int}.
    """
    seq_display.show(action_sequence)

    # Build a flat schedule: for each frame, its action and element index
    frame_schedule = []
    for elem_idx, (symbol, duration_frames) in enumerate(action_sequence):
        action = _symbol_to_action_array(symbol)
        for f in range(duration_frames):
            frame_schedule.append((elem_idx, action, f, duration_frames))

    total_frames = len(frame_schedule)
    frame_idx = 0
    scaled_interval = _FRAME_INTERVAL / speed_factor

    start_time = core.getTime()
    next_step_time = start_time

    while frame_idx < total_frames:
        if input_handler.check_escape():
            return None

        now = core.getTime()

        # Step emulator frames that are due
        if now >= next_step_time:
            elem_idx, action, f, duration_frames = frame_schedule[frame_idx]
            engine.step(action)
            frame_idx += 1
            next_step_time += scaled_interval

            # Update bar fill for current element
            frac = (f + 1) / duration_frames
            seq_display.update_bar_fill(elem_idx, frac, color=ACTION_COLOR_CORRECT)

            # Mark all previous elements as fully filled
            if f == 0 and elem_idx > 0:
                for prev in range(elem_idx):
                    seq_display.update_bar_fill(prev, 1.0, color=ACTION_COLOR_CORRECT)

        # Always render + flip at display refresh rate
        engine.render()
        seq_display.draw()
        win.flip()

    exit_x = engine.get_player_x()
    if verbose():
        print(f"  [PREVIEW] Completed. exit_x={exit_x}, total_frames={total_frames}, "
              f"speed={speed_factor:.2f}x")
    return {"exit_x": exit_x}


def execute_gameplay_with_tracking(win, input_handler, engine, seq_display,
                                   action_sequence, exit_x,
                                   max_duration=GAMEPLAY_MAX_DURATION,
                                   speed_factor=1.0):
    """Run one player gameplay execution with real-time sequence tracking.

    The bar fills green/red based on whether the player's input matches
    the target element at the current frame position.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    engine : GameEngine
    seq_display : ActionSequenceDisplay
    action_sequence : list[tuple[str, int]]
        Compressed action sequence.
    exit_x : int
        Player x-position at which to end (from preview replay).
    max_duration : float
        Maximum seconds before timeout.
    speed_factor : float
        Emulator speed multiplier (1.0 = real-time, 0.5 = half speed).

    Returns
    -------
    dict or None
        None if escape. Otherwise dict with both gameplay and sequence fields.
    """
    n_elements = len(action_sequence)
    target_symbols = [s for s, _ in action_sequence]
    durations_frames = [frames for _, frames in action_sequence]
    scaled_interval = _FRAME_INTERVAL / speed_factor

    # Cumulative frame boundaries: cum[i] = first frame of element i
    cum = [0]
    for d in durations_frames:
        cum.append(cum[-1] + d)
    total_target_frames = cum[-1]

    seq_display.show(action_sequence)
    seq_display.reset()

    input_handler.clear()
    start_time = core.getTime()
    next_step_time = start_time
    frame_counter = 0

    # Per-element tracking (same pattern as MSP's _run_continuous_timeline)
    elem_chord_times = [defaultdict(float) for _ in range(n_elements)]
    elem_dominant = [set() for _ in range(n_elements)]
    prev_elem_idx = -1

    while True:
        now = core.getTime()
        elapsed = now - start_time

        if elapsed >= max_duration:
            outcome = "timeout"
            break

        if input_handler.check_escape():
            return None

        # Step emulator
        if now >= next_step_time:
            action = input_handler.get_action_array()
            info = engine.step(action)
            next_step_time += scaled_interval
            frame_counter += 1

            # Determine current target element from frame counter
            ei = n_elements - 1  # clamp to last
            for i in range(n_elements):
                if frame_counter <= cum[i + 1]:
                    ei = i
                    break

            # Get player's pressed buttons for tracking
            pressed = input_handler.get_nes_state()
            relevant = pressed - {"START", "SELECT", "UP"}

            # Accumulate chord time for current element
            if ei < n_elements:
                elem_chord_times[ei][frozenset(relevant)] += scaled_interval

            # Determine bar colour based on input match
            target_btns = SYMBOL_TO_BUTTONS.get(target_symbols[ei], set())
            is_correct = (relevant == target_btns)
            fill_color = ACTION_COLOR_CORRECT if is_correct else ACTION_COLOR_ERROR

            # Update bar fills up to current position
            for i in range(n_elements):
                if frame_counter <= cum[i]:
                    break
                elif frame_counter >= cum[i + 1]:
                    # Fully past this element
                    seq_display.update_bar_fill(i, 1.0)
                else:
                    # Current element — partial fill
                    frac = (frame_counter - cum[i]) / durations_frames[i]
                    seq_display.update_bar_fill(i, frac, color=fill_color)

            prev_elem_idx = ei

            # Check exit conditions
            player_x = engine.get_player_x(info)
            if player_x >= exit_x:
                outcome = "completed"
                break
            if engine.is_death(info):
                outcome = "death"
                break

        # Render
        engine.render()
        seq_display.draw()
        win.flip()

    # Final timing
    traversal_time = core.getTime() - start_time

    # Evaluate per-element accuracy from dominant chords
    response_sequence = []
    accuracy_per_element = []
    for i in range(n_elements):
        ct = elem_chord_times[i]
        if ct:
            dom_key = max(ct, key=ct.get)
            chord = set(dom_key)
        else:
            chord = set()
        elem_dominant[i] = chord
        sym = buttons_to_symbol(chord) if chord else "NA"
        response_sequence.append(sym)

        target_btns = SYMBOL_TO_BUTTONS.get(target_symbols[i], set())
        accuracy_per_element.append(1 if chord == target_btns else 0)

    accuracy_trial = 1 if all(a == 1 for a in accuracy_per_element) else 0

    # Distance reached
    player_x = engine.get_player_x()
    # Use exit_x as the scene length reference
    distance = max(0.0, player_x / exit_x) if exit_x > 0 else 0.0

    if verbose():
        print(f"  [EXEC] outcome={outcome} | time={traversal_time:.2f}s | "
              f"accuracy={sum(accuracy_per_element)}/{n_elements} | "
              f"x={player_x}/{exit_x}")

    return {
        "outcome": outcome,
        "traversal_time": traversal_time,
        "distance_reached": min(1.0, distance),
        "target_sequence": target_symbols,
        "response_sequence": response_sequence,
        "accuracy_per_element": accuracy_per_element,
        "accuracy_trial": accuracy_trial,
    }


