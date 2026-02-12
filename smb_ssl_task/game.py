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

import numpy as np
import pyglet.gl as GL
from psychopy import core

from smb_ssl_task.config import (
    GAME_RENDER_SIZE,
    GAMEPLAY_MAX_DURATION,
    GAME_NAME,
    EMULATOR_FPS,
)
from smb_ssl_task.scenes import get_savestate_path

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

    def load_scene(self, scene_id, scene_info):
        """Load a savestate for a scene and reset the environment.

        Parameters
        ----------
        scene_id : str
        scene_info : dict
            Scene dict with entry, exit, layout fields.
        """
        state_path = get_savestate_path(scene_id, scenes_path=self.scenes_path)
        if state_path is None:
            raise FileNotFoundError(
                f"No savestate found for scene '{scene_id}'"
            )

        with open(state_path, "rb") as f:
            state_data = f.read()

        self.env.initial_state = state_data
        obs, _info = self.env.reset()
        self._renderer.update(obs)
        self._current_info = {}

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
        """Check if player has died (player_state == 11 or lives decreased)."""
        if info is None:
            info = self._current_info
        player_state = info.get("playerState", 0)
        return player_state == 11

    def close(self):
        """Close the retro environment and free GL resources."""
        self._renderer.cleanup()
        if self.env is not None:
            self.env.close()
            self.env = None


def execute_gameplay_trial(win, input_handler, engine, scene_info,
                           max_duration=GAMEPLAY_MAX_DURATION):
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

    Returns
    -------
    dict or None
        None if escape pressed. Otherwise:
        - outcome: "completed", "death", or "timeout"
        - traversal_time: float
        - distance_reached: float (fraction of scene traversed)
    """
    scene_length = scene_info["exit"] - scene_info["entry"]

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

        # Step the emulator at 60fps; on faster displays, re-render last frame
        if now >= next_step_time:
            action = input_handler.get_action_array()
            info = engine.step(action)
            next_step_time += _FRAME_INTERVAL

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
                                duration):
    """Fixed-time gameplay execution for scan sessions.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    engine : GameEngine
    scene_info : dict
    duration : float
        Fixed execution window in seconds.

    Returns
    -------
    dict or None
    """
    scene_length = scene_info["exit"] - scene_info["entry"]

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

        # Step emulator at 60fps while game is active
        if game_active and now >= next_step_time:
            action = input_handler.get_action_array()
            info = engine.step(action)
            next_step_time += _FRAME_INTERVAL

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


