"""
Entry point for the SMB Scene Sequence Learning (SSL) task.

Presents a GUI dialog to configure the session, then launches the
appropriate task (training, test, scan, or pre-training) in either
MSP or gameplay mode.

Can be run as:
    python -m smb_ssl_task
"""

import argparse
import json
import os
import traceback

from psychopy import visual, gui, core, event

from smb_ssl_task import config
from smb_ssl_task.config import (
    FULLSCREEN,
    MONITOR_NAME,
    BACKGROUND_COLOR,
    SETTINGS_FILE,
    GAME_NAME,
    set_verbose,
)
from smb_ssl_task.advanced_gui import AdvancedConfig
from smb_ssl_task.scenes import set_scenes_path


def _detect_screen_resolution():
    """Detect native screen resolution via pyglet."""
    try:
        import pyglet
        display = pyglet.canvas.Display()
        screen = display.get_default_screen()
        return f"{screen.width}x{screen.height}"
    except Exception:
        return "1920x1080"
from smb_ssl_task.input_handler import InputHandler
from smb_ssl_task.task_training import run_training_session
from smb_ssl_task.task_test import run_test_session
from smb_ssl_task.task_pretrain import run_pretrain_session
from smb_ssl_task.task_scan import run_scan_session


def _load_settings():
    """Load persistent settings from JSON file."""
    if os.path.isfile(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {}


def _save_settings(settings):
    """Save persistent settings to JSON file."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def _register_retro_integration(retro_integration_dir):
    """Register a custom retro integration directory.

    The directory must contain a ``SuperMarioBros-Nes/`` subfolder with the
    ROM (``rom.nes``), ``data.json``, ``scenario.json``, state files, etc.

    Parameters
    ----------
    retro_integration_dir : str
        Path to the directory that *contains* ``SuperMarioBros-Nes/``.

    Raises
    ------
    FileNotFoundError
        If the directory or expected game subfolder is missing.
    """
    import stable_retro

    if not retro_integration_dir or not os.path.isdir(retro_integration_dir):
        raise FileNotFoundError(
            f"Retro integration directory not found: '{retro_integration_dir}'\n"
            "Set the path to the directory containing SuperMarioBros-Nes/."
        )

    game_dir = os.path.join(retro_integration_dir, "SuperMarioBros-Nes")
    if not os.path.isdir(game_dir):
        raise FileNotFoundError(
            f"SuperMarioBros-Nes/ not found inside '{retro_integration_dir}'.\n"
            "The directory should contain a SuperMarioBros-Nes/ subfolder\n"
            "with rom.nes, data.json, scenario.json, and state files."
        )

    rom_file = os.path.join(game_dir, "rom.nes")
    if not os.path.isfile(rom_file):
        raise FileNotFoundError(
            f"rom.nes not found in '{game_dir}'.\n"
            "The SuperMarioBros-Nes/ folder must contain the ROM file."
        )

    stable_retro.data.add_custom_integration(retro_integration_dir)


def main():
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="SMB-SSL Task")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose terminal output (per-element debug)")
    args, _ = parser.parse_known_args()  # ignore PsychoPy's own args
    set_verbose(args.verbose)

    # --- Load persistent settings ---
    settings = _load_settings()

    # --- Detect default screen resolution ---
    default_resolution = settings.get(
        "screen_resolution", _detect_screen_resolution()
    )

    # --- Session configuration dialog ---
    info = {
        "Participant ID": settings.get("participant_id", "01"),
        "Group": [1, 2],
        "Mode": ["msp", "gameplay"],
        "Session type": [
            "training", "test",
            "scan_paced", "scan_fullspeed", "pretrain",
        ],
        "Session number": 1,
        "Blocks / Reps": 4,
        "Screen resolution": default_resolution,
        "Scenes dataset dir": settings.get("scenes_dataset_dir", ""),
        "Retro integration dir": settings.get("retro_integration_dir", ""),
        "Advanced mode": settings.get("advanced_mode", False),
    }

    dlg = gui.DlgFromDict(
        dictionary=info,
        title="SMB-SSL Task Configuration",
        order=[
            "Participant ID",
            "Group",
            "Mode",
            "Session type",
            "Session number",
            "Blocks / Reps",
            "Screen resolution",
            "Scenes dataset dir",
            "Retro integration dir",
            "Advanced mode",
        ],
        tip={
            "Screen resolution": (
                "WIDTHxHEIGHT in pixels (e.g. 1920x1080, 2880x1920). "
                "Auto-detected from your display."
            ),
            "Scenes dataset dir": (
                "Path to the mario.scenes dataset root "
                "(containing sub-01/, sub-02/, etc. with gamelogs) "
                "— used by both MSP and gameplay modes"
            ),
            "Retro integration dir": (
                "Directory containing SuperMarioBros-Nes/ "
                "(with rom.nes, data.json, etc.) — gameplay mode only"
            ),
            "Advanced mode": (
                "Open filter dialogs to pick a specific BK2 clip "
                "and optionally repeat until passed"
            ),
        },
    )

    if not dlg.OK:
        core.quit()

    participant_id = str(info["Participant ID"])
    group = int(info["Group"])
    mode = info["Mode"]
    session_type = info["Session type"]
    session_number = int(info["Session number"])
    n_blocks_or_reps = int(info["Blocks / Reps"])
    scenes_dataset_dir = str(info["Scenes dataset dir"]).strip()
    retro_integration_dir = str(info["Retro integration dir"]).strip()

    # --- Parse screen resolution ---
    resolution_str = str(info["Screen resolution"]).strip()
    try:
        w, h = resolution_str.lower().split("x")
        screen_size = (int(w), int(h))
    except (ValueError, AttributeError):
        screen_size = (1920, 1080)

    advanced_mode_checked = bool(info["Advanced mode"])

    # --- Save settings for next run ---
    settings["participant_id"] = participant_id
    settings["screen_resolution"] = resolution_str
    settings["scenes_dataset_dir"] = scenes_dataset_dir
    settings["retro_integration_dir"] = retro_integration_dir
    settings["advanced_mode"] = advanced_mode_checked
    _save_settings(settings)

    # --- Advanced mode dialogs ---
    advanced_config = AdvancedConfig(enabled=False)
    if advanced_mode_checked:
        from smb_ssl_task.advanced_gui import run_advanced_dialogs
        advanced_config = run_advanced_dialogs(scenes_dataset_dir)

    # --- Apply config overrides from advanced dialog ---
    if advanced_config.enabled:
        overrides = advanced_config.get_config_overrides()
        if overrides:
            config.apply_overrides(overrides)

    # --- Set scenes dataset path (needed by both MSP and gameplay modes) ---
    if scenes_dataset_dir and os.path.isdir(scenes_dataset_dir):
        set_scenes_path(scenes_dataset_dir)

    # --- Validate paths and register retro integration if gameplay mode ---
    # IMPORTANT: The retro emulator must be created BEFORE the PsychoPy
    # window because stable_retro.make() + env.reset() touches the GL
    # context and invalidates any previously compiled shaders.
    retro_env = None
    if mode == "gameplay":
        if not scenes_dataset_dir or not os.path.isdir(scenes_dataset_dir):
            err_dlg = gui.Dlg(title="Scenes Dataset Error")
            err_dlg.addText(
                f"Scenes dataset directory not found: '{scenes_dataset_dir}'\n"
                "Set the path to the mario.scenes dataset root\n"
                "(containing sub-01/, sub-02/, etc. with gamelogs)."
            )
            err_dlg.show()
            core.quit()

        try:
            _register_retro_integration(retro_integration_dir)
        except FileNotFoundError as e:
            err_dlg = gui.Dlg(title="Retro Integration Error")
            err_dlg.addText(str(e))
            err_dlg.show()
            core.quit()

        # Create the retro environment now, before the PsychoPy window.
        # render_mode=None prevents stable_retro from opening its own
        # viewer window (the default is "human" which creates one).
        import stable_retro
        retro_env = stable_retro.make(
            game=GAME_NAME,
            inttype=stable_retro.data.Integrations.ALL,
            use_restricted_actions=stable_retro.Actions.ALL,
            render_mode=None,
        )
        retro_env.reset()  # Trigger emulator init now

    # --- Create window (after retro env, so shaders compile cleanly) ---
    win = visual.Window(
        size=screen_size,
        fullscr=FULLSCREEN,
        monitor=MONITOR_NAME,
        color=BACKGROUND_COLOR,
        units="pix",
        allowGUI=False,
        checkTiming=False,  # Skip frame rate measurement (hangs after retro init)
    )

    # Flush any stale events left over from the GUI dialog
    event.clearEvents()

    input_handler = InputHandler(win)

    # --- Create GameEngine if gameplay mode ---
    engine = None
    if retro_env is not None:
        from smb_ssl_task.game import GameEngine
        engine = GameEngine(win, scenes_dataset_dir, env=retro_env)


    try:
        if session_type == "training":
            run_training_session(
                win=win,
                input_handler=input_handler,
                participant_id=participant_id,
                group=group,
                session_number=session_number,
                n_blocks=n_blocks_or_reps,
                mode=mode,
                engine=engine,
                advanced_config=advanced_config,
            )
        elif session_type == "test":
            run_test_session(
                win=win,
                input_handler=input_handler,
                participant_id=participant_id,
                group=group,
                session_number=session_number,
                n_reps_per_scene=n_blocks_or_reps,
                mode=mode,
                engine=engine,
                advanced_config=advanced_config,
            )
        elif session_type == "scan_paced":
            run_scan_session(
                win=win,
                input_handler=input_handler,
                participant_id=participant_id,
                group=group,
                session_number=session_number,
                paced=True,
                mode=mode,
                engine=engine,
                advanced_config=advanced_config,
            )
        elif session_type == "scan_fullspeed":
            run_scan_session(
                win=win,
                input_handler=input_handler,
                participant_id=participant_id,
                group=group,
                session_number=session_number,
                paced=False,
                mode=mode,
                engine=engine,
                advanced_config=advanced_config,
            )
        elif session_type == "pretrain":
            run_pretrain_session(
                win=win,
                input_handler=input_handler,
                participant_id=participant_id,
                group=group,
                session_number=session_number,
                mode=mode,
                engine=engine,
                advanced_config=advanced_config,
            )
        else:
            raise ValueError(f"Unknown session type: {session_type}")
    except Exception:
        traceback.print_exc()
    finally:
        if engine is not None:
            engine.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
