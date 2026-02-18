"""
Pre-training session for the SMB SSL task.

Familiarization with the apparatus using scenes not in the experimental set.
Supports both MSP and gameplay modes.
"""

import random

from psychopy import core

from smb_ssl_task.config import (
    PRETRAIN_REPS_PER_SCENE,
    INTER_EXECUTION_INTERVAL,
    INTER_TRIAL_INTERVAL,
    SPEED_FACTOR,
)
from smb_ssl_task.config import verbose
from smb_ssl_task.scenes import (
    get_pretrain_scenes,
    get_canonical_sequence,
    get_canonical_sequence_from_bk2,
    get_canonical_sequence_source,
    get_clip_savestate_path,
)
from smb_ssl_task.msp import ActionSequenceDisplay, collect_msp_execution
from smb_ssl_task.game import execute_gameplay_trial
from smb_ssl_task.display import show_instructions
from smb_ssl_task.data_logging import DataLogger


def _show_try_again(win, duration=1.5):
    """Show a brief 'Try again' message between repeat attempts."""
    from psychopy import visual as _vis
    msg = _vis.TextStim(win, text="Try again...", height=40, color=(1, 1, 0))
    msg.draw()
    win.flip()
    core.wait(duration)


def run_pretrain_session(win, input_handler, participant_id, group,
                         session_number, mode="gameplay", engine=None,
                         advanced_config=None):
    """Pre-training: familiarization with apparatus.

    Uses scenes not in the experimental set.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    participant_id : str
    group : int
    session_number : int
    mode : str
        "msp" or "gameplay"
    engine : GameEngine or None
        Required if mode="gameplay".
    advanced_config : AdvancedConfig or None
    """
    pretrain_scenes = get_pretrain_scenes()
    scene_dict = {s["id"]: s for s in pretrain_scenes}

    # --- Advanced mode: override scene list ---
    if advanced_config and advanced_config.enabled and advanced_config.selected_scene_id:
        sid = advanced_config.selected_scene_id
        if advanced_config.selected_scene_info:
            scene_dict = {sid: advanced_config.selected_scene_info}
        elif sid in scene_dict:
            scene_dict = {sid: scene_dict[sid]}
        pretrain_scenes = [scene_dict[sid]]

    logger = DataLogger(participant_id, group, "pretrain", session_number)

    if mode == "msp":
        seq_display = ActionSequenceDisplay(win)

    if mode == "msp":
        show_instructions(
            win,
            "PRE-TRAINING SESSION (MSP Mode)\n\n"
            "You will see a sequence of button symbols with duration bars.\n"
            "Press and hold the correct buttons for the indicated duration,\n"
            "then release to advance to the next element.\n\n"
            "Controls:\n"
            "\u2190 \u2192 = Arrow keys    X = Jump    Z = Run\n\n"
            "The symbols will remain visible for both executions.\n\n"
            "Press any key to begin.",
        )
    else:
        show_instructions(
            win,
            "PRE-TRAINING SESSION (Gameplay Mode)\n\n"
            "You will play through short Mario scenes.\n"
            "Use the arrow keys to move and X to jump, Z to run.\n\n"
            "Each scene will be played twice in a row.\n\n"
            "Press any key to begin.",
        )

    # Build trial list
    trial_list = []
    for scene in pretrain_scenes:
        trial_list.extend([scene["id"]] * PRETRAIN_REPS_PER_SCENE)
    random.shuffle(trial_list)

    # Advanced mode logging helpers
    _adv = advanced_config and advanced_config.enabled
    _adv_bk2 = advanced_config.selected_bk2 if advanced_config else None
    _repeat = advanced_config.repeat_until_passed if advanced_config else False

    trial_counter = 0

    try:
        for scene_id in trial_list:
            trial_counter += 1
            scene_info = scene_dict[scene_id]
            repeat_attempt = 0
            passed = False

            while not passed:
                repeat_attempt += 1

                if repeat_attempt > 1:
                    _show_try_again(win)

                if mode == "msp":
                    # --- Sequence selection (advanced override) ---
                    if _adv_bk2 and repeat_attempt == 1:
                        action_seq = get_canonical_sequence_from_bk2(_adv_bk2)
                    elif repeat_attempt == 1:
                        action_seq = get_canonical_sequence(scene_id)
                    target_symbols = [s for s, _ in action_seq]
                    source_clip = get_canonical_sequence_source(scene_id)
                    if verbose():
                        print(f"[MSP] Scene: {scene_id} | BK2: {source_clip or 'placeholder'} | Sequence: {' '.join(target_symbols)} | Attempt: {repeat_attempt}")

                    # --- Execution 1: visible ---
                    exec1 = collect_msp_execution(
                        win, input_handler, seq_display,
                        action_seq, visible=True,
                    )
                    if exec1 is None:
                        return

                    logger.log_execution(
                        block_number=1,
                        trial_number=trial_counter,
                        scene_id=scene_id,
                        mode=mode,
                        execution_number=1,
                        condition="pretrain",
                        target_sequence=target_symbols,
                        response_sequence=exec1["response_sequence"],
                        target_durations=exec1["target_durations"],
                        response_durations=exec1["response_durations"],
                        accuracy_per_element=exec1["accuracy_per_element"],
                        accuracy_trial=exec1["accuracy_trial"],
                        movement_time=exec1["movement_time"],
                        inter_element_intervals=exec1["inter_element_intervals"],
                        advanced_mode=_adv,
                        source_bk2=_adv_bk2,
                        repeat_attempt=repeat_attempt,
                    )

                    win.flip()
                    core.wait(INTER_EXECUTION_INTERVAL)

                    # --- Execution 2: visible (pretrain) ---
                    seq_display.reset()
                    exec2 = collect_msp_execution(
                        win, input_handler, seq_display,
                        action_seq, visible=True,
                    )
                    if exec2 is None:
                        return

                    logger.log_execution(
                        block_number=1,
                        trial_number=trial_counter,
                        scene_id=scene_id,
                        mode=mode,
                        execution_number=2,
                        condition="pretrain",
                        target_sequence=target_symbols,
                        response_sequence=exec2["response_sequence"],
                        target_durations=exec2["target_durations"],
                        response_durations=exec2["response_durations"],
                        accuracy_per_element=exec2["accuracy_per_element"],
                        accuracy_trial=exec2["accuracy_trial"],
                        movement_time=exec2["movement_time"],
                        inter_element_intervals=exec2["inter_element_intervals"],
                        advanced_mode=_adv,
                        source_bk2=_adv_bk2,
                        repeat_attempt=repeat_attempt,
                    )

                    # Check pass criterion
                    passed = exec2["accuracy_trial"] == 1

                else:  # gameplay mode
                    if _adv_bk2 and repeat_attempt == 1:
                        action_seq = get_canonical_sequence_from_bk2(_adv_bk2)
                    elif repeat_attempt == 1:
                        action_seq = get_canonical_sequence(scene_id)
                    source_clip = get_canonical_sequence_source(scene_id)
                    clip_state = get_clip_savestate_path(scene_id)
                    if verbose():
                        print(f"[GAMEPLAY] Scene: {scene_id} | BK2: {source_clip or 'placeholder'} | Attempt: {repeat_attempt}")
                    engine.load_scene(scene_id, scene_info, state_path=clip_state)

                    # --- Execution 1: normal play ---
                    exec1 = execute_gameplay_trial(
                        win, input_handler, engine, scene_info,
                        speed_factor=SPEED_FACTOR,
                    )
                    if exec1 is None:
                        return

                    logger.log_execution(
                        block_number=1,
                        trial_number=trial_counter,
                        scene_id=scene_id,
                        mode=mode,
                        execution_number=1,
                        condition="pretrain",
                        outcome=exec1["outcome"],
                        traversal_time=exec1["traversal_time"],
                        distance_reached=exec1["distance_reached"],
                        advanced_mode=_adv,
                        source_bk2=_adv_bk2,
                        repeat_attempt=repeat_attempt,
                    )

                    win.flip()
                    core.wait(INTER_EXECUTION_INTERVAL)

                    # --- Execution 2: immediate replay (same savestate) ---
                    engine.load_scene(scene_id, scene_info, state_path=clip_state)
                    exec2 = execute_gameplay_trial(
                        win, input_handler, engine, scene_info,
                        speed_factor=SPEED_FACTOR,
                    )
                    if exec2 is None:
                        return

                    logger.log_execution(
                        block_number=1,
                        trial_number=trial_counter,
                        scene_id=scene_id,
                        mode=mode,
                        execution_number=2,
                        condition="pretrain",
                        outcome=exec2["outcome"],
                        traversal_time=exec2["traversal_time"],
                        distance_reached=exec2["distance_reached"],
                        advanced_mode=_adv,
                        source_bk2=_adv_bk2,
                        repeat_attempt=repeat_attempt,
                    )

                    # Check pass criterion
                    passed = exec2["outcome"] == "completed"

                # If not in repeat mode, always break
                if not _repeat:
                    passed = True

            # Inter-trial interval
            win.flip()
            core.wait(INTER_TRIAL_INTERVAL)

    finally:
        logger.close()
