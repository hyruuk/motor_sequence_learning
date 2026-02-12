"""
Behavioral test session for the SMB SSL task.

All 12 scenes intermixed, self-paced. No reward system.
Supports both MSP and gameplay modes.
"""

import random

from psychopy import core

from smb_ssl_task.config import (
    INTER_EXECUTION_INTERVAL,
    INTER_TRIAL_INTERVAL,
)
from smb_ssl_task.scenes import get_scenes, get_canonical_sequence, get_canonical_sequence_source
from smb_ssl_task.msp import ActionSequenceDisplay, collect_msp_execution
from smb_ssl_task.game import execute_gameplay_trial
from smb_ssl_task.display import show_instructions
from smb_ssl_task.data_logging import DataLogger


def run_test_session(win, input_handler, participant_id, group,
                     session_number, n_reps_per_scene, mode="gameplay",
                     engine=None):
    """Run a complete behavioral test session.

    All 12 scenes (6 trained + 6 untrained) intermixed.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    participant_id : str
    group : int
    session_number : int
    n_reps_per_scene : int
    mode : str
        "msp" or "gameplay"
    engine : GameEngine or None
    """
    trained_scenes, untrained_scenes = get_scenes(group)
    all_scenes = {**trained_scenes, **untrained_scenes}

    logger = DataLogger(participant_id, group, "test", session_number)

    if mode == "msp":
        seq_display = ActionSequenceDisplay(win)

    if mode == "msp":
        show_instructions(
            win,
            "BEHAVIORAL TEST (MSP Mode)\n\n"
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
            "BEHAVIORAL TEST (Gameplay Mode)\n\n"
            "You will play through short Mario scenes.\n"
            "Use the arrow keys to move and X to jump, Z to run.\n\n"
            "Each scene will be played twice in a row.\n\n"
            "Press any key to begin.",
        )

    # Build trial list: all scenes x n_reps, shuffled
    trial_list = []
    for scene_id in sorted(all_scenes.keys()):
        condition = "trained" if scene_id in trained_scenes else "untrained"
        for _ in range(n_reps_per_scene):
            trial_list.append((scene_id, condition))
    random.shuffle(trial_list)

    trial_counter = 0

    try:
        for scene_id, condition in trial_list:
            trial_counter += 1
            scene_info = all_scenes[scene_id]

            if mode == "msp":
                action_seq = get_canonical_sequence(scene_id)
                target_symbols = [s for s, _ in action_seq]
                source_clip = get_canonical_sequence_source(scene_id)
                print(f"[MSP] Scene: {scene_id} | Source: {source_clip or 'placeholder'} | Sequence: {' '.join(target_symbols)}")

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
                    condition=condition,
                    target_sequence=target_symbols,
                    response_sequence=exec1["response_sequence"],
                    target_durations=exec1["target_durations"],
                    response_durations=exec1["response_durations"],
                    accuracy_per_element=exec1["accuracy_per_element"],
                    accuracy_trial=exec1["accuracy_trial"],
                    movement_time=exec1["movement_time"],
                    inter_element_intervals=exec1["inter_element_intervals"],
                )

                win.flip()
                core.wait(INTER_EXECUTION_INTERVAL)

                # --- Execution 2: visible (test mode) ---
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
                    condition=condition,
                    target_sequence=target_symbols,
                    response_sequence=exec2["response_sequence"],
                    target_durations=exec2["target_durations"],
                    response_durations=exec2["response_durations"],
                    accuracy_per_element=exec2["accuracy_per_element"],
                    accuracy_trial=exec2["accuracy_trial"],
                    movement_time=exec2["movement_time"],
                    inter_element_intervals=exec2["inter_element_intervals"],
                )

            else:  # gameplay mode
                engine.load_scene(scene_id, scene_info)

                # --- Execution 1: normal play ---
                exec1 = execute_gameplay_trial(
                    win, input_handler, engine, scene_info,
                )
                if exec1 is None:
                    return

                logger.log_execution(
                    block_number=1,
                    trial_number=trial_counter,
                    scene_id=scene_id,
                    mode=mode,
                    execution_number=1,
                    condition=condition,
                    outcome=exec1["outcome"],
                    traversal_time=exec1["traversal_time"],
                    distance_reached=exec1["distance_reached"],
                )

                win.flip()
                core.wait(INTER_EXECUTION_INTERVAL)

                # --- Execution 2: immediate replay ---
                engine.load_scene(scene_id, scene_info)
                exec2 = execute_gameplay_trial(
                    win, input_handler, engine, scene_info,
                )
                if exec2 is None:
                    return

                logger.log_execution(
                    block_number=1,
                    trial_number=trial_counter,
                    scene_id=scene_id,
                    mode=mode,
                    execution_number=2,
                    condition=condition,
                    outcome=exec2["outcome"],
                    traversal_time=exec2["traversal_time"],
                    distance_reached=exec2["distance_reached"],
                )

            # Inter-trial interval
            win.flip()
            core.wait(INTER_TRIAL_INTERVAL)

    finally:
        logger.close()
