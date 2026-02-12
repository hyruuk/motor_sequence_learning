"""
Scan session logic for the SMB SSL task.

Supports paced and full-speed modes, both MSP and gameplay.
8 functional runs per session. 12 scenes x 6 reps = 72 trials per run.
Scenes appear in consecutive pairs (same scene back-to-back).
5 rest periods (10s fixation) randomly inserted per run.
"""

import random

from psychopy import core, event

from smb_ssl_task.config import (
    ESCAPE_KEY,
    SCANNER_TRIGGER_KEY,
    SCAN_PREP_DURATION,
    SCAN_EXECUTION_DURATION,
    SCAN_ITI,
    SCAN_REPS_PER_SEQ,
    SCAN_N_RUNS,
    SCAN_REST_PERIODS,
    SCAN_REST_DURATION,
    SCAN_POINTS_CORRECT,
    SCAN_POINTS_ERROR,
)
from smb_ssl_task.scenes import get_scenes, get_canonical_sequence, get_canonical_sequence_source
from smb_ssl_task.msp import (
    ActionSequenceDisplay,
    collect_msp_scan_execution,
)
from smb_ssl_task.game import execute_gameplay_scan_trial
from smb_ssl_task.display import (
    PacingLine,
    show_instructions,
    show_run_rest,
    show_fixation_rest,
    show_waiting_for_scanner,
    show_trial_points,
)
from smb_ssl_task.data_logging import DataLogger


def _generate_run_trials(all_scene_ids, reps_per_seq, n_rest_periods):
    """Generate trial list for one run.

    12 scenes x reps_per_seq reps = 72 trials, arranged as
    consecutive pairs (each scene repeated twice in a row).
    n_rest_periods rest markers inserted at random positions.

    Returns
    -------
    list
        Items are ("trial", scene_id, execution_number) or ("rest",).
    """
    # Build pairs: each scene appears reps_per_seq/2 times as a pair
    n_pairs = reps_per_seq // 2
    pairs = []
    for scene_id in all_scene_ids:
        for _ in range(n_pairs):
            pairs.append(scene_id)

    random.shuffle(pairs)

    # Expand pairs into trial items
    items = []
    for scene_id in pairs:
        items.append(("trial", scene_id, 1))
        items.append(("trial", scene_id, 2))

    # Insert rest periods between pairs (even indices)
    pair_boundaries = list(range(0, len(items) + 1, 2))
    interior = pair_boundaries[1:-1]
    rest_positions = sorted(
        random.sample(interior, min(n_rest_periods, len(interior))),
        reverse=True,
    )
    for pos in rest_positions:
        items.insert(pos, ("rest",))

    return items


def _run_single_run_msp(win, input_handler, seq_display, pacing_line,
                        all_scenes, trained_ids, run_number, logger, paced,
                        mode):
    """Execute one functional run in MSP mode."""
    all_scene_ids = sorted(all_scenes.keys())
    items = _generate_run_trials(all_scene_ids, SCAN_REPS_PER_SEQ, SCAN_REST_PERIODS)

    trial_counter = 0

    for item in items:
        if item[0] == "rest":
            show_fixation_rest(win, SCAN_REST_DURATION)
            continue

        _, scene_id, execution_number = item
        trial_counter += 1
        action_seq = get_canonical_sequence(scene_id)
        target_symbols = [s for s, _ in action_seq]
        source_clip = get_canonical_sequence_source(scene_id)
        condition = "trained" if scene_id in trained_ids else "untrained"
        print(f"[MSP] Run {run_number} | Scene: {scene_id} | Source: {source_clip or 'placeholder'} | Sequence: {' '.join(target_symbols)}")

        # --- PREP phase: show sequence ---
        seq_display.show(action_seq)
        if paced:
            pacing_line.reset()
        else:
            pacing_line.hide()
        seq_display.draw()
        win.flip()

        # Wait for prep duration
        prep_timer = core.CountdownTimer(SCAN_PREP_DURATION)
        abort = False
        while prep_timer.getTime() > 0:
            if input_handler.check_escape():
                abort = True
                break
            seq_display.draw()
            if paced:
                pacing_line.draw()
            win.flip()

        if abort:
            return False

        # --- GO signal + EXECUTION phase ---
        if paced:
            pacing_line.reset()
        else:
            pacing_line.show_go_cue()

        result = collect_msp_scan_execution(
            win, input_handler, seq_display,
            action_seq,
            duration=SCAN_EXECUTION_DURATION,
            pacing_line=pacing_line if paced else None,
        )

        if result is None:
            return False

        pacing_line.hide()

        logger.log_execution(
            block_number=1,
            trial_number=trial_counter,
            scene_id=scene_id,
            mode=mode,
            execution_number=execution_number,
            condition=condition,
            run_number=run_number,
            target_sequence=target_symbols,
            response_sequence=result["response_sequence"],
            target_durations=result["target_durations"],
            response_durations=result["response_durations"],
            accuracy_per_element=result["accuracy_per_element"],
            accuracy_trial=result["accuracy_trial"],
            movement_time=result["movement_time"],
            inter_element_intervals=result["inter_element_intervals"],
            points_awarded=result["points"],
        )

        # --- ITI ---
        win.flip()
        core.wait(SCAN_ITI)

    return True


def _run_single_run_gameplay(win, input_handler, engine,
                              all_scenes, trained_ids, run_number,
                              logger, mode):
    """Execute one functional run in gameplay mode."""
    all_scene_ids = sorted(all_scenes.keys())
    items = _generate_run_trials(all_scene_ids, SCAN_REPS_PER_SEQ, SCAN_REST_PERIODS)

    trial_counter = 0

    for item in items:
        if item[0] == "rest":
            show_fixation_rest(win, SCAN_REST_DURATION)
            continue

        _, scene_id, execution_number = item
        trial_counter += 1
        scene_info = all_scenes[scene_id]
        condition = "trained" if scene_id in trained_ids else "untrained"

        # --- PREP phase: brief fixation ---
        prep_timer = core.CountdownTimer(SCAN_PREP_DURATION)
        while prep_timer.getTime() > 0:
            if input_handler.check_escape():
                return False
            win.flip()

        # --- Load and execute gameplay ---
        engine.load_scene(scene_id, scene_info)
        result = execute_gameplay_scan_trial(
            win, input_handler, engine, scene_info,
            duration=SCAN_EXECUTION_DURATION,
        )

        if result is None:
            return False

        # Compute points
        points = SCAN_POINTS_CORRECT if result["outcome"] == "completed" else SCAN_POINTS_ERROR

        logger.log_execution(
            block_number=1,
            trial_number=trial_counter,
            scene_id=scene_id,
            mode=mode,
            execution_number=execution_number,
            condition=condition,
            run_number=run_number,
            outcome=result["outcome"],
            traversal_time=result["traversal_time"],
            distance_reached=result["distance_reached"],
            points_awarded=points,
        )

        # Brief point feedback
        show_trial_points(win, points, 0.3)

        # --- ITI ---
        win.flip()
        core.wait(SCAN_ITI)

    return True


def run_scan_session(win, input_handler, participant_id, group,
                     session_number, paced=True, mode="gameplay",
                     engine=None):
    """Run a complete scan session (8 runs).

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    participant_id : str
    group : int
    session_number : int
    paced : bool
        True for pacing line, False for full speed.
    mode : str
        "msp" or "gameplay"
    engine : GameEngine or None
    """
    trained_scenes, untrained_scenes = get_scenes(group)
    all_scenes = {**trained_scenes, **untrained_scenes}
    trained_ids = set(trained_scenes.keys())

    session_type = "scan_paced" if paced else "scan_fullspeed"
    logger = DataLogger(participant_id, group, session_type, session_number)

    seq_display = None
    pacing_line = None
    if mode == "msp":
        seq_display = ActionSequenceDisplay(win)
        pacing_line = PacingLine(win, seq_display.total_width)

    mode_desc = "paced" if paced else "full speed"
    if mode == "msp":
        show_instructions(
            win,
            f"SCAN SESSION ({mode_desc}, MSP Mode)\n\n"
            "You will see button sequences with duration bars.\n"
            "Press and hold the correct buttons for the indicated duration,\n"
            "then release to advance to the next element.\n\n"
            "Controls:\n"
            "\u2190 \u2192 = Arrow keys    X = Jump    Z = Run\n\n"
            "Each sequence will appear twice in a row.\n\n"
            "Press any key to begin.",
        )
    else:
        show_instructions(
            win,
            f"SCAN SESSION ({mode_desc}, Gameplay Mode)\n\n"
            "You will play through short Mario scenes.\n"
            "Use the arrow keys to move and X to jump, Z to run.\n\n"
            "Each scene will appear twice in a row.\n\n"
            "Press any key to begin.",
        )

    try:
        for run_num in range(1, SCAN_N_RUNS + 1):
            show_waiting_for_scanner(win)
            event.waitKeys(keyList=[SCANNER_TRIGGER_KEY])

            if mode == "msp":
                completed = _run_single_run_msp(
                    win, input_handler, seq_display, pacing_line,
                    all_scenes, trained_ids, run_num, logger, paced,
                    mode,
                )
            else:
                completed = _run_single_run_gameplay(
                    win, input_handler, engine,
                    all_scenes, trained_ids, run_num,
                    logger, mode,
                )

            if not completed:
                return

            if run_num < SCAN_N_RUNS:
                show_run_rest(win, run_num, SCAN_N_RUNS)

    finally:
        logger.close()
