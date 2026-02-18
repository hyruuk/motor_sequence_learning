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
    SPEED_FACTOR,
)
from smb_ssl_task.config import verbose
from smb_ssl_task.scenes import (
    get_scenes,
    get_canonical_sequence,
    get_canonical_sequence_from_bk2,
    get_canonical_sequence_source,
    get_clip_savestate_path,
)
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


def _show_try_again(win, duration=1.5):
    """Show a brief 'Try again' message between repeat attempts."""
    from psychopy import visual as _vis
    msg = _vis.TextStim(win, text="Try again...", height=40, color=(1, 1, 0))
    msg.draw()
    win.flip()
    core.wait(duration)


def _run_single_run_msp(win, input_handler, seq_display, pacing_line,
                        all_scenes, trained_ids, run_number, logger, paced,
                        mode, advanced_config=None):
    """Execute one functional run in MSP mode."""
    _adv = advanced_config and advanced_config.enabled
    _adv_bk2 = advanced_config.selected_bk2 if advanced_config else None
    _repeat = advanced_config.repeat_until_passed if advanced_config else False

    all_scene_ids = sorted(all_scenes.keys())
    items = _generate_run_trials(all_scene_ids, SCAN_REPS_PER_SEQ, SCAN_REST_PERIODS)

    trial_counter = 0

    for item in items:
        if item[0] == "rest":
            show_fixation_rest(win, SCAN_REST_DURATION)
            continue

        _, scene_id, execution_number = item
        trial_counter += 1
        repeat_attempt = 0
        passed = False

        while not passed:
            repeat_attempt += 1
            if repeat_attempt > 1:
                _show_try_again(win)

            if _adv_bk2 and repeat_attempt == 1:
                action_seq = get_canonical_sequence_from_bk2(_adv_bk2)
            elif repeat_attempt == 1:
                action_seq = get_canonical_sequence(scene_id)
            target_symbols = [s for s, _ in action_seq]
            source_clip = get_canonical_sequence_source(scene_id)
            condition = "trained" if scene_id in trained_ids else "untrained"
            if verbose():
                print(f"[MSP] Run {run_number} | Scene: {scene_id} | BK2: {source_clip or 'placeholder'} | Sequence: {' '.join(target_symbols)} | Attempt: {repeat_attempt}")

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
                advanced_mode=_adv,
                source_bk2=_adv_bk2,
                repeat_attempt=repeat_attempt,
            )

            # Check pass criterion
            passed = result["accuracy_trial"] == 1
            if not _repeat:
                passed = True

        # --- ITI ---
        win.flip()
        core.wait(SCAN_ITI)

    return True


def _run_single_run_gameplay(win, input_handler, engine,
                              all_scenes, trained_ids, run_number,
                              logger, mode, advanced_config=None):
    """Execute one functional run in gameplay mode."""
    _adv = advanced_config and advanced_config.enabled
    _adv_bk2 = advanced_config.selected_bk2 if advanced_config else None
    _repeat = advanced_config.repeat_until_passed if advanced_config else False

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
        repeat_attempt = 0
        passed = False

        while not passed:
            repeat_attempt += 1
            if repeat_attempt > 1:
                _show_try_again(win)

            # Select a BK2 clip and get matching savestate
            if _adv_bk2 and repeat_attempt == 1:
                action_seq = get_canonical_sequence_from_bk2(_adv_bk2)
            elif repeat_attempt == 1:
                action_seq = get_canonical_sequence(scene_id)
            source_clip = get_canonical_sequence_source(scene_id)
            clip_state = get_clip_savestate_path(scene_id)
            if verbose():
                print(f"[GAMEPLAY] Run {run_number} | Scene: {scene_id} | BK2: {source_clip or 'placeholder'} | Attempt: {repeat_attempt}")

            # --- PREP phase: brief fixation ---
            prep_timer = core.CountdownTimer(SCAN_PREP_DURATION)
            while prep_timer.getTime() > 0:
                if input_handler.check_escape():
                    return False
                win.flip()

            # --- Load and execute gameplay ---
            engine.load_scene(scene_id, scene_info, state_path=clip_state)
            result = execute_gameplay_scan_trial(
                win, input_handler, engine, scene_info,
                duration=SCAN_EXECUTION_DURATION,
                speed_factor=SPEED_FACTOR,
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
                advanced_mode=_adv,
                source_bk2=_adv_bk2,
                repeat_attempt=repeat_attempt,
            )

            # Brief point feedback
            show_trial_points(win, points, 0.3)

            # Check pass criterion
            passed = result["outcome"] == "completed"
            if not _repeat:
                passed = True

        # --- ITI ---
        win.flip()
        core.wait(SCAN_ITI)

    return True


def run_scan_session(win, input_handler, participant_id, group,
                     session_number, paced=True, mode="gameplay",
                     engine=None, advanced_config=None):
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
    advanced_config : AdvancedConfig or None
    """
    trained_scenes, untrained_scenes = get_scenes(group)
    all_scenes = {**trained_scenes, **untrained_scenes}
    trained_ids = set(trained_scenes.keys())

    # --- Advanced mode: override scene list ---
    if advanced_config and advanced_config.enabled and advanced_config.selected_scene_id:
        sid = advanced_config.selected_scene_id
        if advanced_config.selected_scene_info:
            all_scenes = {sid: advanced_config.selected_scene_info}
        elif sid in all_scenes:
            all_scenes = {sid: all_scenes[sid]}
        trained_ids = trained_ids & {sid}

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
                    mode, advanced_config=advanced_config,
                )
            else:
                completed = _run_single_run_gameplay(
                    win, input_handler, engine,
                    all_scenes, trained_ids, run_num,
                    logger, mode, advanced_config=advanced_config,
                )

            if not completed:
                return

            if run_num < SCAN_N_RUNS:
                show_run_rest(win, run_num, SCAN_N_RUNS)

    finally:
        logger.close()
