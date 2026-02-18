"""
Advanced mode GUI for the SMB SSL task.

When "Advanced mode" is checked in the main dialog, a comprehensive
configuration panel lets the user override any timing / training /
scanner / clip-filtering parameter.  Optionally a second dialog allows
picking a specific BK2 clip from the mario.scenes dataset.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from psychopy import gui

from smb_ssl_task import config
from smb_ssl_task.scenes import (
    get_scene_info_any,
    _scene_id_from_filename,
)


# ---------------------------------------------------------------------------
# Mapping from dialog field names to config constant names
# ---------------------------------------------------------------------------

_FIELD_TO_CONFIG = {
    # Clip filtering
    "Clip min duration (s)":        "CLIP_MIN_DURATION_SEC",
    "Clip min elements":            "CLIP_MIN_ELEMENTS",
    "Clip max elements":            "CLIP_MAX_ELEMENTS",
    # Timing
    "Execution timeout (s)":        "EXECUTION_TIMEOUT",
    "Inter-execution interval (s)": "INTER_EXECUTION_INTERVAL",
    "Inter-trial interval (s)":     "INTER_TRIAL_INTERVAL",
    "Feedback duration (s)":        "FEEDBACK_DURATION",
    "Gameplay max duration (s)":    "GAMEPLAY_MAX_DURATION",
    "Fixation duration (s)":        "FIXATION_DURATION",
    "Countdown step duration (s)":  "COUNTDOWN_STEP_DURATION",
    "Speed factor":                 "SPEED_FACTOR",
    # Training
    "Training reps per sequence":   "TRAINING_REPS_PER_SEQ",
    "Pretrain reps per scene":      "PRETRAIN_REPS_PER_SCENE",
    "Error rate threshold":         "ERROR_RATE_THRESHOLD",
    "Fast bonus fraction":          "FAST_BONUS_FRACTION",
    # Scanner
    "Scan prep duration (s)":       "SCAN_PREP_DURATION",
    "Scan execution duration (s)":  "SCAN_EXECUTION_DURATION",
    "Scan ITI (s)":                 "SCAN_ITI",
    "Scan reps per sequence":       "SCAN_REPS_PER_SEQ",
    "Scan number of runs":          "SCAN_N_RUNS",
    "Scan rest periods":            "SCAN_REST_PERIODS",
    "Scan rest duration (s)":       "SCAN_REST_DURATION",
}


@dataclass
class AdvancedConfig:
    """Carries advanced mode selections to session runners."""
    enabled: bool = False
    selected_bk2: Optional[str] = None
    selected_scene_id: Optional[str] = None
    selected_scene_info: Optional[dict] = None
    repeat_until_passed: bool = False
    # Raw overrides: {CONFIG_CONSTANT_NAME: value}  (populated by dialog)
    _overrides: dict = field(default_factory=dict)

    def get_config_overrides(self):
        """Return a dict of ``{CONFIG_CONSTANT_NAME: value}`` for all
        parameters the user changed from the defaults."""
        return dict(self._overrides)


# ---------------------------------------------------------------------------
# Dataset scanning helpers (unchanged)
# ---------------------------------------------------------------------------

def _parse_bk2_filename(filename):
    """Extract metadata from a BK2 filename.

    Expected pattern::

        sub-01_ses-001_task-mario_level-w1l1_scene-3_clip-01500000000619.bk2

    Returns
    -------
    dict or None
        Keys: subject, session, level_str, scene_num, scene_id, clip_id.
    """
    m = re.match(
        r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\d+l\d+)_scene-(\d+)_clip-(.+)\.bk2$",
        filename,
    )
    if not m:
        return None
    return {
        "subject": m.group(1),
        "session": m.group(2),
        "level_str": m.group(3),
        "scene_num": int(m.group(4)),
        "scene_id": f"{m.group(3)}s{m.group(4)}",
        "clip_id": m.group(5),
    }


def _get_outcome(bk2_path):
    """Read outcome from the companion _summary.json (or 'unknown')."""
    summary_path = bk2_path.replace(".bk2", "_summary.json")
    try:
        with open(summary_path) as f:
            meta = json.load(f)
        return meta.get("Outcome", "unknown")
    except (OSError, json.JSONDecodeError):
        return "unknown"


def scan_dataset(scenes_path):
    """Walk the mario.scenes dataset and build a structured index.

    Returns
    -------
    dict with keys:
        clips : list[dict]
            Each dict: path, filename, subject, session, level_str,
            scene_num, scene_id, clip_id, outcome.
        subjects : sorted list of unique subject strings
        levels : sorted list of unique level strings (e.g. "w1l1")
        scene_ids : sorted list of unique scene_id strings
    """
    clips = []
    subjects = set()
    levels = set()
    scene_ids = set()

    for sub_dir in sorted(os.listdir(scenes_path)):
        if not sub_dir.startswith("sub-"):
            continue
        sub_path = os.path.join(scenes_path, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        for ses_dir in sorted(os.listdir(sub_path)):
            if not ses_dir.startswith("ses-"):
                continue
            gamelogs_dir = os.path.join(sub_path, ses_dir, "gamelogs")
            if not os.path.isdir(gamelogs_dir):
                continue
            for fname in sorted(os.listdir(gamelogs_dir)):
                if not fname.endswith(".bk2"):
                    continue
                info = _parse_bk2_filename(fname)
                if info is None:
                    continue
                full_path = os.path.join(gamelogs_dir, fname)
                outcome = _get_outcome(full_path)
                clip = {
                    "path": full_path,
                    "filename": fname,
                    "outcome": outcome,
                    **info,
                }
                clips.append(clip)
                subjects.add(info["subject"])
                levels.add(info["level_str"])
                scene_ids.add(info["scene_id"])

    return {
        "clips": clips,
        "subjects": sorted(subjects),
        "levels": sorted(levels),
        "scene_ids": sorted(scene_ids),
    }


# ---------------------------------------------------------------------------
# Dialog builders
# ---------------------------------------------------------------------------

def _build_config_dialog(dataset_index):
    """Build Dialog 1 — comprehensive configuration.

    Parameters
    ----------
    dataset_index : dict or None
        Output of ``scan_dataset()`` if the scenes path is valid, else None.

    Returns
    -------
    (dict, list)
        The dialog dictionary and the field order list.
    """
    # --- Scene / Clip filter dropdowns ---
    if dataset_index is not None:
        scene_choices = ["(All)"] + dataset_index["scene_ids"]
        subject_choices = ["(All)"] + dataset_index["subjects"]
        outcome_choices = ["(All)", "completed", "death"]
    else:
        scene_choices = ["(All)"]
        subject_choices = ["(All)"]
        outcome_choices = ["(All)"]

    dlg_dict = {
        # Scene / Clip
        "Scene ID":                     scene_choices,
        "Subject filter":               subject_choices,
        "Outcome filter":               outcome_choices,
        "Clip min duration (s)":        config.CLIP_MIN_DURATION_SEC,
        "Clip min elements":            config.CLIP_MIN_ELEMENTS,
        "Clip max elements":            config.CLIP_MAX_ELEMENTS,
        # Timing
        "Execution timeout (s)":        config.EXECUTION_TIMEOUT,
        "Inter-execution interval (s)": config.INTER_EXECUTION_INTERVAL,
        "Inter-trial interval (s)":     config.INTER_TRIAL_INTERVAL,
        "Feedback duration (s)":        config.FEEDBACK_DURATION,
        "Gameplay max duration (s)":    config.GAMEPLAY_MAX_DURATION,
        "Fixation duration (s)":        config.FIXATION_DURATION,
        "Countdown step duration (s)":  config.COUNTDOWN_STEP_DURATION,
        "Speed factor":                 config.SPEED_FACTOR,
        # Training
        "Training reps per sequence":   config.TRAINING_REPS_PER_SEQ,
        "Pretrain reps per scene":      config.PRETRAIN_REPS_PER_SCENE,
        "Error rate threshold":         config.ERROR_RATE_THRESHOLD,
        "Fast bonus fraction":          config.FAST_BONUS_FRACTION,
        # Scanner
        "Scan prep duration (s)":       config.SCAN_PREP_DURATION,
        "Scan execution duration (s)":  config.SCAN_EXECUTION_DURATION,
        "Scan ITI (s)":                 config.SCAN_ITI,
        "Scan reps per sequence":       config.SCAN_REPS_PER_SEQ,
        "Scan number of runs":          config.SCAN_N_RUNS,
        "Scan rest periods":            config.SCAN_REST_PERIODS,
        "Scan rest duration (s)":       config.SCAN_REST_DURATION,
        # Behavior
        "Repeat until passed":          False,
    }

    order = [
        # Scene / Clip
        "Scene ID",
        "Subject filter",
        "Outcome filter",
        "Clip min duration (s)",
        "Clip min elements",
        "Clip max elements",
        # Timing
        "Execution timeout (s)",
        "Inter-execution interval (s)",
        "Inter-trial interval (s)",
        "Feedback duration (s)",
        "Gameplay max duration (s)",
        "Fixation duration (s)",
        "Countdown step duration (s)",
        "Speed factor",
        # Training
        "Training reps per sequence",
        "Pretrain reps per scene",
        "Error rate threshold",
        "Fast bonus fraction",
        # Scanner
        "Scan prep duration (s)",
        "Scan execution duration (s)",
        "Scan ITI (s)",
        "Scan reps per sequence",
        "Scan number of runs",
        "Scan rest periods",
        "Scan rest duration (s)",
        # Behavior
        "Repeat until passed",
    ]

    return dlg_dict, order


def _collect_overrides(dlg_dict):
    """Compare dialog values to current config defaults; return overrides.

    Returns
    -------
    dict
        ``{CONFIG_CONSTANT_NAME: new_value}`` for every field the user
        changed from the default.
    """
    overrides = {}
    for field_name, const_name in _FIELD_TO_CONFIG.items():
        default = getattr(config, const_name)
        raw = dlg_dict[field_name]
        # Coerce to the same type as the default
        try:
            if isinstance(default, float):
                value = float(raw)
            elif isinstance(default, int):
                value = int(raw)
            else:
                value = raw
        except (ValueError, TypeError):
            continue
        # Reject zero speed factor (would cause division by zero)
        if const_name == "SPEED_FACTOR" and value == 0:
            continue
        if value != default:
            overrides[const_name] = value
    return overrides


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_advanced_dialogs(scenes_path):
    """Run the advanced mode dialog sequence.

    Dialog 1: comprehensive configuration panel (~24 fields).
    Dialog 2 (optional): BK2 clip selection when a scene is filtered.

    Returns an AdvancedConfig.  If the user cancels at any stage,
    returns ``AdvancedConfig(enabled=False)``.
    """
    # --- Optionally scan dataset for dropdown population ---
    dataset_index = None
    if scenes_path and os.path.isdir(scenes_path):
        dataset_index = scan_dataset(scenes_path)
        if dataset_index and not dataset_index["clips"]:
            dataset_index = None  # treat empty dataset same as missing

    # --- Dialog 1: Advanced Configuration ---
    dlg_dict, order = _build_config_dialog(dataset_index)

    dlg1 = gui.DlgFromDict(
        dictionary=dlg_dict,
        title="Advanced Mode — Configuration",
        order=order,
        tip={
            "Scene ID": "Filter clips by scene (e.g. w1l1s3)",
            "Subject filter": "Filter clips by subject (e.g. sub-01)",
            "Outcome filter": "Filter by clip outcome (completed/death)",
            "Clip min duration (s)": f"Default: {config.CLIP_MIN_DURATION_SEC}",
            "Clip min elements": f"Default: {config.CLIP_MIN_ELEMENTS}",
            "Clip max elements": f"Default: {config.CLIP_MAX_ELEMENTS}",
            "Execution timeout (s)": f"Default: {config.EXECUTION_TIMEOUT}",
            "Inter-execution interval (s)": f"Default: {config.INTER_EXECUTION_INTERVAL}",
            "Inter-trial interval (s)": f"Default: {config.INTER_TRIAL_INTERVAL}",
            "Feedback duration (s)": f"Default: {config.FEEDBACK_DURATION}",
            "Gameplay max duration (s)": f"Default: {config.GAMEPLAY_MAX_DURATION}",
            "Fixation duration (s)": f"Default: {config.FIXATION_DURATION}",
            "Countdown step duration (s)": f"Default: {config.COUNTDOWN_STEP_DURATION}",
            "Speed factor": "Playback speed (1.0 = real-time, 0.5 = half speed). Must not be 0.",
            "Training reps per sequence": f"Default: {config.TRAINING_REPS_PER_SEQ}",
            "Pretrain reps per scene": f"Default: {config.PRETRAIN_REPS_PER_SCENE}",
            "Error rate threshold": f"Default: {config.ERROR_RATE_THRESHOLD}",
            "Fast bonus fraction": f"Default: {config.FAST_BONUS_FRACTION}",
            "Scan prep duration (s)": f"Default: {config.SCAN_PREP_DURATION}",
            "Scan execution duration (s)": f"Default: {config.SCAN_EXECUTION_DURATION}",
            "Scan ITI (s)": f"Default: {config.SCAN_ITI}",
            "Scan reps per sequence": f"Default: {config.SCAN_REPS_PER_SEQ}",
            "Scan number of runs": f"Default: {config.SCAN_N_RUNS}",
            "Scan rest periods": f"Default: {config.SCAN_REST_PERIODS}",
            "Scan rest duration (s)": f"Default: {config.SCAN_REST_DURATION}",
            "Repeat until passed": "Loop the trial until the participant passes",
        },
    )
    if not dlg1.OK:
        return AdvancedConfig(enabled=False)

    # --- Collect overrides ---
    overrides = _collect_overrides(dlg_dict)
    repeat_until_passed = bool(dlg_dict["Repeat until passed"])

    # --- Determine scene/clip filters ---
    scene_filter = dlg_dict["Scene ID"]
    subject_filter = dlg_dict["Subject filter"]
    outcome_filter = dlg_dict["Outcome filter"]

    # If no scene filter or no dataset, skip clip selection
    if (
        scene_filter == "(All)"
        or dataset_index is None
    ):
        return AdvancedConfig(
            enabled=True,
            repeat_until_passed=repeat_until_passed,
            _overrides=overrides,
        )

    # --- Filter clips for Dialog 2 ---
    filtered = list(dataset_index["clips"])
    if scene_filter != "(All)":
        filtered = [c for c in filtered if c["scene_id"] == scene_filter]
    if subject_filter != "(All)":
        filtered = [c for c in filtered if c["subject"] == subject_filter]
    if outcome_filter != "(All)":
        filtered = [c for c in filtered if c["outcome"] == outcome_filter]

    if not filtered:
        err = gui.Dlg(title="Advanced Mode")
        err.addText("No clips match the selected filters.")
        err.show()
        return AdvancedConfig(
            enabled=True,
            repeat_until_passed=repeat_until_passed,
            _overrides=overrides,
        )

    # --- Dialog 2: Clip selection ---
    labels = ["(None — use default sequence selection)"]
    label_to_clip = {}
    for clip in filtered:
        label = (
            f"{clip['scene_id']} | {clip['subject']} {clip['session']} | "
            f"{clip['outcome']}"
        )
        labels.append(label)
        label_to_clip[label] = clip

    clip_info = {
        "Select clip": labels,
    }
    dlg2 = gui.DlgFromDict(
        dictionary=clip_info,
        title=f"Advanced Mode — Select Clip ({len(filtered)} matches)",
        order=["Select clip"],
    )
    if not dlg2.OK:
        return AdvancedConfig(enabled=False)

    selected_label = clip_info["Select clip"]
    if selected_label == labels[0]:
        # "(None — use default)" selected
        return AdvancedConfig(
            enabled=True,
            selected_scene_id=scene_filter,
            repeat_until_passed=repeat_until_passed,
            _overrides=overrides,
        )

    clip = label_to_clip[selected_label]
    scene_info = get_scene_info_any(clip["scene_id"], scenes_path)

    return AdvancedConfig(
        enabled=True,
        selected_bk2=clip["path"],
        selected_scene_id=clip["scene_id"],
        selected_scene_info=scene_info,
        repeat_until_passed=repeat_until_passed,
        _overrides=overrides,
    )
