"""
Scene definitions, BK2 parsing, and action vocabulary for the SMB SSL task.

12 hardcoded scenes (2 groups of 6) selected from the CNeuroMod mario.scenes dataset.
Action sequences are dynamically extracted from random BK2 replay clips at runtime.
"""

import csv
import json
import os
import random
import re
import zipfile

from smb_ssl_task.config import (
    ACTION_VOCABULARY,
    CLIP_MIN_DURATION_SEC,
    CLIP_MIN_ELEMENTS,
    CLIP_MAX_ELEMENTS,
    CLIP_ALLOWED_SYMBOLS,
    EMULATOR_FPS,
)

# ---------------------------------------------------------------------------
# Module-level state: path to the mario.scenes dataset
# ---------------------------------------------------------------------------
_scenes_path = None
# Tracks the last source BK2 used per scene (updated on each extraction)
_last_source = {}
# Cached mastersheet data: {scene_id: scene_dict}
_mastersheet_cache = None


def set_scenes_path(path):
    """Set the path to the mario.scenes dataset root."""
    global _scenes_path
    _scenes_path = path


def get_scenes_path():
    """Return the current mario.scenes dataset path (or None)."""
    return _scenes_path

# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------
# Each scene dict: id, world, level, scene, entry (x-pos), exit (x-pos), layout

SCENE_SET_1 = [
    {"id": "w1l1s3", "world": 1, "level": 1, "scene": 3,
     "entry": 617, "exit": 744, "layout": 144},
    {"id": "w1l1s5", "world": 1, "level": 1, "scene": 5,
     "entry": 920, "exit": 1163, "layout": 144},
    {"id": "w1l1s8", "world": 1, "level": 1, "scene": 8,
     "entry": 1650, "exit": 1850, "layout": 144},
    {"id": "w1l2s2", "world": 1, "level": 2, "scene": 2,
     "entry": 551, "exit": 772, "layout": 55},
    {"id": "w1l3s3", "world": 1, "level": 3, "scene": 3,
     "entry": 580, "exit": 732, "layout": 245},
    {"id": "w2l1s3", "world": 2, "level": 1, "scene": 3,
     "entry": 545, "exit": 756, "layout": 207},
]

SCENE_SET_2 = [
    {"id": "w1l1s4", "world": 1, "level": 1, "scene": 4,
     "entry": 744, "exit": 920, "layout": 144},
    {"id": "w1l1s10", "world": 1, "level": 1, "scene": 10,
     "entry": 2142, "exit": 2413, "layout": 144},
    {"id": "w1l2s4", "world": 1, "level": 2, "scene": 4,
     "entry": 1030, "exit": 1250, "layout": 55},
    {"id": "w1l3s7", "world": 1, "level": 3, "scene": 7,
     "entry": 1616, "exit": 1902, "layout": 245},
    {"id": "w2l1s5", "world": 2, "level": 1, "scene": 5,
     "entry": 983, "exit": 1203, "layout": 207},
    {"id": "w2l3s3", "world": 2, "level": 3, "scene": 3,
     "entry": 679, "exit": 940, "layout": 74},
]

ALL_SCENES = {s["id"]: s for s in SCENE_SET_1 + SCENE_SET_2}

# ---------------------------------------------------------------------------
# Pre-extracted canonical action sequences (from reference BK2 replays)
# ---------------------------------------------------------------------------
# Each sequence is a list of action symbols representing the compressed
# button-state trajectory for traversing the scene.
# These are placeholders — will be re-extracted from actual BK2 data once
# the scene selection is finalized.

CANONICAL_SEQUENCES = {
    # --- Set 1 ---
    # Each element: (action_symbol, duration_in_frames_at_60fps)
    # Placeholder durations — will be re-extracted from BK2 data.
    "w1l1s3":  [("R", 20), ("rR", 25), ("rRJ", 40), ("rR", 20), ("R", 15), ("rRJ", 35), ("rR", 25), ("R", 15)],
    "w1l1s5":  [("rR", 25), ("rRJ", 40), ("R", 18), ("rR", 22), ("rRJ", 38), ("rR", 25), ("rRJ", 42), ("R", 20), ("rR", 22)],
    "w1l1s8":  [("rR", 22), ("rRJ", 38), ("rR", 25), ("R", 18), ("rRJ", 42), ("rR", 20), ("rR", 28), ("rRJ", 35), ("R", 15), ("rR", 22)],
    "w1l2s2":  [("R", 18), ("rR", 25), ("rRJ", 40), ("rR", 22), ("RJ", 35), ("R", 20), ("rR", 25), ("rRJ", 38), ("rR", 22)],
    "w1l3s3":  [("R", 20), ("RJ", 35), ("R", 18), ("RJ", 38), ("rR", 25), ("rRJ", 40), ("R", 15)],
    "w2l1s3":  [("rR", 25), ("rRJ", 42), ("rR", 22), ("R", 18), ("rRJ", 38), ("R", 20), ("rR", 25), ("rRJ", 35)],
    # --- Set 2 ---
    "w1l1s4":  [("rR", 22), ("R", 18), ("rRJ", 40), ("rR", 25), ("rR", 22), ("rRJ", 38), ("R", 15), ("rR", 25), ("R", 18)],
    "w1l1s10": [("R", 20), ("RJ", 35), ("R", 18), ("R", 15), ("RJ", 38), ("R", 20), ("R", 18), ("RJ", 35), ("R", 15), ("RJ", 38)],
    "w1l2s4":  [("rR", 25), ("rRJ", 40), ("rR", 22), ("RJ", 35), ("rR", 25), ("rRJ", 38), ("rR", 22), ("R", 18), ("rRJ", 35)],
    "w1l3s7":  [("R", 20), ("RJ", 38), ("R", 18), ("RJ", 35), ("R", 15), ("rRJ", 42), ("R", 20), ("RJ", 38), ("R", 18)],
    # Extracted from sub-01_ses-015_task-mario_level-w2l1_scene-5_clip-01500000000619.bk2
    "w2l1s5":  [("L", 19), ("J", 21), ("rLJ", 3), ("L", 8), ("rR", 16), ("rRJ", 18), ("J", 3), ("L", 13), ("rR", 11), ("rRJ", 29), ("rR", 35), ("rRJ", 11), ("rR", 11)],
    "w2l3s3":  [("rR", 25), ("rRJ", 38), ("rR", 22), ("R", 18), ("rRJ", 40), ("rR", 25), ("rR", 22), ("rRJ", 35), ("rR", 20)],
}

# Source BK2 clip filenames for each canonical sequence.
# None = placeholder data (not yet extracted from a real BK2).
CANONICAL_SEQUENCE_SOURCES = {
    # --- Set 1 ---
    "w1l1s3":  None,
    "w1l1s5":  None,
    "w1l1s8":  None,
    "w1l2s2":  None,
    "w1l3s3":  None,
    "w2l1s3":  None,
    # --- Set 2 ---
    "w1l1s4":  None,
    "w1l1s10": None,
    "w1l2s4":  None,
    "w1l3s7":  None,
    "w2l1s5":  "sub-01_ses-015_task-mario_level-w2l1_scene-5_clip-01500000000619.bk2",
    "w2l3s3":  None,
}

PRETRAIN_SEQUENCE_SOURCES = {
    "w1l1s1": None,
    "w1l1s2": None,
    "w1l2s0": None,
    "w1l3s1": None,
    "w2l1s1": None,
    "w2l3s1": None,
}

# ---------------------------------------------------------------------------
# Pre-training scenes (not in experimental set)
# ---------------------------------------------------------------------------

PRETRAIN_SCENES = [
    {"id": "w1l1s1", "world": 1, "level": 1, "scene": 1,
     "entry": 240, "exit": 456, "layout": 144},
    {"id": "w1l1s2", "world": 1, "level": 1, "scene": 2,
     "entry": 456, "exit": 617, "layout": 144},
    {"id": "w1l2s0", "world": 1, "level": 2, "scene": 0,
     "entry": 40, "exit": 286, "layout": 55},
    {"id": "w1l3s1", "world": 1, "level": 3, "scene": 1,
     "entry": 200, "exit": 382, "layout": 245},
    {"id": "w2l1s1", "world": 2, "level": 1, "scene": 1,
     "entry": 195, "exit": 394, "layout": 207},
    {"id": "w2l3s1", "world": 2, "level": 3, "scene": 1,
     "entry": 212, "exit": 451, "layout": 74},
]

PRETRAIN_CANONICAL_SEQUENCES = {
    "w1l1s1":  [("R", 18), ("rR", 25), ("RJ", 35), ("R", 20), ("rR", 22), ("rRJ", 40), ("rR", 25), ("R", 15)],
    "w1l1s2":  [("rR", 25), ("R", 18), ("rRJ", 38), ("R", 20), ("rR", 22), ("R", 15)],
    "w1l2s0":  [("R", 20), ("rR", 25), ("rRJ", 40), ("rR", 22), ("R", 18), ("rRJ", 38), ("R", 20), ("rR", 22)],
    "w1l3s1":  [("R", 18), ("RJ", 35), ("R", 20), ("RJ", 38), ("R", 15), ("R", 20), ("RJ", 35)],
    "w2l1s1":  [("R", 20), ("RJ", 38), ("R", 18), ("rR", 25), ("rRJ", 40), ("rR", 22), ("RJ", 35), ("R", 18)],
    "w2l3s1":  [("rR", 25), ("rRJ", 38), ("rR", 22), ("rR", 25), ("rRJ", 40), ("rR", 20), ("rR", 22)],
}


# ---------------------------------------------------------------------------
# Scene access functions
# ---------------------------------------------------------------------------

def get_scenes(group):
    """Return (trained_scenes, untrained_scenes) dicts for a group.

    Parameters
    ----------
    group : int
        1 or 2.

    Returns
    -------
    trained : dict
        {scene_id: scene_dict} for the 6 trained scenes.
    untrained : dict
        {scene_id: scene_dict} for the 6 untrained scenes.
    """
    if group == 1:
        trained = {s["id"]: s for s in SCENE_SET_1}
        untrained = {s["id"]: s for s in SCENE_SET_2}
    elif group == 2:
        trained = {s["id"]: s for s in SCENE_SET_2}
        untrained = {s["id"]: s for s in SCENE_SET_1}
    else:
        raise ValueError(f"Group must be 1 or 2, got {group}")
    return trained, untrained


def _parse_scene_id(scene_id):
    """Parse a scene_id like 'w1l1s3' into (world, level, scene_num)."""
    parts = scene_id.replace("w", "").replace("l", " ").replace("s", " ").split()
    return int(parts[0]), int(parts[1]), int(parts[2])


def find_all_clips(scene_id, scenes_path=None):
    """Find ALL available BK2 clips for a scene across all subjects/sessions.

    Parameters
    ----------
    scene_id : str
        e.g. "w1l1s3"
    scenes_path : str, optional
        Root path to the mario.scenes dataset. Uses module-level default
        if not provided.

    Returns
    -------
    list[str]
        Sorted list of full paths to matching .bk2 files.
    """
    if scenes_path is None:
        scenes_path = _scenes_path
    if not scenes_path or not os.path.isdir(scenes_path):
        return []

    world, level, scene_num = _parse_scene_id(scene_id)
    level_str = f"w{world}l{level}"
    pattern = f"_level-{level_str}_scene-{scene_num}_clip-"

    clips = []
    # Walk all sub-*/ses-*/gamelogs/ directories
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
            for fname in os.listdir(gamelogs_dir):
                if fname.endswith(".bk2") and pattern in fname:
                    clips.append(os.path.join(gamelogs_dir, fname))

    return sorted(clips)


def _clip_is_cleared(bk2_path):
    """Return True if the clip's summary JSON reports outcome 'completed'.

    Each BK2 has a companion ``*_summary.json`` with an ``Outcome`` field
    (``"completed"`` or ``"death"``).  Returns False if the summary is
    missing or the outcome is not ``"completed"``.
    """
    summary_path = bk2_path.replace(".bk2", "_summary.json")
    try:
        with open(summary_path) as f:
            meta = json.load(f)
        return meta.get("Outcome") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def _clip_passes_filter(seq):
    """Check whether an extracted sequence meets the clip filter criteria."""
    n_elements = len(seq)
    if n_elements < CLIP_MIN_ELEMENTS or n_elements > CLIP_MAX_ELEMENTS:
        return False
    total_frames = sum(dur for _, dur in seq)
    duration_sec = total_frames / EMULATOR_FPS
    if duration_sec < CLIP_MIN_DURATION_SEC:
        return False
    # Reject clips containing forbidden symbols (D, LEFT+RIGHT combos, etc.)
    if not all(sym in CLIP_ALLOWED_SYMBOLS for sym, _ in seq):
        return False
    return True


def get_canonical_sequence(scene_id):
    """Return an action sequence for a scene, extracted from a random BK2 clip.

    Tries random clips until one passes the filter criteria
    (duration >= CLIP_MIN_DURATION_SEC, elements between CLIP_MIN/MAX_ELEMENTS).
    Falls back to hardcoded placeholder sequences if no clips pass.
    """
    global _last_source

    # Try dynamic extraction — shuffle cleared clips and pick the first that passes
    clips = find_all_clips(scene_id)
    clips = [c for c in clips if _clip_is_cleared(c)]
    if clips:
        random.shuffle(clips)
        for bk2_path in clips:
            try:
                seq = extract_action_sequence(bk2_path)
                if not seq:
                    continue
                # Filter out idle '_' elements (standing still)
                seq = [(sym, dur) for sym, dur in seq if sym != "_"]
                if seq and _clip_passes_filter(seq):
                    _last_source[scene_id] = bk2_path  # full path
                    return seq
            except Exception as e:
                print(f"[WARNING] Failed to extract from {bk2_path}: {e}")
                continue

    # Fallback to hardcoded placeholders
    _last_source[scene_id] = None
    if scene_id in CANONICAL_SEQUENCES:
        return list(CANONICAL_SEQUENCES[scene_id])
    if scene_id in PRETRAIN_CANONICAL_SEQUENCES:
        return list(PRETRAIN_CANONICAL_SEQUENCES[scene_id])
    raise KeyError(f"No canonical sequence for scene '{scene_id}'")


def get_canonical_sequence_source(scene_id):
    """Return the full path to the source BK2 clip for the last extracted sequence.

    Returns None if the sequence is a placeholder.
    """
    return _last_source.get(scene_id)


def get_clip_savestate_path(scene_id):
    """Return the .state file corresponding to the last BK2 used for this scene.

    Each BK2 in the mario.scenes dataset has a matching .state file
    (same path and name, different extension).

    Returns None if no BK2 was used (placeholder sequence) or if the
    .state file doesn't exist.
    """
    bk2_path = _last_source.get(scene_id)
    if bk2_path is None:
        return None
    state_path = bk2_path.replace(".bk2", ".state")
    if os.path.isfile(state_path):
        return state_path
    return None


def get_pretrain_scenes():
    """Return list of pretrain scene dicts (not in experimental set)."""
    return list(PRETRAIN_SCENES)


# ---------------------------------------------------------------------------
# Mastersheet loading and scene lookup helpers
# ---------------------------------------------------------------------------

def load_mastersheet(scenes_path=None):
    """Parse the mastersheet CSV and return {scene_id: scene_dict}.

    Each scene_dict has keys: id, world, level, scene, entry, exit, layout.
    Cached at module level after first call.
    """
    global _mastersheet_cache
    if _mastersheet_cache is not None:
        return _mastersheet_cache

    if scenes_path is None:
        scenes_path = _scenes_path
    if not scenes_path:
        return {}

    csv_path = os.path.join(
        scenes_path, "sourcedata", "scenes_info", "scenes_mastersheet.csv"
    )
    if not os.path.isfile(csv_path):
        return {}

    result = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("World") or not row["World"].strip():
                continue
            w = int(row["World"])
            l = int(row["Level"])
            s = int(row["Scene"])
            scene_id = f"w{w}l{l}s{s}"
            result[scene_id] = {
                "id": scene_id,
                "world": w,
                "level": l,
                "scene": s,
                "entry": int(row["Entry point"]),
                "exit": int(row["Exit point"]),
                "layout": int(row["Layout"]),
            }

    _mastersheet_cache = result
    return _mastersheet_cache


def get_scene_info_any(scene_id, scenes_path=None):
    """Look up scene info from any source: ALL_SCENES, pretrain, or mastersheet.

    Returns a scene_dict or None if not found anywhere.
    """
    if scene_id in ALL_SCENES:
        return ALL_SCENES[scene_id]

    pretrain_dict = {s["id"]: s for s in PRETRAIN_SCENES}
    if scene_id in pretrain_dict:
        return pretrain_dict[scene_id]

    ms = load_mastersheet(scenes_path)
    return ms.get(scene_id)


def _scene_id_from_filename(filename):
    """Parse a BK2 filename to extract scene_id.

    Expects patterns like ``_level-w1l1_scene-3_clip-`` in the filename.
    Returns e.g. ``"w1l1s3"`` or None if parsing fails.
    """
    m = re.search(r"_level-(w\d+l\d+)_scene-(\d+)_clip-", filename)
    if m:
        return f"{m.group(1)}s{m.group(2)}"
    return None


def get_canonical_sequence_from_bk2(bk2_path):
    """Extract an action sequence from a specific BK2 file.

    Like ``get_canonical_sequence`` but deterministic — uses the given BK2
    without random selection. Updates ``_last_source`` for the scene.

    Returns the filtered sequence or raises ValueError if extraction fails.
    """
    global _last_source

    scene_id = _scene_id_from_filename(os.path.basename(bk2_path))

    seq = extract_action_sequence(bk2_path)
    if not seq:
        raise ValueError(f"No action data extracted from {bk2_path}")

    # Filter out idle '_' elements
    seq = [(sym, dur) for sym, dur in seq if sym != "_"]
    if not seq:
        raise ValueError(f"Sequence is empty after filtering idle frames: {bk2_path}")

    if scene_id:
        _last_source[scene_id] = bk2_path

    return seq


def get_savestate_path(scene_id, participant="01", session="001",
                       scenes_path=None):
    """Return path to a .state file for a scene from the mario.scenes dataset.

    Searches for the first matching clip state file.

    Parameters
    ----------
    scene_id : str
        e.g. "w1l1s3"
    participant : str
        Subject ID in the dataset (default "01").
    session : str
        Session ID (default "001").
    scenes_path : str, optional
        Root path to the mario.scenes dataset. Uses module-level default
        if not provided.
    """
    if scenes_path is None:
        scenes_path = _scenes_path
    if not scenes_path:
        raise ValueError("scenes_path is required (set 'Scenes dataset dir' in the GUI)")

    world, level, scene_num = _parse_scene_id(scene_id)
    level_str = f"w{world}l{level}"

    gamelogs_dir = os.path.join(
        scenes_path,
        f"sub-{participant}",
        f"ses-{session}",
        "gamelogs",
    )
    if not os.path.isdir(gamelogs_dir):
        return None

    # Look for state files matching this level and scene
    pattern = f"_level-{level_str}_scene-{scene_num}_clip-"
    for fname in os.listdir(gamelogs_dir):
        if fname.endswith(".state") and pattern in fname:
            return os.path.join(gamelogs_dir, fname)
    return None


def find_reference_bk2(scene_id, participant="01", session="001",
                       scenes_path=None):
    """Find a successful clip BK2 for a scene from the mario.scenes dataset.

    Parameters
    ----------
    scene_id : str
    participant : str
    session : str
    scenes_path : str, optional
        Root path to the mario.scenes dataset. Uses module-level default
        if not provided.

    Returns
    -------
    str or None
        Path to the BK2 file, or None if not found.
    """
    if scenes_path is None:
        scenes_path = _scenes_path
    if not scenes_path:
        raise ValueError("scenes_path is required (set 'Scenes dataset dir' in the GUI)")

    world, level, scene_num = _parse_scene_id(scene_id)
    level_str = f"w{world}l{level}"

    gamelogs_dir = os.path.join(
        scenes_path,
        f"sub-{participant}",
        f"ses-{session}",
        "gamelogs",
    )
    if not os.path.isdir(gamelogs_dir):
        return None

    pattern = f"_level-{level_str}_scene-{scene_num}_clip-"
    for fname in sorted(os.listdir(gamelogs_dir)):
        if fname.endswith(".bk2") and pattern in fname:
            return os.path.join(gamelogs_dir, fname)
    return None


# ---------------------------------------------------------------------------
# BK2 parser (standalone, no external dependencies)
# ---------------------------------------------------------------------------

# BK2 input log button positions (0-indexed within the 8-char field)
_BK2_BUTTON_POSITIONS = {
    0: "A",
    1: "RIGHT",
    2: "LEFT",
    3: "DOWN",
    4: "UP",
    5: "START",
    6: "SELECT",
    7: "B",
}


def parse_bk2(bk2_path):
    """Extract per-frame button states from a BK2 file.

    Parameters
    ----------
    bk2_path : str
        Path to a .bk2 file (ZIP archive).

    Returns
    -------
    list[set[str]]
        One set per frame, containing pressed button names.
        e.g. [{"RIGHT"}, {"RIGHT", "A"}, set(), ...]
    """
    with zipfile.ZipFile(bk2_path, "r") as zf:
        raw = zf.read("Input Log.txt").decode("utf-8")

    frames = []
    in_input = False
    header_seen = False

    for line in raw.splitlines():
        line = line.strip()
        if line == "[Input]":
            in_input = True
            continue
        if line == "[/Input]":
            break
        if not in_input:
            continue

        # Skip header line (starts with "P1")
        if not header_seen:
            header_seen = True
            continue

        # Parse frame: |..|XXXXXXXX|
        parts = line.split("|")
        # parts[0] is empty (before first pipe), parts[1] is "..", parts[2] is 8-char buttons
        if len(parts) < 3:
            continue
        button_field = parts[2]
        if len(button_field) < 8:
            continue

        pressed = set()
        for pos, button_name in _BK2_BUTTON_POSITIONS.items():
            if pos < len(button_field) and button_field[pos] != ".":
                pressed.add(button_name)
        frames.append(pressed)

    return frames


def buttons_to_symbol(buttons):
    """Map a set of pressed buttons to the closest action vocabulary symbol.

    Strips START/SELECT/UP as they are not in the action vocabulary.
    """
    # Only keep gameplay-relevant buttons
    relevant = buttons - {"START", "SELECT", "UP"}
    key = frozenset(relevant)

    if key in ACTION_VOCABULARY:
        return ACTION_VOCABULARY[key]

    # Fallback: find closest match by subset overlap
    best_symbol = "_"
    best_overlap = -1
    for vocab_key, symbol in ACTION_VOCABULARY.items():
        if vocab_key <= relevant:  # vocab_key is a subset of what's pressed
            overlap = len(vocab_key)
            if overlap > best_overlap:
                best_overlap = overlap
                best_symbol = symbol
    return best_symbol


def extract_action_sequence(bk2_path, min_frames=3):
    """Parse a BK2 and compress into a list of (action_symbol, frame_count) tuples.

    1. Parse per-frame button states
    2. Map each frame to an action symbol
    3. Collapse consecutive identical symbols
    4. Filter very short actions (< min_frames)

    Parameters
    ----------
    bk2_path : str
        Path to a .bk2 file.
    min_frames : int
        Minimum consecutive frames for an action to be kept (noise filter).

    Returns
    -------
    list[tuple[str, int]]
        Ordered list of (action_symbol, duration_in_frames) tuples.
    """
    frames = parse_bk2(bk2_path)
    if not frames:
        return []

    # Map each frame to a symbol
    symbols = [buttons_to_symbol(f) for f in frames]

    # Collapse consecutive identical symbols with frame counts
    collapsed = []
    current = symbols[0]
    count = 1
    for s in symbols[1:]:
        if s == current:
            count += 1
        else:
            collapsed.append((current, count))
            current = s
            count = 1
    collapsed.append((current, count))

    # Filter noise (very short actions)
    filtered = [(sym, cnt) for sym, cnt in collapsed if cnt >= min_frames]

    return filtered
