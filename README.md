# SMB-MSL: Motor Sequence Learning Tasks

This repository contains two PsychoPy-based motor sequence learning tasks for the CNeuroMod project:

1. **Berlot DSP Task** (`berlot2020_task/`) — Digit sequence production task from Berlot et al. (2020)
2. **SMB Scene Sequence Learning Task** (`smb_ssl_task/`) — Naturalistic sequence learning using Super Mario Bros. scenes

Both tasks share a common experimental structure (training with adaptive rewards, behavioral tests, fMRI scan sessions) but differ in the motor sequences used: abstract digit sequences vs. NES button-chord sequences derived from Mario gameplay.

## Installation

```bash
pip install --upgrade pip
```

Requires Python 3.9+ and PsychoPy. The SMB-SSL task additionally requires `stable-retro` and `numpy` for gameplay mode.

PsychoPy requires wxPython, which might not install cleanly on Linux.

For Ubuntu 24.04 / Linux Mint 22 (Python 3.10):
```bash
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-24.04/wxpython-4.2.5-cp310-cp310-linux_x86_64.whl
```

> **Other setups:** The URL encodes your distro version and Python version. Browse the [wxPython extras index](https://extras.wxpython.org/wxPython4/extras/linux/gtk3/) to find the wheel matching your OS, then pick the file whose `cpXYZ` tag matches your Python (e.g., `cp311` for Python 3.11). If you get an `ImportError` about a missing `.so` file at runtime, install the corresponding system library (`apt search <name>`) or create a symlink from the version you have to the one expected.

```bash
# From the repository root
pip install -e .
```

### Linux: realtime scheduling permissions

PsychoPy's keyboard backend (PsychToolbox) requires realtime thread scheduling. Without it the task will crash with a segfault. Run the following once, then **log out and back in**:

```bash
sudo groupadd --force psychopy
sudo usermod -a -G psychopy $USER
sudo tee /etc/security/limits.d/99-psychopy.conf > /dev/null << 'EOF'
@psychopy - nice -20
@psychopy - rtprio 50
@psychopy - memlock unlimited
EOF
```

If you manage PsychoPy separately (e.g., standalone installer), you can install without dependencies:

```bash
pip install -e . --no-deps
```

---

## Task 1: Berlot DSP Task

PsychoPy implementation of the discrete sequence production (DSP) task from:

> Berlot, E., Popp, N. J., & Diedrichsen, J. (2020). A critical re-evaluation of fMRI signatures of motor sequence learning. *eLife*, 9, e55241.

### Running the task

```bash
# As a module
python -m berlot2020_task

# Or via the installed entry point
berlot2020-task
```

A GUI dialog will prompt you for:

| Field | Description |
|---|---|
| **Participant ID** | String identifier (e.g., `01`, `sub-02`) |
| **Group** | `1` or `2` — determines which 6 sequences are trained vs. untrained |
| **Session type** | `training`, `test`, `test_left`, `scan_paced`, `scan_fullspeed`, or `pretrain` |
| **Session number** | Integer session index |
| **Blocks / Reps** | Number of blocks (training), reps per sequence (test), or ignored (scan/pretrain) |

### Session types

**Pre-training** (`pretrain`) — Familiarization with the apparatus. Uses 6 random sequences not in the experimental set. Same self-paced trial structure as the behavioral test (digits visible, no points).

**Training** (`training`) — 6 trained sequences, blocks of 24 trials (4 reps per sequence). Each trial has two executions: first with digits visible, then from memory. An adaptive reward system awards 0, 1, or 3 points based on accuracy and speed.

**Behavioral test** (`test`) — All 12 sequences (6 trained + 6 untrained) intermixed. Digits remain visible for both executions. No points system.

**Left-hand transfer test** (`test_left`) — Trained sequences only, played with the left hand. Includes intrinsically-matched trials (same finger sequence) and extrinsically-matched trials (mirrored fingers: finger N becomes finger 6-N). No points system.

**Scan session — paced** (`scan_paced`) — 8 functional runs for fMRI scanning (scans 1–3). All 12 sequences, each repeated 6 times per run (72 trials). Sequences appear in consecutive pairs. Each trial lasts exactly 5s: 1s preparation, 3.5s execution with expanding pink pacing line, 0.5s ITI. Points: +3 correct, +0 error. 5 rest periods (10s fixation) per run. Waits for scanner trigger (`=` key) before each run.

**Scan session — full speed** (`scan_fullspeed`) — Same as paced but with a short static go-cue instead of the expanding pacing line (scan 4). Execute as fast as possible.

### Key mapping

**Right hand** (training, test, scan, pretrain):

| Finger | Key | Digit |
|---|---|---|
| Thumb | `Space` | 1 |
| Index | `J` | 2 |
| Middle | `K` | 3 |
| Ring | `L` | 4 |
| Pinky | `;` | 5 |

**Left hand** (test_left):

| Finger | Key | Digit |
|---|---|---|
| Thumb | `Space` | 1 |
| Index | `F` | 2 |
| Middle | `D` | 3 |
| Ring | `S` | 4 |
| Pinky | `A` | 5 |

Press `Escape` at any time to abort the session (data collected so far is saved).

### Berlot output

Data is saved to `data/` in the working directory:

```
data/
└── sub-01/
    ├── training/
    │   └── sub-01_training_ses-01.tsv
    ├── test/
    │   └── sub-01_test_ses-01.tsv
    ├── test_left/
    │   └── sub-01_test_left_ses-01.tsv
    ├── pretrain/
    │   └── sub-01_pretrain_ses-01.tsv
    ├── scan_paced/
    │   └── sub-01_scan_paced_ses-01.tsv
    └── scan_fullspeed/
        └── sub-01_scan_fullspeed_ses-01.tsv
```

Each TSV file has one row per **execution** (2 rows per trial) with the following columns:

| Column | Description |
|---|---|
| `participant_id` | Participant identifier |
| `group` | Group assignment (1 or 2) |
| `session_type` | `training`, `test`, `test_left`, `pretrain`, `scan_paced`, or `scan_fullspeed` |
| `session_number` | Session index |
| `block_number` | Block number (1 for test/scan sessions) |
| `run_number` | Functional run number (scan sessions) or 0 |
| `trial_number` | Global trial counter |
| `sequence_id` | Sequence identifier (1–12 experimental, 101+ pretrain) |
| `sequence_digits` | Target sequence (semicolon-separated) |
| `execution_number` | 1 or 2 |
| `hand` | `right` or `left` |
| `condition` | `trained`, `untrained`, `intrinsic`, `extrinsic`, or `pretrain` |
| `response_keys` | Keys pressed (semicolon-separated) |
| `response_times` | Press timestamps in seconds (semicolon-separated) |
| `accuracy_per_press` | 1/0 per position (semicolon-separated) |
| `accuracy_trial` | 1 if all presses correct, else 0 |
| `movement_time` | Time from first to last keypress (seconds) |
| `inter_press_intervals` | Intervals between consecutive presses (semicolon-separated) |
| `points_awarded` | Points for this execution (training/scan) or 0 |

### Berlot configuration

Edit `berlot2020_task/config.py` to adjust:

- Display settings (screen size, fullscreen, colors, font sizes)
- Timing parameters (timeouts, inter-trial intervals, scan trial timing)
- Training parameters (trials per block, error threshold, point values)
- Key mappings (right and left hand)
- Scanner settings (trigger key, TR)
- Pacing line appearance (color, height, position)
- Pre-training parameters (number of sequences, repetitions)

---

## Task 2: SMB Scene Sequence Learning (SSL) Task

Bridges naturalistic Super Mario Bros. gameplay with the motor sequence learning literature. Uses scene segments from the CNeuroMod `mario.scenes` dataset.

Two modes of operation:

- **MSP mode** (Motor Sequence Production) — Canonical button sequences extracted from BK2 replay files are presented as abstract NES button-chord sequences with duration bars. The player reproduces them by pressing the correct button combinations for the correct duration (analogous to the Berlot digit task, but with NES button chords and timing).
- **Gameplay mode** — Loads SMB savestates via gym-retro. The player actually plays the scenes in real time.

### Retro integration setup (gameplay mode)

Gameplay mode requires a **retro integration directory** — a folder that contains a `SuperMarioBros-Nes/` subfolder with the ROM and game metadata. The expected structure is:

```
/path/to/your/integration_dir/
└── SuperMarioBros-Nes/
    ├── rom.nes            # The NES ROM file
    ├── data.json          # RAM variable definitions
    ├── scenario.json      # Reward/done conditions
    ├── metadata.json      # Game metadata
    ├── rom.sha            # ROM hash
    └── *.state            # Savestate files
```

On the first run, enter the path to the directory that *contains* `SuperMarioBros-Nes/` (not the `SuperMarioBros-Nes/` folder itself) in the **Retro integration dir** field of the GUI dialog. This path is saved to `.smb_ssl_settings.json` and will be pre-filled on subsequent runs.

MSP mode does not require this — you can leave the field empty.

### Running the task

```bash
# As a module
python -m smb_ssl_task

# Or via the installed entry point
smb-ssl-task
```

A GUI dialog will prompt you for:

| Field | Description |
|---|---|
| **Participant ID** | String identifier (e.g., `01`) |
| **Group** | `1` or `2` — determines which 6 scenes are trained vs. untrained |
| **Mode** | `msp` (chord sequences with duration) or `gameplay` (actual Mario play) |
| **Session type** | `training`, `test`, `scan_paced`, `scan_fullspeed`, or `pretrain` |
| **Session number** | Integer session index |
| **Blocks / Reps** | Number of blocks (training) or reps per scene (test) |
| **Retro integration dir** | Directory containing `SuperMarioBros-Nes/` (gameplay mode only, saved across runs) |

### Session types

**Pre-training** (`pretrain`) — Familiarization using 6 scenes not in the experimental set. Self-paced, two executions per trial (both guided). No points.

**Training** (`training`) — 6 trained scenes, blocks of 24 trials (4 reps per scene, shuffled). Each trial has two executions:
- *MSP mode*: Execution 1 with symbols visible, Execution 2 from memory (symbols hidden)
- *Gameplay mode*: Execution 1 normal play, Execution 2 immediate replay

Adaptive reward system: 0 points (error/slow), 1 point (correct), 3 points (correct and fast). Speed threshold updates per block when error rate < 15%.

**Behavioral test** (`test`) — All 12 scenes (6 trained + 6 untrained) intermixed. Self-paced, no points.

**Scan session — paced** (`scan_paced`) — 8 functional runs for fMRI. 12 scenes x 6 reps = 72 trials per run, arranged as consecutive pairs. Each trial: 1s prep + 5s execution + 0.5s ITI. Expanding pacing line in MSP mode. 5 rest periods (10s fixation) per run. Waits for scanner trigger (`=` key) before each run.

**Scan session — full speed** (`scan_fullspeed`) — Same structure but with a short static go-cue instead of the expanding pacing line.

### Action vocabulary

NES inputs are compressed into distinct actions, each representing a set of simultaneously held buttons:

| Symbol | Buttons | Description |
|--------|---------|-------------|
| `_` | (none) | Wait / stand |
| `R` | RIGHT | Walk right |
| `rR` | RIGHT+B | Run right |
| `J` | A | Standing jump |
| `RJ` | RIGHT+A | Walk-jump right |
| `rRJ` | RIGHT+A+B | Run-jump right |
| `L` | LEFT | Walk left |
| `LJ` | LEFT+A | Walk-jump left |
| `rLJ` | LEFT+A+B | Run-jump left |
| `D` | DOWN | Crouch / enter pipe |

In MSP mode, each element in the sequence is a chord with a target duration. The player must press the correct button combination and hold it for the indicated duration (shown by a horizontal bar below each symbol). A configurable timing tolerance (default 50ms) allows for small timing errors.

### Input mapping

**Keyboard:**

| Key | NES Button |
|-----|------------|
| Arrow Right | RIGHT |
| Arrow Left | LEFT |
| Arrow Up | UP |
| Arrow Down | DOWN |
| X | A (jump) |
| Z | B (run) |

**Gamepad** (optional): D-pad or left stick for directions, face buttons for A/B. Configurable in `smb_ssl_task/config.py`.

Press `Escape` at any time to abort the session (data collected so far is saved).

### Scene selection

12 scenes are hardcoded (2 groups of 6), drawn from worlds 1–2 of the CNeuroMod `mario.scenes` dataset:

- **Set 1**: w1l1s3, w1l1s5, w1l1s8, w1l2s2, w1l3s3, w2l1s3
- **Set 2**: w1l1s4, w1l1s10, w1l2s4, w1l3s7, w2l1s5, w2l3s3

Group 1 trains on Set 1 (Set 2 untrained); Group 2 trains on Set 2 (Set 1 untrained). 6 additional pretrain scenes are selected from non-overlapping positions.

### SSL output

Data is saved to `data/` with the same directory structure as the Berlot task. Each TSV file has one row per execution with columns covering both modes:

| Column | Description |
|---|---|
| `participant_id` | Participant identifier |
| `group` | Group assignment (1 or 2) |
| `session_type` | Session type string |
| `session_number` | Session index |
| `block_number` | Block number (1 for test/scan) |
| `run_number` | Functional run (scan) or 0 |
| `trial_number` | Global trial counter |
| `scene_id` | Scene identifier (e.g., `w1l1s3`) |
| `mode` | `msp` or `gameplay` |
| `execution_number` | 1 or 2 |
| `condition` | `trained`, `untrained`, or `pretrain` |
| `target_sequence` | Target action symbols, semicolon-separated (MSP) |
| `response_sequence` | Pressed action symbols, semicolon-separated (MSP) |
| `target_durations` | Target hold durations in seconds, semicolon-separated (MSP) |
| `response_durations` | Actual hold durations in seconds, semicolon-separated (MSP) |
| `accuracy_per_element` | 1/0 per position, semicolon-separated (MSP) |
| `accuracy_trial` | 1 if all elements correct (chord + timing), else 0 (MSP) |
| `movement_time` | First to last element time in seconds (MSP) |
| `inter_element_intervals` | Intervals between elements, semicolon-separated (MSP) |
| `outcome` | `completed`, `death`, or `timeout` (gameplay) |
| `traversal_time` | Scene traversal time in seconds (gameplay) |
| `distance_reached` | Fraction of scene traversed, 0.0–1.0 (gameplay) |
| `points_awarded` | Points for this execution |

Columns not applicable to the current mode are filled with `NA`.

### SSL configuration

Edit `smb_ssl_task/config.py` to adjust:

- Display settings (screen size, game render size, colors, font sizes)
- Action display parameters (symbol spacing, feedback colors)
- Timing parameters (timeouts, intervals, gameplay max duration)
- Training parameters (trials per block, error threshold, point values)
- Scanner settings (trigger key, trial timing, runs, rest periods)
- Input mappings (keyboard and gamepad)
- Duration bar appearance and timing tolerance
- Path to the `mario.scenes` dataset (set in GUI)

### BK2 parser

`smb_ssl_task/scenes.py` includes a standalone BK2 parser for extracting action sequences from replay files:

```python
from smb_ssl_task.scenes import parse_bk2, extract_action_sequence

# Raw per-frame button states
frames = parse_bk2("path/to/clip.bk2")
# [{"RIGHT"}, {"RIGHT", "A"}, {"RIGHT", "A", "B"}, ...]

# Compressed action sequence (collapsed, noise-filtered)
actions = extract_action_sequence("path/to/clip.bk2", min_frames=3)
# ["R", "RJ", "rRJ", "rR", ...]
```

The 12 experimental scenes currently use pre-extracted placeholder sequences. These will be replaced with actual BK2 data once scene selection is finalized.

---

## Project structure

```
SMB-MSL/
├── pyproject.toml
├── README.md
├── SMB-MSL-brainstorm.md
├── berlot2020_task/
│   ├── __init__.py
│   ├── __main__.py          # Entry point and GUI dialog
│   ├── config.py            # All configurable parameters
│   ├── sequences.py         # 12 sequence definitions, pretrain generation, mirror functions
│   ├── display.py           # Visual stimuli, pacing line, and feedback screens
│   ├── data_logging.py      # TSV file writer
│   ├── task_training.py     # Training session logic
│   ├── task_test.py         # Behavioral test (right and left hand)
│   ├── task_pretrain.py     # Pre-training session
│   └── task_scan.py         # Scan session (paced and full-speed)
└── smb_ssl_task/
    ├── __init__.py
    ├── __main__.py          # Entry point, GUI dialog, mode dispatch
    ├── config.py            # All parameters (display, timing, input, paths)
    ├── scenes.py            # Scene definitions, BK2 parser, action vocabulary
    ├── input_handler.py     # Unified keyboard/gamepad -> NES button mapping
    ├── data_logging.py      # TSV writer (MSP + gameplay columns)
    ├── display.py           # Shared instruction/feedback/rest screens
    ├── msp.py               # MSP mode: ActionSequenceDisplay + chord/duration collection
    ├── game.py              # Gameplay mode: gym-retro wrapper + rendering
    ├── task_training.py     # Training session
    ├── task_test.py         # Behavioral test
    ├── task_pretrain.py     # Pre-training
    └── task_scan.py         # Scan session
```
