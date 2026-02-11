# Berlot et al. (2020) Motor Sequence Learning Task

PsychoPy implementation of the discrete sequence production (DSP) task from:

> Berlot, E., Popp, N. J., & Diedrichsen, J. (2020). A critical re-evaluation of fMRI signatures of motor sequence learning. *eLife*, 9, e55241.

## Installation

```bash
# Don't forget to upgrade pip if needed
pip install --upgrade pip
```

Requires Python 3.9+ and PsychoPy.
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

## Usage

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

## Output

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

## Configuration

Edit `berlot2020_task/config.py` to adjust:

- Display settings (screen size, fullscreen, colors, font sizes)
- Timing parameters (timeouts, inter-trial intervals, scan trial timing)
- Training parameters (trials per block, error threshold, point values)
- Key mappings (right and left hand)
- Scanner settings (trigger key, TR)
- Pacing line appearance (color, height, position)
- Pre-training parameters (number of sequences, repetitions)

## Project structure

```
SMB-MSL/
├── pyproject.toml
├── README.md
└── berlot2020_task/
    ├── __init__.py
    ├── __main__.py        # Entry point and GUI dialog
    ├── config.py          # All configurable parameters
    ├── sequences.py       # 12 sequence definitions, pretrain generation, mirror functions
    ├── display.py         # Visual stimuli, pacing line, and feedback screens
    ├── data_logging.py    # TSV file writer
    ├── task_training.py   # Training session logic
    ├── task_test.py       # Behavioral test session logic (right and left hand)
    ├── task_pretrain.py   # Pre-training session logic
    └── task_scan.py       # Scan session logic (paced and full-speed)
```
