"""
Configuration parameters for the SMB Scene Sequence Learning (SSL) task.
"""

# --- Paths ---
GAME_NAME = "SuperMarioBros-Nes"
SETTINGS_FILE = ".smb_ssl_settings.json"  # Persistent settings (next to data/)

# --- Display ---
FULLSCREEN = True
MONITOR_NAME = "default"
BACKGROUND_COLOR = (-1, -1, -1)  # black
GAME_RENDER_SIZE = (768, 720)    # Scaled NES output for gameplay mode

# --- Action display (MSP mode) ---
ACTION_FONT_SIZE = 48
ACTION_SPACING = 70   # horizontal spacing between action symbol centers (pixels)
ACTION_Y_POS = 0      # vertical position of action row
ACTION_COLOR_DEFAULT = (1, 1, 1)    # white
ACTION_COLOR_CORRECT = (-1, 1, -1)  # green
ACTION_COLOR_ERROR = (1, -1, -1)    # red

# --- Font ---
DISPLAY_FONT = "DejaVu Sans"  # Unicode-capable (arrows: → ← ↓)

# --- Feedback / instruction text ---
TEXT_FONT_SIZE = 40
TEXT_COLOR = (1, 1, 1)  # white

# --- Keyboard -> NES mapping ---
KEY_TO_NES_KEYBOARD = {
    "right": "RIGHT",
    "left": "LEFT",
    "up": "UP",
    "down": "DOWN",
    "x": "A",
    "z": "B",
}

# --- Gamepad -> NES mapping ---
GAMEPAD_ENABLED = True
GAMEPAD_BUTTON_A = 0       # Face button index for A (jump)
GAMEPAD_BUTTON_B = 1       # Face button index for B (run)
GAMEPAD_DPAD_THRESHOLD = 0.5  # Analog stick threshold

# --- NES buttons (ordered as in gym-retro) ---
NES_BUTTONS = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]

# --- Action vocabulary ---
# Maps frozenset of NES buttons -> action symbol
ACTION_VOCABULARY = {
    frozenset():              "_",
    frozenset({"RIGHT"}):     "R",
    frozenset({"RIGHT", "B"}): "rR",
    frozenset({"A"}):         "J",
    frozenset({"RIGHT", "A"}): "RJ",
    frozenset({"RIGHT", "A", "B"}): "rRJ",
    frozenset({"LEFT"}):      "L",
    frozenset({"LEFT", "A"}): "LJ",
    frozenset({"LEFT", "A", "B"}): "rLJ",
    frozenset({"DOWN"}):      "D",
}

# Reverse mapping: action symbol -> set of NES buttons
SYMBOL_TO_BUTTONS = {v: set(k) for k, v in ACTION_VOCABULARY.items()}

# Display mapping: action symbol -> what to show on screen
# Uses arrow icons for directions, keyboard keys for buttons (X=jump, Z=run)
SYMBOL_DISPLAY = {
    "_":   "·",
    "R":   "→",
    "rR":  "→Z",
    "J":   "X",
    "RJ":  "→X",
    "rRJ": "→XZ",
    "L":   "←",
    "LJ":  "←X",
    "rLJ": "←XZ",
    "D":   "↓",
}

# --- Timing (seconds) ---
EXECUTION_TIMEOUT = 30.0       # max time for one MSP execution
INTER_EXECUTION_INTERVAL = 0.5 # pause between execution 1 and 2 within a trial
INTER_TRIAL_INTERVAL = 1.0     # pause between trials
FEEDBACK_DURATION = 0.8        # how long to show points after a trial
GAMEPLAY_MAX_DURATION = 10.0   # max seconds per gameplay attempt

# --- Training parameters ---
TRAINING_TRIALS_PER_BLOCK = 24  # 6 scenes x 4 reps
TRAINING_REPS_PER_SEQ = 4
ERROR_RATE_THRESHOLD = 0.15     # max error rate to update MT threshold
FAST_BONUS_FRACTION = 0.20      # >=20% faster than threshold for 3 points
POINTS_ERROR = 0
POINTS_CORRECT = 1
POINTS_FAST = 3

# --- Scanner settings ---
SCANNER_TRIGGER_KEY = "equal"  # '=' key
SCAN_PREP_DURATION = 1.0
SCAN_EXECUTION_DURATION = 5.0  # wider than Berlot (3.5s) for gameplay
SCAN_ITI = 0.5
SCAN_TRIAL_DURATION = SCAN_PREP_DURATION + SCAN_EXECUTION_DURATION + SCAN_ITI
SCAN_REPS_PER_SEQ = 6
SCAN_N_RUNS = 8
SCAN_REST_PERIODS = 5
SCAN_REST_DURATION = 10.0
SCAN_POINTS_CORRECT = 3
SCAN_POINTS_ERROR = 0

# --- Pacing line ---
PACING_LINE_COLOR = (1, 0.4, 0.7)  # pink
PACING_LINE_HEIGHT = 4
PACING_LINE_Y_OFFSET = -50  # below action symbols

# --- Duration bars (MSP mode) ---
SEQUENCE_DISPLAY_WIDTH = 900       # total pixel width of the action timeline
BAR_GAP = 2                        # pixel gap between adjacent duration bars
DURATION_BAR_HEIGHT = 8
DURATION_BAR_Y_OFFSET = -35        # below action symbols
DURATION_BAR_BG_COLOR = (0.4, 0.4, 0.4)  # dim gray background
TIMING_TOLERANCE = 0.050           # 50ms tolerance on hold duration
EMULATOR_FPS = 60                  # NES frame rate for duration conversion
CHORD_STABILIZE_TIME = 0.050       # 50ms (~3 frames) stabilization for chord transitions

# --- BK2 clip filtering ---
CLIP_MIN_DURATION_SEC = 3.0   # minimum total sequence duration in seconds
CLIP_MIN_ELEMENTS = 5         # minimum number of action elements
CLIP_MAX_ELEMENTS = 15        # maximum number of action elements
# Only these non-idle symbols are allowed in a clip (no DOWN, no LEFT+RIGHT)
CLIP_ALLOWED_SYMBOLS = {"R", "rR", "J", "RJ", "rRJ", "L", "LJ", "rLJ"}

# --- Pre-training ---
PRETRAIN_N_SCENES = 6
PRETRAIN_REPS_PER_SCENE = 4

# --- Data output ---
DATA_DIR = "data"
TSV_SEPARATOR = "\t"

# --- Escape ---
ESCAPE_KEY = "escape"

# --- Verbose output ---
_verbose = False


def set_verbose(v):
    """Set verbose mode (controls terminal debug output)."""
    global _verbose
    _verbose = v


def verbose():
    """Return True if verbose mode is enabled (--verbose / -v)."""
    return _verbose
