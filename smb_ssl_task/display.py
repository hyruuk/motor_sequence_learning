"""
Shared display functions for the SMB SSL task.

Instruction screens, feedback, rest screens, and scan-session displays.
Mirrors the Berlot task display patterns.
"""

from psychopy import visual, event, core

from smb_ssl_task.config import (
    ACTION_FONT_SIZE,
    ACTION_COLOR_DEFAULT,
    ACTION_COLOR_CORRECT,
    ACTION_COLOR_ERROR,
    ACTION_Y_POS,
    DISPLAY_FONT,
    TEXT_FONT_SIZE,
    TEXT_COLOR,
    PACING_LINE_COLOR,
    PACING_LINE_HEIGHT,
    PACING_LINE_Y_OFFSET,
)


# --- Pacing line (reused from Berlot pattern) ---

class PacingLine:
    """Expanding pink line under the action symbols for paced scan sessions.

    Parameters
    ----------
    win : psychopy.visual.Window
    full_width : float
        Maximum width in pixels.
    """

    def __init__(self, win, full_width):
        self.win = win
        self.full_width = full_width
        self._rect = visual.Rect(
            win,
            width=0,
            height=PACING_LINE_HEIGHT,
            pos=(0, ACTION_Y_POS + PACING_LINE_Y_OFFSET),
            fillColor=PACING_LINE_COLOR,
            lineColor=PACING_LINE_COLOR,
            units="pix",
        )
        self._visible = False

    def reset(self):
        """Reset line to zero width and make visible."""
        self._rect.width = 0
        self._visible = True

    def update(self, fraction):
        """Set line width to fraction (0.0-1.0) of full width."""
        fraction = max(0.0, min(1.0, fraction))
        self._rect.width = self.full_width * fraction

    def show_go_cue(self):
        """Show short static line as go-cue (full-speed mode)."""
        self._rect.width = self.full_width * 0.1
        self._visible = True

    def hide(self):
        """Make invisible."""
        self._visible = False

    def draw(self):
        """Draw if visible."""
        if self._visible:
            self._rect.draw()


# --- Instruction / feedback screens ---

def show_instructions(win, text, keys=None):
    """Display instruction text and wait for a keypress to continue.

    Parameters
    ----------
    win : psychopy.visual.Window
    text : str
    keys : list[str] or None
        Keys to accept. None = any key.
    """
    msg = visual.TextStim(
        win,
        text=text,
        height=TEXT_FONT_SIZE,
        color=TEXT_COLOR,
        font=DISPLAY_FONT,
        units="pix",
        wrapWidth=win.size[0] * 0.8,
    )
    # Flush stale events so the screen isn't skipped
    event.clearEvents()
    msg.draw()
    win.flip()
    event.waitKeys(keyList=keys)


def show_trial_points(win, points, duration):
    """Briefly display points earned after a trial.

    Parameters
    ----------
    win : psychopy.visual.Window
    points : int
    duration : float
        Seconds to display.
    """
    if points == 0:
        color = ACTION_COLOR_ERROR
    elif points >= 3:
        color = ACTION_COLOR_CORRECT
    else:
        color = TEXT_COLOR

    msg = visual.TextStim(
        win,
        text=f"+{points}",
        height=ACTION_FONT_SIZE,
        color=color,
        font=DISPLAY_FONT,
        units="pix",
        bold=True,
    )
    msg.draw()
    win.flip()
    core.wait(duration)


def show_block_feedback(win, block_number, error_rate, median_mt, total_points):
    """Display end-of-block summary. Self-paced (wait for keypress).

    Parameters
    ----------
    win : psychopy.visual.Window
    block_number : int
    error_rate : float
    median_mt : float
    total_points : int
    """
    text = (
        f"Block {block_number} Complete\n\n"
        f"Error rate: {error_rate * 100:.0f}%\n"
        f"Median movement time: {median_mt:.2f}s\n"
        f"Total points: {total_points}\n\n"
        f"Press any key to continue"
    )
    msg = visual.TextStim(
        win,
        text=text,
        height=TEXT_FONT_SIZE,
        color=TEXT_COLOR,
        font=DISPLAY_FONT,
        units="pix",
        wrapWidth=win.size[0] * 0.8,
    )
    msg.draw()
    win.flip()
    event.waitKeys()


def show_rest(win):
    """Display a rest screen between blocks. Self-paced."""
    text = "Take a break.\n\nPress any key when ready to continue."
    msg = visual.TextStim(
        win,
        text=text,
        height=TEXT_FONT_SIZE,
        color=TEXT_COLOR,
        font=DISPLAY_FONT,
        units="pix",
        wrapWidth=win.size[0] * 0.8,
    )
    msg.draw()
    win.flip()
    event.waitKeys()


def show_scan_feedback(win, seq_display, points, remaining_time):
    """Show +N feedback for the remainder of the execution window.

    Parameters
    ----------
    win : psychopy.visual.Window
    seq_display : object
        Display object with a draw() method (action symbols remain visible).
    points : int
    remaining_time : float
    """
    if remaining_time <= 0:
        return

    if points == 0:
        color = ACTION_COLOR_ERROR
    else:
        color = ACTION_COLOR_CORRECT

    msg = visual.TextStim(
        win,
        text=f"+{points}",
        height=ACTION_FONT_SIZE,
        color=color,
        font=DISPLAY_FONT,
        pos=(0, ACTION_Y_POS + 80),
        units="pix",
        bold=True,
    )

    timer = core.CountdownTimer(remaining_time)
    while timer.getTime() > 0:
        seq_display.draw()
        msg.draw()
        win.flip()


def show_run_rest(win, run_number, n_runs):
    """Rest screen between runs. Self-paced."""
    text = (
        f"Run {run_number} of {n_runs} complete.\n\n"
        "Take a break.\n\n"
        "Press any key when ready for the next run."
    )
    msg = visual.TextStim(
        win,
        text=text,
        height=TEXT_FONT_SIZE,
        color=TEXT_COLOR,
        font=DISPLAY_FONT,
        units="pix",
        wrapWidth=win.size[0] * 0.8,
    )
    msg.draw()
    win.flip()
    event.waitKeys()


def show_fixation_rest(win, duration):
    """Timed rest period with fixation cross.

    Parameters
    ----------
    win : psychopy.visual.Window
    duration : float
        Seconds to show fixation.
    """
    cross = visual.TextStim(
        win,
        text="+",
        height=ACTION_FONT_SIZE,
        color=TEXT_COLOR,
        font=DISPLAY_FONT,
        units="pix",
        bold=True,
    )
    cross.draw()
    win.flip()
    core.wait(duration)


def show_waiting_for_scanner(win):
    """'Waiting for scanner...' screen until trigger received."""
    msg = visual.TextStim(
        win,
        text="Waiting for scanner...",
        height=TEXT_FONT_SIZE,
        color=TEXT_COLOR,
        font=DISPLAY_FONT,
        units="pix",
    )
    msg.draw()
    win.flip()
