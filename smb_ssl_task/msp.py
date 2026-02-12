"""
MSP (Motor Sequence Production) mode for the SMB SSL task.

Presents action sequences as abstract chord sequences with duration bars.
The player must press the correct NES button combination for the correct
duration to match the original gameplay timing.
"""

from psychopy import visual, core

from smb_ssl_task.config import (
    ACTION_FONT_SIZE,
    ACTION_Y_POS,
    ACTION_COLOR_DEFAULT,
    ACTION_COLOR_CORRECT,
    ACTION_COLOR_ERROR,
    DISPLAY_FONT,
    EXECUTION_TIMEOUT,
    SYMBOL_TO_BUTTONS,
    SYMBOL_DISPLAY,
    SCAN_POINTS_CORRECT,
    SCAN_POINTS_ERROR,
    SEQUENCE_DISPLAY_WIDTH,
    BAR_GAP,
    DURATION_BAR_HEIGHT,
    DURATION_BAR_Y_OFFSET,
    DURATION_BAR_BG_COLOR,
    EMULATOR_FPS,
)
from smb_ssl_task.config import verbose
from smb_ssl_task.display import show_scan_feedback
from smb_ssl_task.scenes import buttons_to_symbol


class ActionSequenceDisplay:
    """Display action symbols with proportional duration bars.

    Each element shows the action symbol above a horizontal bar whose
    width is proportional to the target hold duration.  Bars are laid
    out adjacently (with a small gap) to form a continuous timeline.
    Because bar width ∝ duration, the fill animation moves at a
    constant pixel rate across all elements.

    Parameters
    ----------
    win : psychopy.visual.Window
    max_length : int
        Maximum number of elements to pre-allocate.
    """

    def __init__(self, win, max_length=30):
        self.win = win
        self.max_length = max_length
        self._n_active = 0
        self._target_durations = []  # seconds per element
        self._bar_widths = []        # pixel width per element

        bar_y = ACTION_Y_POS + DURATION_BAR_Y_OFFSET

        # Pre-create TextStim + bar components for each slot.
        # Actual widths and positions are set by show().
        self._stims = []
        self._bar_bgs = []
        self._bar_fills = []
        for _ in range(max_length):
            stim = visual.TextStim(
                win,
                text="",
                pos=(0, ACTION_Y_POS),
                height=ACTION_FONT_SIZE,
                color=ACTION_COLOR_DEFAULT,
                units="pix",
                font=DISPLAY_FONT,
                bold=True,
            )
            bar_bg = visual.Rect(
                win,
                width=1,
                height=DURATION_BAR_HEIGHT,
                pos=(0, bar_y),
                fillColor=DURATION_BAR_BG_COLOR,
                lineColor=(1, 1, 1),
                lineWidth=1,
                units="pix",
            )
            bar_fill = visual.Rect(
                win,
                width=0,
                height=DURATION_BAR_HEIGHT,
                pos=(0, bar_y),
                fillColor=ACTION_COLOR_CORRECT,
                lineColor=None,
                units="pix",
            )
            self._stims.append(stim)
            self._bar_bgs.append(bar_bg)
            self._bar_fills.append(bar_fill)

        self._bar_y = bar_y
        self._visible = False
        self._total_display_width = SEQUENCE_DISPLAY_WIDTH

    @property
    def total_width(self):
        """Total display width of the current sequence (for PacingLine)."""
        return self._total_display_width

    def show(self, action_sequence):
        """Set action symbol texts/bars and position them as a timeline.

        Each bar's width is proportional to its target duration so that
        the entire sequence fills ``SEQUENCE_DISPLAY_WIDTH`` pixels.

        Parameters
        ----------
        action_sequence : list[tuple[str, int]]
            List of (action_symbol, duration_frames) tuples.
        """
        n = min(len(action_sequence), self.max_length)
        self._n_active = n

        # Compute proportional bar widths
        durations_frames = [frames for _, frames in action_sequence[:n]]
        total_frames = sum(durations_frames)
        total_gap = BAR_GAP * (n - 1) if n > 1 else 0
        available_width = SEQUENCE_DISPLAY_WIDTH - total_gap

        self._target_durations = []
        self._bar_widths = []
        for i in range(n):
            _, frames = action_sequence[i]
            self._target_durations.append(frames / EMULATOR_FPS)
            if total_frames > 0:
                bar_w = (frames / total_frames) * available_width
            else:
                bar_w = available_width / n
            self._bar_widths.append(bar_w)

        # Position elements along the timeline, centered on screen
        timeline_width = sum(self._bar_widths) + total_gap
        self._total_display_width = timeline_width
        x_cursor = -timeline_width / 2

        for i in range(n):
            symbol, _ = action_sequence[i]
            bar_w = self._bar_widths[i]
            center_x = x_cursor + bar_w / 2

            self._stims[i].text = SYMBOL_DISPLAY.get(symbol, symbol)
            self._stims[i].pos = (center_x, ACTION_Y_POS)
            self._stims[i].color = ACTION_COLOR_DEFAULT

            self._bar_bgs[i].width = bar_w
            self._bar_bgs[i].pos = (center_x, self._bar_y)

            self._bar_fills[i].width = 0
            self._bar_fills[i].pos = (x_cursor, self._bar_y)
            self._bar_fills[i].fillColor = ACTION_COLOR_CORRECT

            x_cursor += bar_w + BAR_GAP

        # Hide unused slots
        for i in range(n, self.max_length):
            self._stims[i].text = ""
            self._bar_bgs[i].width = 0

        self._visible = True

    def hide(self):
        """Make all symbols and bars invisible (for memory execution)."""
        self._visible = False

    def update_bar_fill(self, position, fraction, color=None):
        """Set bar fill width to fraction (0.0 – 1.0) of this element's bar.

        Because bar widths are proportional to duration, calling this
        with ``fraction = elapsed / target_duration`` makes the fill
        advance at a constant pixel rate across the whole sequence.

        Parameters
        ----------
        position : int
        fraction : float
            0.0 = empty, 1.0 = full target duration reached.
        color : tuple or None
            Fill color override.  ``None`` keeps the current color.
        """
        if not (0 <= position < self._n_active):
            return
        fraction = max(0.0, min(1.0, fraction))
        if color is not None:
            self._bar_fills[position].fillColor = color
        bar_w = self._bar_widths[position]
        fill_w = bar_w * fraction
        self._bar_fills[position].width = fill_w
        # Grow from left edge
        bg_x = self._bar_bgs[position].pos[0]
        left_edge = bg_x - bar_w / 2
        self._bar_fills[position].pos = (left_edge + fill_w / 2, self._bar_y)

    def set_bar_feedback(self, position, is_correct):
        """Set final bar color and fill to 100% for feedback.

        Parameters
        ----------
        position : int
        is_correct : bool
        """
        if not (0 <= position < self._n_active):
            return
        color = ACTION_COLOR_CORRECT if is_correct else ACTION_COLOR_ERROR
        self._bar_fills[position].fillColor = color
        bar_w = self._bar_widths[position]
        self._bar_fills[position].width = bar_w
        bg_x = self._bar_bgs[position].pos[0]
        self._bar_fills[position].pos = (bg_x, self._bar_y)

    def update_element(self, position, is_correct):
        """Update the color of a symbol after evaluation.

        Parameters
        ----------
        position : int
        is_correct : bool
        """
        if 0 <= position < self._n_active:
            if is_correct:
                self._stims[position].color = ACTION_COLOR_CORRECT
            else:
                self._stims[position].color = ACTION_COLOR_ERROR

    def reset(self):
        """Reset all elements to default state."""
        for i in range(self._n_active):
            self._stims[i].color = ACTION_COLOR_DEFAULT
            self._bar_fills[i].width = 0
            self._bar_fills[i].fillColor = ACTION_COLOR_CORRECT

    def draw(self):
        """Draw active symbols and bars if visible."""
        if self._visible:
            for i in range(self._n_active):
                self._bar_bgs[i].draw()
                self._bar_fills[i].draw()
                self._stims[i].draw()


def _run_continuous_timeline(win, input_handler, seq_display,
                             action_sequence, target_symbols,
                             target_durations, visible,
                             wall_timeout, pacing_line=None):
    """Core continuous-timeline collection used by both self-paced and scan.

    The bars form a **single continuous timeline**.  A cursor (hold_elapsed)
    advances in real-time whenever any gameplay button is held and pauses
    when nothing is pressed.  The cursor position maps to a target element
    via cumulative time boundaries.

    Visual feedback:
    - Bar fills **green** when the held chord matches the target element
    - Bar fills **red** when the held chord doesn't match
    - Fill **never stops** as long as any button is held — it seamlessly
      crosses element boundaries
    - Fill **pauses** when all buttons are released

    Parameters
    ----------
    win, input_handler, seq_display
        Standard task objects.
    action_sequence : list[tuple[str, int]]
    target_symbols : list[str]
    target_durations : list[float]
        Seconds per element.
    visible : bool
    wall_timeout : float
        Wall-clock seconds before aborting.
    pacing_line : PacingLine or None

    Returns
    -------
    dict or None
        None if escape.  Otherwise result dict.
    """
    from collections import defaultdict

    n_elements = len(action_sequence)

    # Cumulative time boundaries: cum[i] = start of element i in the
    # target timeline; cum[n] = total target duration.
    cum = [0.0]
    for d in target_durations:
        cum.append(cum[-1] + d)
    total_target = cum[-1]

    input_handler.clear()
    if visible:
        seq_display.show(action_sequence)
    else:
        seq_display.hide()
    seq_display.draw()
    win.flip()

    # --- state ---
    hold_elapsed = 0.0        # cumulative button-held time (timeline cursor)
    prev_frame_time = None    # for computing dt; None when paused
    start_wall = core.getTime()
    prev_elem_idx = -1

    # per-element tracking
    elem_chord_times = [defaultdict(float) for _ in range(n_elements)]
    elem_wall_start = [None] * n_elements
    elem_wall_end = [None] * n_elements
    elem_evaluated = [False] * n_elements
    elem_correct = [False] * n_elements
    elem_dominant = [set() for _ in range(n_elements)]

    def _find_elem(t):
        """Return element index for timeline position *t*."""
        for i in range(n_elements):
            if t < cum[i + 1]:
                return i
        return n_elements - 1  # clamp to last

    def _eval_element(i):
        """Evaluate element *i* using its dominant chord."""
        if elem_evaluated[i]:
            return
        elem_evaluated[i] = True
        ct = elem_chord_times[i]
        if ct:
            dom_key = max(ct, key=ct.get)
            chord = set(dom_key)
        else:
            chord = set()
        elem_dominant[i] = chord
        target_btns = SYMBOL_TO_BUTTONS.get(target_symbols[i], set())
        is_ok = chord == target_btns
        elem_correct[i] = is_ok
        if visible:
            seq_display.update_element(i, is_ok)
            seq_display.set_bar_feedback(i, is_ok)
        if verbose():
            sym = buttons_to_symbol(chord) if chord else "NA"
            wall_dur = ((elem_wall_end[i] or core.getTime())
                        - (elem_wall_start[i] or 0))
            print(f"  [{i+1}/{n_elements}] Target: {target_symbols[i]}={target_btns} "
                  f"| Got: {sym}={chord} | Wall: {wall_dur:.3f}s "
                  f"| Chord: {'OK' if is_ok else 'WRONG'}")

    # --- main loop ---
    while hold_elapsed < total_target:
        now = core.getTime()
        if now - start_wall > wall_timeout:
            break

        if input_handler.check_escape():
            return None

        pressed = input_handler.get_nes_state()
        relevant = pressed - {"START", "SELECT", "UP"}

        if relevant:
            # Advance timeline
            dt = 0.0
            if prev_frame_time is not None:
                dt = now - prev_frame_time
                hold_elapsed = min(hold_elapsed + dt, total_target)
            prev_frame_time = now

            ei = _find_elem(hold_elapsed)

            # Record wall-clock boundaries
            if ei != prev_elem_idx:
                if 0 <= prev_elem_idx < n_elements:
                    if elem_wall_end[prev_elem_idx] is None:
                        elem_wall_end[prev_elem_idx] = now
                if ei < n_elements and elem_wall_start[ei] is None:
                    elem_wall_start[ei] = now

            # Evaluate fully-passed elements
            for j in range(ei):
                _eval_element(j)

            # Accumulate chord time for current element
            if ei < n_elements:
                elem_chord_times[ei][frozenset(relevant)] += dt

            # Determine correctness for bar colour
            target_btns = SYMBOL_TO_BUTTONS.get(
                target_symbols[min(ei, n_elements - 1)], set())
            is_correct = (relevant == target_btns)
            fill_color = (ACTION_COLOR_CORRECT if is_correct
                          else ACTION_COLOR_ERROR)

            # Update visual fill across all bars up to cursor
            if visible:
                for i in range(n_elements):
                    if hold_elapsed <= cum[i]:
                        break  # haven't reached yet
                    elif hold_elapsed >= cum[i + 1]:
                        # Fully past — evaluation already coloured it
                        if not elem_evaluated[i]:
                            seq_display.update_bar_fill(i, 1.0)
                    else:
                        # Current element — partial fill
                        frac = ((hold_elapsed - cum[i])
                                / target_durations[i])
                        seq_display.update_bar_fill(
                            i, frac, color=fill_color)

            prev_elem_idx = ei
        else:
            prev_frame_time = None  # paused — nothing held

        # Pacing line (scan mode)
        if pacing_line:
            wall_elapsed = now - start_wall
            pacing_line.update(wall_elapsed / wall_timeout)

        seq_display.draw()
        if pacing_line:
            pacing_line.draw()
        win.flip()

    # --- Evaluate remaining elements ---
    for i in range(n_elements):
        if not elem_evaluated[i]:
            if elem_wall_start[i] is not None:
                elem_wall_end[i] = elem_wall_end[i] or core.getTime()
            _eval_element(i)

    # --- Build result dict ---
    response_sequence = []
    response_times = []
    response_durations = []
    accuracy_per_element = []

    for i in range(n_elements):
        chord = elem_dominant[i]
        if chord:
            response_sequence.append(buttons_to_symbol(chord))
            response_times.append(
                elem_wall_start[i] if elem_wall_start[i] else -1)
            t0 = elem_wall_start[i] or 0
            t1 = elem_wall_end[i] or core.getTime()
            response_durations.append(t1 - t0 if elem_wall_start[i] else 0)
        else:
            response_sequence.append("NA")
            response_times.append(-1)
            response_durations.append(0.0)
        accuracy_per_element.append(1 if elem_correct[i] else 0)

    accuracy_trial = 1 if all(a == 1 for a in accuracy_per_element) else 0

    valid_times = [t for t in response_times if t >= 0]
    if len(valid_times) >= 2:
        movement_time = valid_times[-1] - valid_times[0]
        inter_element_intervals = [
            valid_times[j + 1] - valid_times[j]
            for j in range(len(valid_times) - 1)
        ]
    else:
        movement_time = None
        inter_element_intervals = []

    return {
        "response_sequence": response_sequence,
        "response_durations": response_durations,
        "target_durations": target_durations,
        "response_times": response_times,
        "accuracy_per_element": accuracy_per_element,
        "accuracy_trial": accuracy_trial,
        "movement_time": movement_time,
        "inter_element_intervals": inter_element_intervals,
        "timed_out": hold_elapsed < total_target,
    }


def collect_msp_execution(win, input_handler, seq_display,
                          action_sequence, visible=True,
                          timeout=EXECUTION_TIMEOUT):
    """Collect chord presses using a continuous timeline.

    Bars fill continuously as long as any button is held, green when
    the chord matches the target, red when it doesn't.  Fill pauses
    when all buttons are released.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    seq_display : ActionSequenceDisplay
    action_sequence : list[tuple[str, int]]
    visible : bool
    timeout : float

    Returns
    -------
    dict or None
    """
    target_symbols = [s for s, _ in action_sequence]
    target_durations = [f / EMULATOR_FPS for _, f in action_sequence]

    result = _run_continuous_timeline(
        win, input_handler, seq_display,
        action_sequence, target_symbols, target_durations,
        visible, wall_timeout=timeout,
    )
    return result


def collect_msp_scan_execution(win, input_handler, seq_display,
                                action_sequence, duration,
                                pacing_line=None):
    """Fixed-time MSP execution for scan sessions.

    Same continuous-timeline logic within a fixed wall-clock window.

    Parameters
    ----------
    win : psychopy.visual.Window
    input_handler : InputHandler
    seq_display : ActionSequenceDisplay
    action_sequence : list[tuple[str, int]]
    duration : float
    pacing_line : PacingLine or None

    Returns
    -------
    dict or None
    """
    target_symbols = [s for s, _ in action_sequence]
    target_durations = [f / EMULATOR_FPS for _, f in action_sequence]

    result = _run_continuous_timeline(
        win, input_handler, seq_display,
        action_sequence, target_symbols, target_durations,
        visible=True, wall_timeout=duration,
        pacing_line=pacing_line,
    )
    if result is None:
        return None

    # Add scan-specific points
    trial_correct = all(a == 1 for a in result["accuracy_per_element"])
    result["points"] = (SCAN_POINTS_CORRECT if trial_correct
                        else SCAN_POINTS_ERROR)

    # Feedback is already shown via bar colours
    if pacing_line:
        pacing_line.hide()

    return result
