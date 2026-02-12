"""
Data logging utilities for the SMB SSL task.

Writes one TSV row per execution (2 rows per trial).
Supports both MSP mode and gameplay mode columns.
"""

import os

from smb_ssl_task.config import DATA_DIR, TSV_SEPARATOR

COLUMNS = [
    "participant_id",
    "group",
    "session_type",
    "session_number",
    "block_number",
    "run_number",
    "trial_number",
    "scene_id",
    "mode",
    "execution_number",
    "condition",
    # MSP-specific
    "target_sequence",
    "response_sequence",
    "target_durations",
    "response_durations",
    "accuracy_per_element",
    "accuracy_trial",
    "movement_time",
    "inter_element_intervals",
    # Gameplay-specific
    "outcome",
    "traversal_time",
    "distance_reached",
    # Shared
    "points_awarded",
]


def _ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_session_dir(participant_id, session_type):
    """Return the output directory for a session, creating it if needed."""
    d = os.path.join(DATA_DIR, f"sub-{participant_id}", session_type)
    _ensure_dir(d)
    return d


class DataLogger:
    """Logs execution-level data to a TSV file.

    Parameters
    ----------
    participant_id : str
    group : int
    session_type : str
    session_number : int
    """

    def __init__(self, participant_id, group, session_type, session_number):
        self.participant_id = participant_id
        self.group = group
        self.session_type = session_type
        self.session_number = session_number

        out_dir = get_session_dir(participant_id, session_type)
        filename = f"sub-{participant_id}_{session_type}_ses-{session_number:02d}.tsv"
        self.filepath = os.path.join(out_dir, filename)

        self._file = open(self.filepath, "w")
        self._file.write(TSV_SEPARATOR.join(COLUMNS) + "\n")
        self._file.flush()

    def log_execution(
        self,
        block_number,
        trial_number,
        scene_id,
        mode,
        execution_number,
        condition="trained",
        run_number=0,
        # MSP fields
        target_sequence=None,
        response_sequence=None,
        target_durations=None,
        response_durations=None,
        accuracy_per_element=None,
        accuracy_trial=None,
        movement_time=None,
        inter_element_intervals=None,
        # Gameplay fields
        outcome=None,
        traversal_time=None,
        distance_reached=None,
        # Shared
        points_awarded=0,
    ):
        """Append one row (one execution) to the TSV file."""
        row = [
            self.participant_id,
            str(self.group),
            self.session_type,
            str(self.session_number),
            str(block_number),
            str(run_number),
            str(trial_number),
            str(scene_id),
            str(mode),
            str(execution_number),
            str(condition),
            # MSP
            _format_list(target_sequence),
            _format_list(response_sequence),
            _format_list(target_durations, fmt=".4f"),
            _format_list(response_durations, fmt=".4f"),
            _format_list(accuracy_per_element),
            str(accuracy_trial) if accuracy_trial is not None else "NA",
            f"{movement_time:.4f}" if movement_time is not None else "NA",
            _format_list(inter_element_intervals, fmt=".4f"),
            # Gameplay
            str(outcome) if outcome is not None else "NA",
            f"{traversal_time:.4f}" if traversal_time is not None else "NA",
            f"{distance_reached:.4f}" if distance_reached is not None else "NA",
            # Shared
            str(points_awarded),
        ]
        self._file.write(TSV_SEPARATOR.join(row) + "\n")
        self._file.flush()

    def close(self):
        """Close the output file."""
        if self._file and not self._file.closed:
            self._file.close()


def _format_list(lst, fmt=None):
    """Format a list as a semicolon-separated string for TSV storage."""
    if not lst:
        return "NA"
    if fmt:
        return ";".join(
            f"{x:{fmt}}" if isinstance(x, float) else str(x) for x in lst
        )
    return ";".join(str(x) for x in lst)
