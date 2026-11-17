"""
link_id_converter.py
--------------------
Maps each link index (0-69) to a human-readable ID of the form "<pair><direction>",
where <pair> is a number (1-35) shared by both links in a bidirectional pair, and
<direction> is 'N' for the link that travels most toward the north, 'S' for the other.

Pairs are detected from the dataset by matching link midpoints; within each pair the
link with the greater dy_norm (northward unit-vector component) receives the 'N' label.

Usage
-----
As a module:
    from link_id_converter import to_label, to_index, LINK_PAIRS

    to_label(34)    # e.g. '20N'
    to_index('20N') # 34
    to_index('20S') # 37

As a script:
    python link_id_converter.py
        -- prints the full mapping table

    python link_id_converter.py 34 37 '20N'
        -- converts the given indices/labels and prints results
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hard-coded mapping derived from dataset_speed_phase_excl_weights0_exttest.pkl
# Each entry: (pair_number, link_N, link_S)
# link_N  -- index of the northward link
# link_S  -- index of the southward link
# ---------------------------------------------------------------------------
_PAIRS: list[tuple[int, int, int]] = [
    (1,  0,  15),
    (2,  19,  1),
    (3,   2, 21),
    (4,   7,  3),
    (5,  22,  4),
    (6,   5, 16),
    (7,  20,  6),
    (8,   8, 10),
    (9,  26,  9),
    (10, 12, 11),
    (11, 24, 13),
    (12, 27, 14),
    (13, 17, 25),
    (14, 18, 28),
    (15, 23, 29),
    (16, 36, 30),
    (17, 31, 42),
    (18, 32, 44),
    (19, 33, 39),
    (20, 37, 34),
    (21, 40, 35),
    (22, 45, 38),
    (23, 41, 46),
    (24, 43, 47),
    (25, 48, 50),
    (26, 52, 49),
    (27, 53, 51),
    (28, 54, 59),
    (29, 56, 55),
    (30, 60, 57),
    (31, 67, 58),
    (32, 61, 68),
    (33, 62, 63),
    (34, 64, 65),
    (35, 69, 66),
]

# ---------------------------------------------------------------------------
# Build lookup dicts at import time
# ---------------------------------------------------------------------------
_INDEX_TO_LABEL: dict[int, str] = {}
_LABEL_TO_INDEX: dict[str, int] = {}

# Also expose a structured table for reference
LINK_PAIRS: list[dict] = []

for _pair, _n, _s in _PAIRS:
    _label_n = f"{_pair}N"
    _label_s = f"{_pair}S"
    _INDEX_TO_LABEL[_n] = _label_n
    _INDEX_TO_LABEL[_s] = _label_s
    _LABEL_TO_INDEX[_label_n] = _n
    _LABEL_TO_INDEX[_label_s] = _s
    LINK_PAIRS.append({"pair": _pair, "link_N": _n, "link_S": _s,
                        "label_N": _label_n, "label_S": _label_s})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_label(link_index: int) -> str:
    """Convert an integer link index (0-69) to its pair label, e.g. '20N'.

    Parameters
    ----------
    link_index : int
        An integer in [0, 69].

    Returns
    -------
    str
        A string like '1N', '20S', etc.

    Raises
    ------
    KeyError
        If *link_index* is not a known link.
    """
    if link_index not in _INDEX_TO_LABEL:
        raise KeyError(f"Unknown link index: {link_index!r}. Must be 0-69.")
    return _INDEX_TO_LABEL[link_index]


def to_index(label: str) -> int:
    """Convert a pair label (e.g. '20N') to the integer link index.

    Parameters
    ----------
    label : str
        A string of the form '<pair><direction>', e.g. '20N' or '20S'.

    Returns
    -------
    int
        The corresponding link index (0-69).

    Raises
    ------
    KeyError
        If *label* is not a valid pair label.
    """
    label = label.strip()
    if label not in _LABEL_TO_INDEX:
        raise KeyError(f"Unknown label: {label!r}.")
    return _LABEL_TO_INDEX[label]


def convert(value: int | str) -> str | int:
    """Auto-detect input type and convert in the appropriate direction.

    * ``int`` input  → calls :func:`to_label`
    * ``str`` input  → calls :func:`to_index`
    """
    if isinstance(value, int):
        return to_label(value)
    if isinstance(value, str):
        return to_index(value)
    raise TypeError(f"Expected int or str, got {type(value).__name__!r}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table() -> None:
    header = f"{'pair':>4}  {'link_N':>6}  {'label_N':>7}    {'link_S':>6}  {'label_S':>7}"
    print(header)
    print("-" * len(header))
    for entry in LINK_PAIRS:
        print(f"{entry['pair']:>4}  {entry['link_N']:>6}  {entry['label_N']:>7}    "
              f"{entry['link_S']:>6}  {entry['label_S']:>7}")


def _main(args: list[str]) -> None:
    if not args:
        _print_table()
        return

    for arg in args:
        try:
            # Try integer first
            idx = int(arg)
            print(f"{idx} -> {to_label(idx)}")
        except ValueError:
            # Treat as label
            try:
                print(f"{arg} -> {to_index(arg)}")
            except KeyError as exc:
                print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    _main(sys.argv[1:])
