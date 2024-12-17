"""Number consistency checker for AgMigration paper.

Verifies that all headline numbers appear correctly in paper TeX files
and that no stale values remain. Prints warnings for any mismatches.

Args:
    None (reads from canonical paths).
Returns:
    Exit code 0 if all checks pass, 1 if any warnings found.
Raises:
    FileNotFoundError if any required file is missing.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Load headline numbers
# ---------------------------------------------------------------------------
HEADLINE_JSON = PROJECT_ROOT / "state/headline_numbers_preliminary.json"
with open(HEADLINE_JSON) as f:
    h = json.load(f)

# ---------------------------------------------------------------------------
# Define checks
# ---------------------------------------------------------------------------
TEX_FILES = [
    PROJECT_ROOT / "paper/main.tex",
    PROJECT_ROOT / "paper/extended_data.tex",
    PROJECT_ROOT / "paper/supplementary_information.tex",
]

# (label, regex pattern that MUST appear in the file)
REQUIRED = {
    "hedonic $168B": r"\\\$168",
    "DCF conservative $56B": r"\\\$5[56][^0-9]",
