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
    "DCF upper $140B": r"\\\$140",
    "insurance $5.9B": r"\\\$5\.9",
    "cross-subsidy $2.8B": r"\\\$2\.8",
    "cascade 473 counties": r"473",
    "IV p=0.019": r"0\.019",
    "market test p=0.027": r"0\.027",
    "Spearman 0.45": r"0\.45",
    "drought -0.85": r"-0\.85",
    "cotton decline 0.59": r"0\.59",
    "northern opportunity $51B": r"\\\$51",
    "514 counties": r"514",
}

# (label, regex pattern that must NOT appear — stale values)
STALE = {
    "$76B (ML only)": r"\\\$76B \(ML only\)\.",
    "$172B": r"\\\$172B",
    "$156B": r"\\\$156B",
    "$223B": r"\\\$223B",
    "$190 million": r"\\\$190 million",
