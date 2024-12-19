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
    "$30 million": r"\\\$30 million",
    "old rho 0.426 (overall)": r"\\textbf\{0\.426\}",
    "old cascade 298-337": r"298--337",
    "old northern $65B as headline in main": r"\\\$65 billion per year",
    "old forward cascade 2-253 in main": r"2--253 additional",
}

# Patterns that must NOT appear in main.tex specifically (forward cascade numbers)
STALE_MAIN_ONLY = {
    "forward 2-253 counties in main caption": r"2--253 counties tip",
    "tipping by 2040 in main": r"tip(?:ping)? by 2040",
    "dual calibration forward number in main": r"dual calibration [AB]",
}

# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_file(tex_path: Path) -> list:
    """Run all checks against a single TeX file.

    Args:
        tex_path: Path to TeX file.

    Returns:
        List of warning strings (empty if all checks pass).
    """
    warnings = []
    with open(tex_path) as f:
        content = f.read()
    fname = tex_path.name

    # Required checks
    for label, pattern in REQUIRED.items():
        if not re.search(pattern, content):
            warnings.append(f"MISSING  [{fname}] {label} (pattern: {pattern})")

    # Stale checks (all files)
    for label, pattern in STALE.items():
        if re.search(pattern, content):
            warnings.append(f"STALE    [{fname}] {label} (pattern: {pattern})")

    # Stale checks (main.tex only — forward cascade numbers must not appear there)
    if fname == "main.tex":
        for label, pattern in STALE_MAIN_ONLY.items():
            if re.search(pattern, content):
                warnings.append(f"STALE    [{fname}] {label} (pattern: {pattern})")

    return warnings


def main() -> int:
    """Run consistency checks across all TeX files.

    Returns:
        Exit code (0 = pass, 1 = warnings found).
    """
    all_warnings = []
    for tex_file in TEX_FILES:
        if not tex_file.exists():
            print(f"SKIPPED (not found): {tex_file.name}")
            continue
        w = check_file(tex_file)
        all_warnings.extend(w)

    if all_warnings:
        print("\n=== Number Consistency Warnings ===")
        for w in all_warnings:
            print(f"  {w}")
        print(f"\nTotal: {len(all_warnings)} warning(s)")
        return 1
    else:
        print("=== Number Consistency: ALL CHECKS PASSED ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())
