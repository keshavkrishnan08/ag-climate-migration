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

