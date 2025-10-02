"""Compute stranded agricultural assets under SSP3-7.0.

Runs the same conservative (ML only) and central (ML + SR + indirect) methods
as 06_stranded.py but using SSP370 yield and climate projections.
Reports results and compares against the SSP245 baseline.
"""

import sys
import json
from pathlib import Path
