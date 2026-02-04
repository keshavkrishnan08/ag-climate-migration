"""Unit tests — PRD Section 10.1.

Required unit tests:
    1. GDD computation: within 5% of NASS reported GDD for Iowa corn 2012
    2. Yield detrending: detrended series has zero linear trend
    3. Temporal CV no leakage: val performance doesn't exceed train by >20%
    4. Crop switch detection: recovers known corn→soy switch in Ohio 2019
    5. Stranded asset monotone: higher RCP → larger stranded value
    6. Cascade threshold logic: county with all 4 conditions returns tipping year
    7. Insurance mispricing direction: southern overpriced, northern underpriced
    8. Deflation to 2023 USD: 2010 corn price deflated to 2023 matches BLS
    9. CMIP6 ensemble loading: all 10 models load without error
    10. County FIPS completeness: all 3,108 CONUS counties present
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


class TestGDDComputation:
    """Test GDD computation matches agronomic standards."""

    def test_gdd_corn_basic(self):
        """GDD for corn: base=10°C, upper=30°C."""
        from src.utils import deflator  # noqa — just checking import path structure

        # We import from features module
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

        # Manually test the GDD formula
        tmax = np.array([25.0, 30.0, 35.0, 40.0])
        tmin = np.array([15.0, 20.0, 25.0, 30.0])
        base, upper = 10.0, 30.0
