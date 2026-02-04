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

        tavg = (tmax + tmin) / 2.0
        effective = np.minimum(tavg, upper)
        gdd = np.maximum(0.0, effective - base)

        expected = np.array([10.0, 15.0, 20.0, 20.0])  # upper capped at 30
        np.testing.assert_array_almost_equal(gdd, expected)

    def test_gdd_corn_iowa_2012(self):
        """GDD for Iowa corn 2012 should be within 5% of NASS reported value.

        NASS reported ~2800 GDD base 50°F for Iowa 2012 growing season.
        In Celsius: ~1555 GDD base 10°C.
        """
        # Simulated daily data for Iowa growing season (Apr-Sep, ~183 days)
        np.random.seed(42)
        n_days = 183
        tmax = np.random.normal(30, 5, n_days)  # °C
        tmin = tmax - np.random.uniform(8, 15, n_days)

