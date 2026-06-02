"""R1-v: regenerate Fig 7B as a MARGINAL-EFFECTS panel (not coincident counts).
Panel A: geographic distribution of decline-indicator count in farming-dependent
counties. Panel B: marginal effect of a 1-SD adverse yield trend on each decline
indicator (linear probability model, farming-dependent counties, HC1 SE)."""
import numpy as np, pandas as pd, json
from pathlib import Path
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("."); OUT=ROOT/"results/revision"; FIG=ROOT/"results/figures_revision"
