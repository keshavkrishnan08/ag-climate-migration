"""Close the magnitude question: test the migration claim on PRIME-AGE (25-54)
population, the outcome that responds to economic shocks (total population is diluted
by retirees, children, births/deaths). Prime-age county panel from Census PEP
(2010-2019 alldata + 2020-2023 agesex), no API key. Pre-COVID long difference
2010->2019 with the leave-one-out shift-share IV; farming-dependent counties. Seed 42."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/"src"/"revision"))
