"""Migration: overidentified multi-instrument 2SLS to tighten precision (#2).

Instead of one combined shift-share shock, use crop-specific leave-one-out
shift-share instruments (corn, soybeans, wheat) as a vector. Overidentification
improves efficiency (tighter CI), and the Hansen J statistic tests instrument
validity. Estimated on high farm-intensity counties. Seed 42.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
