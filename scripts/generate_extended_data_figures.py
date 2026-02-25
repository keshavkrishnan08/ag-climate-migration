"""
Generate Extended Data figures for Nature Food submission.

ED Figure 1: GCM Ensemble Spread by Region (fan charts)
ED Figure 2: Historical Cascade Score Distribution (bar chart)

Args: none
Returns: saves PDFs to results/figures/
Raises: FileNotFoundError if input data missing
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

