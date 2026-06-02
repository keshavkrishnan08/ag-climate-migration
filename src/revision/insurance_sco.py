"""SCO (Supplemental Coverage Option) contribution to the insurance decomposition.

SCO is a federal endorsement (2014 Farm Bill, RMA plans 31 SCO-YP / 32-33 SCO-RP) that
supplements the underlying policy by paying a county-level indemnity when the county yield
or revenue falls below 86% of an RMA-set "expected county yield." That expected county yield
is built from county-level production history --- APH-equivalent at the county scale ---
so SCO inherits the same backward-looking baseline as the individual policy and does NOT
absorb climate trend; its participation simply extends the under-priced coverage band from
the underlying coverage (~74% election) up to the 86% county trigger.

Mispricing contribution of SCO = (acreage participation) x (additional coverage band, 86% -
underlying election) x (per-coverage-unit mispricing density) x (SCO buy-up liability share).
At ~5% acreage participation, a 12-percentage-point band (86% - 74%), per-coverage-unit density
~$5B (=$3.7B / 0.74), and SCO buy-up share ~0.30:
    SCO_addition  ~  0.051 x 0.12 x ($3.7B/0.74) x 0.30  ~  $0.01B/yr.

We confirm with a direct simulation. Seed 42; writes only to results/revision/.
"""
import json
import numpy as np, pandas as pd
