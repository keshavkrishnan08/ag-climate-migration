"""R2-1b: pull county-level SSURGO soil properties via USDA Soil Data Access REST."""
import json, requests, pandas as pd, numpy as np
from pathlib import Path
OUT = Path("results/revision"); OUT.mkdir(parents=True, exist_ok=True)
# area-weighted available water storage (0-150cm) and clay% by survey area (areasymbol -> county FIPS)
q = ("SELECT l.areasymbol, "
     "SUM(mu.muacres) AS acres, "
     "SUM(m.aws0150wta*mu.muacres)/NULLIF(SUM(mu.muacres),0) AS aws0150, "
     "SUM(m.claytotal_r*mu.muacres)/NULLIF(SUM(mu.muacres),0) AS clay "
     "FROM legend l JOIN mapunit mu ON mu.lkey=l.lkey "
     "JOIN muaggatt m ON m.mukey=mu.mukey "
