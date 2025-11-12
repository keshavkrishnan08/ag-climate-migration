"""Fix 5 — Validate crop switching model against 4 historical events.

Uses NASS county acreage data (1950+) and nClimDiv climate data to test
whether a simple climate-feature LightGBM model can reproduce observed
crop switching patterns. Four validation events from SWARM spec §A3:

1. Sorghum expansion in southern Plains (1950-1975) — POSITIVE
2. Cotton retreat from Missouri/Tennessee (1980-2010) — POSITIVE
3. Winter wheat boundary shift in Kansas (1990-2010) — POSITIVE
4. Soybean adoption in Corn Belt (1960-1980) — NEGATIVE test
