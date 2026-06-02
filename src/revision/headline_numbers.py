"""Consolidate every headline number cited in the manuscript into one JSON.

Reads all per-experiment result JSONs in results/revision/ and writes a single
HEADLINE_NUMBERS.json that pairs each number cited in the paper with its source script,
input JSON, and computed value. Reviewers can regenerate everything with `make headline`
and grep this single file to verify any cited number against its provenance.

Seed 42; reads results/revision/*.json; writes results/revision/HEADLINE_NUMBERS.json.
"""
import json, glob, os
from pathlib import Path
OUT = Path("results/revision")

def jl(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None

def get(d, *path, default=None):
    cur = d
    for k in path:
        if cur is None: return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default

# Load every per-experiment JSON
master = jl("MASTER_NUMBERS.json") or {}
ranged = jl("dcf_ci_fixed.json") or {}
floor = jl("stranded_floor_sensitivity.json") or {}
hedonic = jl("hedonic_strengthened.json") or {}
dollar = jl("dollar_robustness.json") or {}
