"""Consolidate every headline number cited in the manuscript into one JSON.

Reads all per-experiment result JSONs in results/revision/ and writes a single
HEADLINE_NUMBERS.json that pairs each number cited in the paper with its source script,
input JSON, and computed value. Reviewers can regenerate everything with `make headline`
and grep this single file to verify any cited number against its provenance.

Seed 42; reads results/revision/*.json; writes results/revision/HEADLINE_NUMBERS.json.
"""
import json, glob, os
