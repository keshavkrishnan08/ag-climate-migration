# `src/revision/` — Reviewer Guide

The scripts in this directory produce every number cited in the revised manuscript. They are
grouped below by status. The `Makefile` at the repo root runs the **headline** and **robustness**
sets in dependency order. **Superseded** scripts are kept for transparency (showing what was
tried and abandoned) but are not part of the production pipeline.

## Headline scripts (cited in the paper)

Each produces a result that maps to a specific number in the manuscript. See `../../REPRODUCE.md`
