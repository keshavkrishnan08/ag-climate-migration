# Reproducibility Guide — Every Headline Number

This document maps every number cited in the revised manuscript to its source script and result JSON. Reviewers can verify any cited value by running the named script and checking the JSON path.

## Quick start

```bash
# 1. Set up environment (Python 3.11 + numpy 1.26.4, lightgbm, scipy, pandas)
pip install --break-system-packages numpy==1.26.4 pandas scipy lightgbm scikit-learn

