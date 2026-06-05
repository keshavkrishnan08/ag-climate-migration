"""Audit improvement B: MLP on the monthly climate sequence + stacked ensemble.

Orthogonal architecture to the gradient-boosted trees. Two distinct ideas:

1. MLP over the raw MONTHLY climate sequence. A tree splits one feature at a time
   and approximates smooth multivariate climate response with axis-aligned steps.
   A small multi-layer perceptron fed the standardized monthly tmax/tmin/precip/PDSI
   vector (Apr-Sep) plus soil/latitude/trend can capture the smooth, interacting
   temperature x water response directly. We keep CLIMATE-ONLY inputs so it stays
   projectable from CMIP6 monthly deltas (no spatial-yield lags).
