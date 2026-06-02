"""R2-2d: full Revenue-Protection put with harvest-price reset (vs the yield-only
put), and R2-3b: residual mispricing across the full TAY-participation range.

R2-2d: RP guarantees revenue at APH*cov*max(P_proj, P_harvest); indemnity =
max(guarantee - yield*P_harvest, 0), a price-yield interaction. We Monte-Carlo the
harvest price (lognormal, negatively correlated with yield -- the natural hedge)
and recompute the CLIMATE mispricing (forward vs rolling-APH yield) under RP, then
compare to the yield-only (YP) put. If the RP price channel is climate-neutral, the
two climate-mispricing aggregates coincide, validating the closed-form yield put.

