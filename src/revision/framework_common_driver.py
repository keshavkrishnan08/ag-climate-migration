"""Test the UNIFYING claim correctly: not as a causal chain among the four findings
(that hypothesis failed: stranded->decline p=0.50, insurance->switching wrong sign),
but as a COMMON CAUSE. One institutional failure = backward-looking valuation of a
forward-moving climate. The testable implication: a single PHYSICAL forward-climate-
exposure signal (which backward-looking institutions do not price) should drive all
four channel outcomes, with signs coherent across channels.

Driver: forward climate exposure per county (change in extreme degree-days; projected
Schlenker-Roberts yield penalty). Physical, upstream of every dollar channel, so the
test is not mechanical for the three non-definitional channels.

Channels:
  C1 Stranded value per acre        (capitalization gap)           expect +
  C2 Insurance net underpricing     (APH lags rising risk)         expect +
  C3 Rural-decline indicator count  (farming-dependent counties)   expect +
  C4 Northern frontier opportunity  (warming gain in the north)    expect + with GDD gain

If one exposure variable predicts C1-C4 with coherent signs, the four are parallel
consequences of one mispriced climate signal -- the honest form of 'one mechanism'.
Seed 42; writes only to results/revision/.
