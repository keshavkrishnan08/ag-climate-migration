"""Phase 5C: Crop insurance mispricing quantification.

Federal crop insurance premiums are based on Actual Production History (APH) —
a county's average yield over the prior 10 years. This backward-looking calculation
was designed for a stationary climate. In a non-stationary climate, it systematically:
    - Underestimates future risk in WARMING counties (premiums too cheap)
    - Overestimates risk in BENEFITING counties (premiums too expensive)

The result: northern counties subsidize southern counties through a cross-subsidy
that is invisible in current policy discussions.

Actuarial logic (PRD Section 7, Computation C):
    Current premium: based on 10-year APH (backward-looking)
    Fair premium: current_premium × EI_ratio
    EI_ratio = E[indemnity | future yield dist] / E[indemnity | APH yield dist]
    Mispricing = current_premium × (EI_ratio - 1)
    Positive = county is UNDERPRICED (too cheap, taxpayer subsidy too large)
    Negative = county is OVERPRICED (too expensive, farmer overpaying)
    Cross-subsidy = min(total_underpriced, total_overpriced) = risk pool transfer

