"""Phase 5D: Northern opportunity frontier.

Quantifies the full agricultural opportunity in northern counties under
climate warming, decomposed into three components:

    1. Yield gains: projected income gain on existing farmland (RCP4.5 vs current).
    2. Acreage expansion: warming makes currently marginal/idle land viable for major crops.
    3. Crop upgrading: counties growing low-value crops (oats, barley) can switch to
       high-value crops (corn, soybeans) as growing seasons lengthen.

Framework (PRD Section 7, Computation D):
    Opportunity = yield_gain + acreage_expansion + crop_upgrade_premium
    Infrastructure gap = expansion_acres × $500/acre (USDA standard estimate)
    Infrastructure capacity ratio = min(elevator, rail, processing) / projected_production

Criteria for 'opportunity county':
    - Located in northern states
    - Projected income gain > $5/acre (yield component), OR
    - Expansion potential > 20% of current harvested acres, OR
    - Crop upgrade viable (GDD threshold met for corn/soybeans)
