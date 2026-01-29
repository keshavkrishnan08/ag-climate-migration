"""Integration tests — PRD Section 10.2.

Required integration tests:
    1. test_full_pipeline_corn_iowa: Run complete pipeline for Iowa corn 1950-2050
    2. test_projection_monotone_in_rcp: RCP 8.5 > RCP 4.5 > RCP 2.6
    3. test_cascade_feedback_nonlinear: Feedback loop generates superlinear response
    4. test_insurance_aggregate_correct: Aggregate mispricing in $1-10B/year range
    5. test_northern_opportunity_positive: All opportunity counties have positive gain
    6. test_figure_generation: All 12 figures generate without error
    7. test_latex_compiles: paper/main.tex compiles to PDF
