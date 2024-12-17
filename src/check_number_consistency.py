"""Number consistency checker for AgMigration paper.

Verifies that all headline numbers appear correctly in paper TeX files
and that no stale values remain. Prints warnings for any mismatches.

Args:
    None (reads from canonical paths).
Returns:
    Exit code 0 if all checks pass, 1 if any warnings found.
Raises:
