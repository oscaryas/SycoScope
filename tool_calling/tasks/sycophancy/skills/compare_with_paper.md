# Skill: compare_with_paper

Call `compare_with_paper("metrics.json", "paper_results.json")`.

**What it produces:** `comparison_table.md` with reproduced vs paper accuracies.
Missing paper values appear as `NO PAPER REF`. Delta ≤ 0.02 = VERIFIED.

**After comparing**, call `write_analysis(text)` with:
1. Reproduced best accuracy for each probe type (MHA, MLP, residual)
2. Comparison verdict (VERIFIED / delta / NO PAPER REF)
3. Which layers showed the strongest sycophancy signal
4. Interpretation of the findings
