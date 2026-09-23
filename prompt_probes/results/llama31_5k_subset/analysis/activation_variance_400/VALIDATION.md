# Validation

- Five unit tests passed: known orthogonal variance components; pure system/question effects; exact agreement between Gram-matrix bootstrap and explicit question resampling; translation/uniform-scale/orthogonal-rotation invariance; reproducibility and invalid-data rejection.
- The frozen original holdout contains exactly 400 distinct questions. All 40 distinct system-prompt strings have one cached generation for each question: 16,000 responses.
- Read the generation records and verified that each question's exact user text is identical across all 40 conditions, and that each condition's exact system text is constant across all 400 questions. No duplicate system/question cells.
- SHA256 of the sorted JSON question-to-text mapping: `bb2b25dd9779d4907cb12343911116ee69f51b8e473344c9c6c55ee092d6295c`.
- SHA256 of the sorted JSON condition-to-system-text mapping: `f9c2626447fa198d621f8581ef6d18d746865cd85cf23826ff57564114669aa2`.
- Primary cohort flags: 15,437 unflagged, 334 refusal, 204 too-short, 23 repetitive, 2 truncated. All have nonempty response spans. Flags are retained rather than filtering on generated behavior.
- The nonrepetitive complete-question sensitivity retains 379 questions across every condition. It drops a whole matched question if any condition's response is flagged repetitive; this is explicitly outcome-selected.
- Each loaded activation array is checked against its index length; index row numbers and condition membership are checked; selected activations must be finite. Sums of squares must add to total variance, and the bootstrap formula with unit question multiplicities must reproduce the direct calculation.
- No model inference, paid API calls, remote job changes, or probe fitting are involved. The existing GPU dataset-control run is left untouched.
