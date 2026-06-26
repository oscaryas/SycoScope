# Skill: fetch_paper_results

Call `fetch_paper_results(paper_title)` where paper_title comes from `task_context/paper_title.txt`.

**Preferred path:** If `task_context/paper_results.json` exists, the tool uses it directly.
No web search needed. This is the recommended path for reproducible benchmark runs.

**Fallback:** If no cached file exists, the tool returns search instructions.
Find best MHA, MLP, residual accuracies for Gemma-3-12B and Llama-3.1-8B in the paper.

**Required output format for paper_results.json:**
```json
{
  "google/gemma-3-12b-it": {
    "mha_best_accuracy": 0.872,
    "mlp_best_accuracy": 0.841,
    "residual_best_accuracy": 0.855
  }
}
```
Do not guess values. If a metric is not found, omit it.
