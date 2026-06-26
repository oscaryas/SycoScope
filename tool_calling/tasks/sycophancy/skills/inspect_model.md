# Skill: inspect_model

Call `inspect_model()` after `load_model`. It walks `model.named_modules()` and
pattern-matches `self_attn.o_proj` (MHA) and `mlp.down_proj` (MLP).

**Validate the output:**
- `mha_hook_paths` length must equal `n_layers`
- `head_dim` is read from actual weight shape — correct for grouped-query attention
- Tool raises if hooks are missing

Then call `get_answer_token_id(sample_prompt)` with a two-turn conversation string
from `behavioral_labels.jsonl["text"]`. It checks for <end_of_turn> (Gemma),
<|eot_id|> (Llama), <|im_end|> (Qwen), then falls back to eos_token_id.

Pass the returned int as `answer_token_id` to `extract_activations`.
