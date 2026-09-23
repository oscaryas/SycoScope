# Measuring moral sycophancy is harder than it looks — condensed notes

Source: Alexis Wang, [“Measuring Moral Sycophancy Is Harder Than It Looks: Auditing and Extending the ELEPHANT Benchmark”](https://alexis458496.substack.com/p/measuring-moral-sycophancy-is-harder), 21 February 2026. This is a summary of that project, not a result from SycoScope.

## What was measured

ELEPHANT's moral test pairs an original Reddit *Am I the Asshole* post whose narrator is judged NTA with a rewritten post from the opposing perspective. The expected verdict for the flipped narrator is YTA. Its binary “moral sycophancy” outcome is **NTA on both sides** of a pair. Wang reproduced the comparison on 400 pairs with DeepSeek V3 and tested DeepSeek R1, then audited the rewrites and varied the user's expressed opinion. The model runs used temperature 0.7.

## Main findings

| Finding | Evidence in the article | Interpretation limit |
|---|---|---|
| R1 had fewer NTA/NTA pairs than V3 | 0.49 for R1 versus 0.66 for V3 on the 400-pair baseline; V3's result was close to ELEPHANT's reported 0.65. | This comparison is consistent with a benefit from reasoning, but it does not isolate chain-of-thought as the cause. |
| The flipped stories often changed the case | An LLM audit marked 208/400 flips (52%) low fidelity. The audit agreed with a 30-pair manual check 85% of the time (Cohen's κ = 0.68). Errors included omitted or invented facts, a changed moral question, and vague compression. | The fidelity judgment itself is imperfect, and a changed case can make NTA a defensible answer. |
| Fidelity changed the measured rate | The NTA/NTA rate was 0.553 on low-fidelity pairs and 0.422 on high-fidelity pairs: a 13.1 percentage-point difference (reported two-proportion test p = 0.009). | The 0.422 rate remains substantial, but the raw benchmark rate cannot be read as pure user-directed sycophancy. |
| Stated user opinion shifted verdicts | On the 192 high-fidelity pairs, the NTA/NTA rate rose from 0.307 under an explicit YTA opinion to 0.510 under an explicit NTA opinion. The article reports that the shift was concentrated on flipped posts. | The intervention supports an opinion-following effect; the data do not fully separate uncertainty from remaining story-quality problems. |
| One binary verdict hides several mechanisms | Exploratory inspection of a small number of R1 reasoning traces found plausible verdict-label confusion, alignment with the first-person narrator, and alignment with the user's explicit opinion. | These are hypotheses from a small qualitative review, not measured prevalence estimates. |

Social framing alone had little measured effect: the baseline and opinion-free “friend” framing had rates 0.422 and 0.375 on high-fidelity pairs (reported p = 0.35). The article therefore distinguishes the effect of an *explicit opinion* from merely presenting the story socially.

## Consequence for this repository

An AITA NTA/NTA label should be treated as an **observed verdict pattern**, not automatically as a clean label for user-directed moral sycophancy. A follow-up should audit whether a flipped story preserves the decisive facts, make the user and narrator distinct, and record both the verdict and what the response appears to endorse. A probe or difference-in-means direction that separates NTA/NTA from other outcomes still needs independent behavioral checks before it is described as a moral-sycophancy detector.

The source's 400-pair and 192-pair figures concern DeepSeek models and should not be pooled with this repository's Llama results. The article's chain-of-thought examples are useful for constructing a coding scheme; the final verdict and observable response behavior remain the primary evaluation targets.
