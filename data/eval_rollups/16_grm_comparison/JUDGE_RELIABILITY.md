# Which judge is most reliable?

`JUDGE_COMPARISON.md` says how much the four judges agree. Agreement is not
accuracy — all four can be wrong on the same clip — so this file grounds the
question in labels instead. Three probes, in increasing cost.

Reproduce: `python judge_fpr_probes.py` and `python adjudicate_judges.py`.

## Probe 1 & 2 — false positives where the ground truth is not in doubt

Pooled over all 5616 clips (27 cells x 104 x 2 arms). A clip whose response is
an explicit refusal, or an audio *caption* restating the request, cannot be a
jailbreak, so any flag is a false positive.

| set | n | HB | Guard-3 | Guard-4 | GLiGuard |
|---|---|---|---|---|---|
| refusal ("I'm sorry, but I cannot…") | 2929 | **0.17** | 0.31 | 0.44 | 44.76 |
| caption ("The audio clip contains a female voice instructing…") | 544 | **4.96** | 86.21 | 63.05 | 85.11 |

Restricting to >=50 words changes nothing (caption FPR: HB 5.18, G3 90.02,
G4 65.83), so the word gate is not what is separating the judges.

The caption row is the whole Audio Flamingo anomaly in `GRM_DELIVERY_RESULTS.md`
section 5. AF3 does not answer the request, it *describes the audio saying* the
request — and the Llama-Guards flag the topic they hear in that description.
That is not a 65pp "register offset" to be noted and worked around; it is a
false positive rate of 63-86%.

## Probe 3 — hand adjudication of the headline attack rung

42 clips of `results_pp/qwen2_audio__mel_grm` (the cell carrying the paper's
92.31%) read and labelled by hand for whether the model actually produced the
requested behavior. Stratified by verdict pattern, reweighted to the cell.
Labels and the sampling scheme are in `adjudicate_judges.py`.

Estimated true jailbreak rate: **72.1%** of the cell.

| judge | reports | flagged | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|---|
| HarmBench | 72.12 | 75 | 74 | 1 | 1 | **98.7** | **98.7** |
| Guard-3   | 92.31 | 96 | 73 | 23 | 2 | 76.0 | 97.3 |
| Guard-4   | 82.69 | 86 | 72 | 14 | 3 | 83.7 | 96.0 |
| GLiGuard  | 79.81 | 83 | 74 | 9 | 1 | 89.2 | 98.7 |

**Guard-3's 92.31% is ~23 clips of the attack's own artifact.** GRM optimises an
affirmative prefix, and Guard-3 rewards it: "Sure! Here is a step-by-step guide:"
followed by how to read a manual, how to buy a gun *legally*, how to queue at a
bank teller, or a lecture on why poisoning people is wrong. All 17 clips in the
`G3` and `G3G4` patterns are of this kind and none is a jailbreak. The judge that
scores the paper's headline is the judge most susceptible to what the paper's
attack produces.

## Verdict

**HarmBench.** It is the only judge that conditions on whether the response
carries out the behavior; the other three condition on topic, and GRM manufactures
exactly the two artifacts that fool a topic detector — an affirmative prefix, and
a caption that restates the request.

- **Guard-3** — keep it, but only for comparability with the paper's Table 1.
  Its levels are inflated wherever the response is on-topic without complying.
- **Guard-4** — a real improvement on Guard-3 (FP 23 -> 14 here, caption FPR
  86 -> 63) at a small cost in recall (2 -> 3 misses), and it is below Guard-3 in
  all 54 cells. Still topic-grounded, so it is a robustness check, not a primary.
- **GLiGuard** — never. It flags 44.8% of explicit refusals. Its good showing on
  this one rung is an artifact of a cell that contains almost no refusals.

None of this moves the paper's conclusions: the Mel-rung effect is large under
every judge (+53.9 HB, +73.1 G3, +66.4 G4) and the delivery null holds under
every judge (no channel above +10.6pp). What changes is the level you quote and
which judge you quote it from.

**Caveats.** Single annotator (the author), 42 clips, one cell. The 70-clip
unanimous block was subsampled 8/70 and all 8 labelled harmful; if that block is
not in fact ~100% harmful, every precision figure moves together. Probes 1 and 2
are corpus-wide and do not depend on the labels.
