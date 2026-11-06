# Cross-Codec Generalization: SNAC (Multi-Scale Neural Audio Codec)

**Status:** DEFERRED. Gated on `results_DAC/` v1 outcome AND `results_MIMI/` outcome.
**Authored:** 2026-05-05.

---

## 1. Purpose of this folder

Placeholder for a future cross-codec experiment using **SNAC** (Siuzdak's multi-scale neural audio codec, `pip install snac`) as the attack substrate, mirroring `results_DAC/` but adapted for SNAC-specific structure.

SNAC is the **third and most engineering-heavy** cross-codec test. It is gated on both DAC and Mimi succeeding first — SNAC's hierarchical multi-scale latent requires non-trivial changes to the perturbation tensor, so we don't invest unless the cross-codec story is already strong enough that SNAC adds material breadth.

## 2. Why SNAC specifically

- **Hierarchical multi-scale RVQ:** unlike EnCodec / DAC / Mimi (single (D, T) latent), SNAC produces three latent tensors at different temporal resolutions — 12 Hz, 23 Hz, 47 Hz — each carrying different scales of structure.
- **Disconfirmation power:** if our latent attack still works on SNAC (which has fundamentally different latent geometry), the bit-allocation theory generalizes beyond single-scale codecs.
- **Open weights, 24 kHz:** matches the rest of the codec lineup; same sample rate as EnCodec/DAC/Mimi.

## 3. What would change vs `results_DAC/`

Material differences from the parent design:

| Property | EnCodec / DAC / Mimi | SNAC |
|---|---|---|
| Latent shape | single tensor `[B, D, T_z]` | tuple of 3 tensors at 12 / 23 / 47 Hz |
| Perturbation δ | single tensor | **tuple of 3 tensors, each with its own ε bound** |
| ε projection | single L∞ clamp | per-scale L∞ clamp (or unified perceptual budget) |
| SNR-match search | 1-D binary search over single ε | multi-dimensional — either jointly optimize 3 εs or fix relative ratios and search a single scalar |

These changes touch:
- `latent_attack.py`: the `δ ← 0`, `δ + z`, `clamp(δ, -ε, ε)` operations all become tuple-aware.
- The codec wrapper's `encode_to_continuous` / `decode_from_continuous` return / accept a tuple.
- The SNR-match search becomes multi-dimensional (real complexity).

## 4. Estimated cost

- Engineering: 3–5 days for tuple-aware perturbation + multi-scale SNR-match.
- Compute: similar to DAC v1 (~6 GPU-hr) once the engineering is done.

## 5. What has to be true before we run this

- `results_DAC/` v1 succeeded (cross-codec story is real, not EnCodec-specific).
- `results_MIMI/` v1 succeeded (cross-codec story holds across two codec families).
- Reviewer feedback or paper deadline pressure makes a third codec worth the engineering cost.

## 6. Design questions to resolve before building this out

- Per-scale ε vs single unified ε: does the perceptual budget split cleanly across the three scales, or does optimizing them jointly require a different approach?
- Where does the cliff land? With three scales, "cliff at Opus 32k" is one of many possible answers — the answer might be "different cliff per scale" which is a richer (but more complex) story.
- SNAC's multi-scale structure may interact non-trivially with our perceptual mel loss — verify before claiming results.

## 7. References

- SNAC: https://github.com/hubertsiuzdak/snac
- Parent design: `0_all_combined/results_DAC/README.md`
- Companion deferred experiment: `0_all_combined/results_MIMI/README.md`
