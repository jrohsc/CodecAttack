# Cross-Codec Generalization: Mimi (Kyutai)

**Status:** DEFERRED. Gated on `results_DAC/` v1 outcome.
**Authored:** 2026-05-05.

---

## 1. Purpose of this folder

Placeholder for a future cross-codec experiment using **Mimi** (the codec from Kyutai's Moshi project) as the attack substrate, mirroring the design in `results_DAC/README.md` but adapted for Mimi-specific differences.

Mimi is the **second** off-the-shelf neural codec we'd test. It is gated on DAC: if DAC tracks EnCodec's compression cliff at SNR-match (`results_DAC/`), Mimi extends the cross-codec story; if DAC fails, we re-design before adding Mimi.

## 2. Why Mimi specifically

- **Different temporal resolution:** 12.5 Hz frame rate vs EnCodec/DAC's 75 Hz. Tests whether the cliff depends on temporal granularity.
- **Transformer in encoder/decoder:** unlike EnCodec/DAC's pure-conv architecture. If the cliff still appears, that argues the cliff is bit-allocation-driven rather than architecture-specific.
- **Split semantic/acoustic RVQ:** Mimi factors the latent into a semantic codebook plus 31 acoustic codebooks (Moshi uses 1 + 7). Rich ablation territory.
- **Already partially wired:** `external/whisper-inject-v2-fork/attacks/latent_codec_mimi.py` exists with a `MimiCodecWrapper` for the PersonaPLEX/Moshi attack path. Most of the integration work is already done; the v1 experiment just needs the SNR-match procedure and results plumbing.

## 3. What would change vs `results_DAC/`

Compared to the DAC spec (which is the parent design), the Mimi-specific deltas are:

| Property | EnCodec | DAC | Mimi |
|---|---|---|---|
| Sample rate | 24 kHz | 24 kHz | 24 kHz |
| Frame rate | 75 Hz | 75 Hz | **12.5 Hz** |
| Continuous latent dim | 128 | 1024 | **512** |
| Quantizer | RVQ | RVQ | **Split RVQ (semantic + acoustic), with transformer encoder** |
| Wrapper status | exists | new | exists (`latent_codec_mimi.py`) |

The PGD inner loop is unchanged. The SNR-match procedure is unchanged. The only material difference for the v1 experiment is which codec wrapper is loaded.

## 4. What has to be true before we run this

- `results_DAC/` v1 has produced a clean cross-codec table.
- We have decided (based on DAC) whether to extend to a 2-codec or 3-codec story.
- Mimi's decode path is confirmed differentiable end-to-end *outside* the PersonaPLEX-specific soft-RVQ flow (the existing `MimiCodecWrapper` is intended for that; verify it works as a generic continuous-latent path).

## 5. Open questions to resolve before building this out

- Does the 12.5 Hz frame rate change the perturbation's spectral signature in a way that interacts with Opus/AAC bit allocation differently from 75 Hz codecs?
- Should we evaluate Mimi attacks against Qwen2.5-Omni only (matching DAC's setup) or also Moshi/PersonaPLEX (since Mimi is Moshi's native codec)? The latter is a different paper claim.

## 6. References

- Mimi codec: https://github.com/kyutai-labs/moshi
- Existing wrapper: `external/whisper-inject-v2-fork/attacks/latent_codec_mimi.py`
- Parent design: `0_all_combined/results_DAC/README.md`
- Memory pointers (in `MEMORY.md`): "PersonaPLEX/Moshi Attack Technical Details (2026-02-24)" — full Mimi config + soft-RVQ details
