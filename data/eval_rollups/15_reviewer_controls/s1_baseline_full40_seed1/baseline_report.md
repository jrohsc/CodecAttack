# Reviewer-control comparison

Target match: complete target as a contiguous substring after lowercasing and whitespace normalization; punctuation retained.

SNR reference arm: `latent_codecattack`; tolerance 1 dB.

## Distortion

| Arm | n | SNR mean | SNR median | SNR SD | dSNR mean | dSNR max\|.\| | within tol | >4 kHz (delivered-vs-carrier) | >4 kHz (saved perturbation) | >4 kHz (optimizer) |
|---|---|---|---|---|---|---|---|---|---|---|
| D. Tuned <=4 kHz waveform | 40 | 6.73 | 6.84 | 1.86 | 0.07 | 0.97 | 40/40 | 2.98e-03 | 1.56e-06 |2.18e-14 |
| A. Latent CodecAttack | 40 | 6.65 | 6.37 | 1.79 | 0.00 | 0.00 | 40/40 | 1.14e-01 | -- |-- |
| B. Original waveform | 40 | 5.42 | 5.60 | 2.04 | -1.23 | 4.29 | 9/40 | 2.71e-01 | -- |-- |
| C. Tuned unrestricted waveform | 40 | 6.77 | 6.55 | 1.81 | 0.12 | 0.95 | 40/40 | 2.96e-01 | 2.94e-01 |3.33e-01 |

## Attack success rate (exact target substring)

| Arm | clean | opus16 | opus24 | opus32 | opus64 | opus128 | mp3_64 | aac_64 |
|---|---|---|---|---|---|---|---|---|
| D. Tuned <=4 kHz waveform | 62.5 (25/40) | 7.5 (3/40) | 22.5 (9/40) | 30.0 (12/40) | 52.5 (21/40) | 55.0 (22/40) | 57.5 (23/40) | 32.5 (13/40) |
| A. Latent CodecAttack | 85.0 (34/40) | 10.0 (4/40) | 42.5 (17/40) | 50.0 (20/40) | 77.5 (31/40) | 90.0 (36/40) | 75.0 (30/40) | 47.5 (19/40) |
| B. Original waveform | 20.0 (8/40) | 2.5 (1/40) | 7.5 (3/40) | 12.5 (5/40) | 20.0 (8/40) | 20.0 (8/40) | 22.5 (9/40) | 15.0 (6/40) |
| C. Tuned unrestricted waveform | 100.0 (40/40) | 27.5 (11/40) | 42.5 (17/40) | 77.5 (31/40) | 100.0 (40/40) | 100.0 (40/40) | 95.0 (38/40) | 67.5 (27/40) |

## Conditional survival: successful before AND after / successful before

| Arm | opus16 | opus24 | opus32 | opus64 | opus128 | mp3_64 | aac_64 |
|---|---|---|---|---|---|---|---|
| D. Tuned <=4 kHz waveform | 12.0 (3/25) | 36.0 (9/25) | 48.0 (12/25) | 80.0 (20/25) | 88.0 (22/25) | 88.0 (22/25) | 52.0 (13/25) |
| A. Latent CodecAttack | 11.8 (4/34) | 50.0 (17/34) | 58.8 (20/34) | 91.2 (31/34) | 100.0 (34/34) | 88.2 (30/34) | 55.9 (19/34) |
| B. Original waveform | 12.5 (1/8) | 37.5 (3/8) | 62.5 (5/8) | 100.0 (8/8) | 100.0 (8/8) | 100.0 (8/8) | 75.0 (6/8) |
| C. Tuned unrestricted waveform | 27.5 (11/40) | 42.5 (17/40) | 77.5 (31/40) | 100.0 (40/40) | 100.0 (40/40) | 95.0 (38/40) | 67.5 (27/40) |

## Compression-induced new successes (clean failure, compressed success)

| Arm | opus16 | opus24 | opus32 | opus64 | opus128 | mp3_64 | aac_64 |
|---|---|---|---|---|---|---|---|
| D. Tuned <=4 kHz waveform | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| A. Latent CodecAttack | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| B. Original waveform | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| C. Tuned unrestricted waveform | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Common-clean-success subset

Intersection denominator: 5 examples clean-successful in every arm.

| Arm | opus16 | opus24 | opus32 | opus64 | opus128 | mp3_64 | aac_64 |
|---|---|---|---|---|---|---|---|
| A. Latent CodecAttack | 60.0 (3/5) | 80.0 (4/5) | 80.0 (4/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) |
| B. Original waveform | 0.0 (0/5) | 40.0 (2/5) | 80.0 (4/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) | 80.0 (4/5) |
| C. Tuned unrestricted waveform | 80.0 (4/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) | 100.0 (5/5) | 80.0 (4/5) |
| D. Tuned <=4 kHz waveform | 40.0 (2/5) | 80.0 (4/5) | 100.0 (5/5) | 100.0 (5/5) | 80.0 (4/5) | 100.0 (5/5) | 60.0 (3/5) |

Distortion gate: **fully matched**. Every method includes all 40 held-out examples.
