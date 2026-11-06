#!/usr/bin/env python3
"""Superseded by ``measure_reference_snr.py``.

Importing an existing bundle and measuring its realized SNR are the same
operation, and having two implementations of "reconstruct the carrier the
attack saw" is how the two arms ended up referenced to different carriers in
the first place (AUDIT.md A4, A5).  ``measure_reference_snr.py`` emits the same
JSONL schema that ``evaluate.py`` consumes, so it replaces this script outright.

The previous implementation is kept verbatim as ``import_existing.py.superseded``.
"""
import sys

MESSAGE = """\
import_existing.py is superseded. Use:

  python -m reviewer_controls.measure_reference_snr \\
      --scenario {s1|s3a} --split {development|evaluation|all} \\
      --method <arm name> --bundle <bundle dir> --output <out.jsonl>

It reconstructs the common 24 kHz peak-normalized carrier reference, reports
per-example SNR and >4 kHz energy, and lists missing or ambiguous examples in
<out>.summary.json instead of dropping them.
"""

if __name__ == "__main__":
    sys.exit(MESSAGE)
