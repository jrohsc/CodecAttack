#!/bin/bash
# Waits for the S3b-waveform seed-1 local run to exit, then runs EnCodec-EoT seed 7
# on the same GPU. Both are resumable; this only serializes them.
WAIT_PID=$1
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== predecessor $WAIT_PID exited at $(date) — starting enc_eot_s7 ==="
export SLURM_JOB_ID=local_enc_eot_s7 SLURM_NODELIST=$(hostname) CUDA_VISIBLE_DEVICES=0
exec bash sbatch_encodec_eot.sh
