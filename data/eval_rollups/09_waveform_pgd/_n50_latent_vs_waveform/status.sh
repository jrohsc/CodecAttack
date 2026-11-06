#!/usr/bin/env bash
# Live dashboard for the rebuttal seed runs. Run anytime: bash status.sh
REPO=${REPO_ROOT}
cd "$REPO"
echo "================ SLURM QUEUE ================"
squeue -u "$USER" -o "%.10i %.16j %.9T %.6M %.16R" | grep -E "JOBID|ms_|omni|lp400" || echo "(no jobs)"
echo
echo "================ BUNDLE PROGRESS (attacked/50 · sidecar/50) ================"
row() { # label  bundle-relpath  n
  local p="0_all_combined/$2"
  printf "  %-22s %s/%s · %s/%s\n" "$1" \
    "$(ls $p/audio/*_attacked.wav 2>/dev/null|wc -l)" "$3" \
    "$(ls $p/audio/*_codecs.json 2>/dev/null|wc -l)" "$3"
}
echo "-- Qwen2-Audio S1 (Q2+Q3) --"
row "latent   seed1"  results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0_seed1 50
row "latent   seed7*" results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0_seed7 50
row "latent   seed42" results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0_seed42 50
row "waveform seed1"  results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075_seed1 50
row "waveform seed7*" results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075_seed7 50
row "waveform seed42" results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075_seed42 50
row "lp400    seed1"  results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075_lp400_seed1 50
row "lp400    seed7"  results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075_lp400_seed7 50
row "lp400    seed42" results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075_lp400_seed42 50
echo "-- Qwen2.5-Omni S3b (Q3 flagship) --"
row "omni     seed1"  results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed1 45
row "omni     seed7"  results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed7 45
row "omni     seed42" results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed42 45
echo
echo "* seed7 latent/waveform = your terminal run (not SLURM)"
