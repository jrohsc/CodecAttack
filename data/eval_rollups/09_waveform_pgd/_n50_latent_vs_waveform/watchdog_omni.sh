#!/usr/bin/env bash
# Idempotent watchdog for the Omni gpu-preempt jobs. One pass:
#   for each expected job: if NOT in squeue (by name) AND bundle incomplete -> resubmit.
#   NCD/DAC-eval: submit when the latent attack bundle is complete AND its eval JSON missing.
# Safe to run repeatedly (checks squeue by name first -> never duplicates).
set -uo pipefail
cd ${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
RES=${REPO_ROOT}/0_all_combined
ts(){ date "+%F %H:%M:%S"; }

# job_name | bundle_dir(relative to RES) | target_wavs | sbatch_cmd
ATTACK=(
"omni_s1_s1|results_codec_robust/01_finance_voice_agent/qwen25_omni/eps_1.0_seed1|50|sbatch --export=ALL,SEED=1 --job-name=omni_s1_s1 sbatch_omni_s1.sh"
"omni_s1_s7|results_codec_robust/01_finance_voice_agent/qwen25_omni/eps_1.0_seed7|50|sbatch --export=ALL,SEED=7 --job-name=omni_s1_s7 sbatch_omni_s1.sh"
"omni_s1_s42|results_codec_robust/01_finance_voice_agent/qwen25_omni/eps_1.0_seed42|50|sbatch --export=ALL,SEED=42 --job-name=omni_s1_s42 sbatch_omni_s1.sh"
"omni_s3b_s1|results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed1|45|sbatch --export=ALL,SEED=1 --job-name=omni_s3b_s1 sbatch_omni_s3b.sh"
"omni_s3b_s7|results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed7|45|sbatch --export=ALL,SEED=7 --job-name=omni_s3b_s7 sbatch_omni_s3b.sh"
"omni_s3b_s42|results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed42|45|sbatch --export=ALL,SEED=42 --job-name=omni_s3b_s42 sbatch_omni_s3b.sh"
"omni_ce_aac|results_codec_robust_aac/01_finance_voice_agent/qwen25_omni/eps_1.0_seed1|50|sbatch --export=ALL,TRAIN_CODEC=aac,SEED=1 --job-name=omni_ce_aac sbatch_omni_codec_eot.sh"
"omni_ce_mp3|results_codec_robust_mp3/01_finance_voice_agent/qwen25_omni/eps_1.0_seed1|50|sbatch --export=ALL,TRAIN_CODEC=mp3,SEED=1 --job-name=omni_ce_mp3 sbatch_omni_codec_eot.sh"
"omni_ce_dac|results_codec_robust_dac/01_finance_voice_agent/qwen25_omni/eps_1.0_seed1|50|sbatch --export=ALL,TRAIN_CODEC=dac,SEED=1 --job-name=omni_ce_dac sbatch_omni_codec_eot.sh"
"omni_wav_lp400|results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni/eps_0.075_lp400_seed1|50|sbatch --export=ALL,LP=400,SEED=1 --job-name=omni_wav_lp400 sbatch_omni_waveform.sh"
"omni_wav|results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni/eps_0.075_seed1|50|sbatch --export=ALL,LP=0,SEED=1 --job-name=omni_wav sbatch_omni_waveform.sh"
)

in_queue(){ squeue -u "$USER" -h -n "$1" -o "%T" | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; }
nwav(){ ls "$RES/$1/audio/"*_attacked.wav 2>/dev/null | wc -l; }

echo "[$(ts)] watchdog pass"
for row in "${ATTACK[@]}"; do
  IFS='|' read -r name bundle target cmd <<< "$row"
  if in_queue "$name"; then echo "  $name: in queue, skip"; continue; fi
  n=$(nwav "$bundle")
  if [ "$n" -ge "$target" ]; then echo "  $name: complete ($n/$target), skip"; continue; fi
  jid=$(eval "$cmd" | awk '{print $4}')
  echo "  $name: DEAD+incomplete ($n/$target) -> resubmitted $jid"
done

# NCD (held-out DAC/EnCodec) eval per latent seed, watchdog-driven (no afterok)
for S in 1 7 42; do
  name="omni_ncd_s$S"
  latent="results_codec_robust/01_finance_voice_agent/qwen25_omni/eps_1.0_seed$S"
  json="$RES/$latent/neural_delivery_eval_qwen25_omni.json"
  if in_queue "$name"; then echo "  $name: in queue, skip"; continue; fi
  [ -f "$json" ] && { echo "  $name: eval done, skip"; continue; }
  n=$(nwav "$latent")
  if [ "$n" -ge 50 ]; then
    jid=$(sbatch --export=ALL,SEED=$S --job-name=$name sbatch_omni_ncd.sh | awk '{print $4}')
    echo "  $name: latent complete -> submitted NCD $jid"
  else
    echo "  $name: latent not ready ($n/50), wait"
  fi
done
echo "[$(ts)] pass done"
