#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PIPELINE_DIR="results/prinz_pipeline"
MEW_OG_CONFIG="mew_og/config/prinz_mew_og.json"
REWEIGHTING_CONFIG="mew_og/config/prinz_reweighting.json"
DATASET="trajectory"
SEED=1
BIASED_DDPM_N_SAMPLES=10000
BIASED_DDPM_BATCH_SIZE=10000
UNBIASED_N_SAMPLES=100000
GMM_N_COMPONENTS=4

BIASED_SAMPLES_H5="${PIPELINE_DIR}/biased_ddpm_samples.h5"
UNBIASED_SAMPLES_H5="${PIPELINE_DIR}/unbiased_prinz_samples.h5"
GMM_PARAMS="${PIPELINE_DIR}/gmm_params.npy"
REWEIGHTING_DIR="${PIPELINE_DIR}/reweighting"
MEW_OG_OUTPUT_DIR="${PIPELINE_DIR}/mew_og"
PLOT_ARGS=(--plot)

mkdir -p "${PIPELINE_DIR}" "${REWEIGHTING_DIR}" "${MEW_OG_OUTPUT_DIR}"

echo "Step 1/5: biased DDPM samples"
python scripts/sample_ddpm_prinz.py \
  --config "${MEW_OG_CONFIG}" \
  --output "${BIASED_SAMPLES_H5}" \
  --dataset_name "${DATASET}" \
  --n_samples "${BIASED_DDPM_N_SAMPLES}" \
  --batch_size "${BIASED_DDPM_BATCH_SIZE}" \
  --seed "${SEED}" \
  "${PLOT_ARGS[@]}"

echo "Step 2/5: unbiased Prinz samples"
python scripts/generate_prinz_trajectory.py \
  --output "${UNBIASED_SAMPLES_H5}" \
  --dataset_name "${DATASET}" \
  --n_samples "${UNBIASED_N_SAMPLES}" \
  --seed "${SEED}" \
  "${PLOT_ARGS[@]}"

echo "Step 3/5: GMM observable"
python scripts/fit_gmm_observable.py \
  --input "${UNBIASED_SAMPLES_H5}" \
  --dataset "${DATASET}" \
  --output "${GMM_PARAMS}" \
  --n_components "${GMM_N_COMPONENTS}" \
  --seed "${SEED}" \
  "${PLOT_ARGS[@]}"

echo "Step 4/5: reweighting lambdas"
python scripts/fit_lambdas_prinz.py \
  --config "${REWEIGHTING_CONFIG}" \
  --biased_data "${BIASED_SAMPLES_H5}" \
  --gt_data "${UNBIASED_SAMPLES_H5}" \
  --dataset "${DATASET}" \
  --gt_dataset "${DATASET}" \
  --seed "${SEED}"

echo "Step 5/5: MEW-OG training"
python scripts/train_mew_og_prinz.py \
  --config "${MEW_OG_CONFIG}" \
  --seed "${SEED}"

echo "Done: ${PIPELINE_DIR}"
