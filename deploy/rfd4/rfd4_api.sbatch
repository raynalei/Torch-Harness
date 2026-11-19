#!/usr/bin/env bash
#SBATCH --job-name=rfd4-api
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=rfd4-api-%j.log
#SBATCH --error=rfd4-api-%j.log

set -euo pipefail

ROOT=${RFD4_ROOT:-/lustre/grp/cmclab/leiy}
export RFD4_SIF=${RFD4_SIF:-"$ROOT/containers/rfd4-proteina-cu12-amd64.sif"}
export RFD4_WEIGHTS=${RFD4_WEIGHTS:-"$ROOT/models/rfd4-proteina"}
export RFD4_INPUTS=${RFD4_INPUTS:-"$ROOT/rfd4_api_inputs"}
export RFD4_JOBS=${RFD4_JOBS:-"$ROOT/rfd4_api_jobs"}
export RFD4_APPTAINER=${RFD4_APPTAINER:-apptainer}
export PYTHONUNBUFFERED=1

cd "${RFD4_API_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -r main.py
test -s "$RFD4_SIF"
test -s "$RFD4_WEIGHTS/config.json"
test -s "$RFD4_WEIGHTS/model.safetensors"
mkdir -p "$RFD4_INPUTS" "$RFD4_JOBS"
echo "RFD4 API node: ${SLURMD_NODENAME:-$(hostname)} port=${RFD4_PORT:-8080}"
exec "${RFD4_API_PYTHON:-python3}" -u -m uvicorn main:app \
  --host "${RFD4_BIND:-0.0.0.0}" --port "${RFD4_PORT:-8080}" --lifespan on
