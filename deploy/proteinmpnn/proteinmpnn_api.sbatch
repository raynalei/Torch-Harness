#!/bin/bash
#SBATCH --job-name=proteinmpnn-api
#SBATCH --partition=gpu51
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=proteinmpnn-api-%j.out

set -euo pipefail
umask 077

cd "${SLURM_SUBMIT_DIR:?Submit from the proteinmpnn_api directory}"

export PROTEINMPNN_SIF="${PROTEINMPNN_SIF:-/lustre/grp/cmclab/leiy/containers/proteinmpnn.sif}"
export PROTEINMPNN_WEIGHTS="${PROTEINMPNN_WEIGHTS:-/lustre/grp/cmclab/leiy/models/proteinMPNN/v_48_020.pt}"
export PROTEINMPNN_WORK="${PROTEINMPNN_WORK:-/lustre/grp/cmclab/leiy/proteinmpnn_api_jobs}"
export PROTEINMPNN_PARTITION="${PROTEINMPNN_PARTITION:-gpu51}"
export PROTEINMPNN_GRES="${PROTEINMPNN_GRES:-gpu:1}"
export PROTEINMPNN_CPUS="${PROTEINMPNN_CPUS:-4}"
export PROTEINMPNN_MEM="${PROTEINMPNN_MEM:-16G}"
export PROTEINMPNN_TIME="${PROTEINMPNN_TIME:-02:00:00}"

api_python="${PROTEINMPNN_API_PYTHON:-/lustre/grp/cmclab/leiy/conda_envs/proteinmpnn_api/bin/python}"
mkdir -p "$PROTEINMPNN_WORK"
echo "ProteinMPNN API node: ${SLURMD_NODENAME:-$(hostname)} port=8090"
exec "$api_python" -u -m uvicorn main:app --host "${PROTEINMPNN_HOST:-0.0.0.0}" --port 8090
