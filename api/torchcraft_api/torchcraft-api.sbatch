#!/bin/bash
#SBATCH --job-name=torchcraft-api
#SBATCH --partition=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/grp/cmclab/leiy/logs/torchcraft-%j.out

set -e
source /lustre/grp/cmclab/share/anaconda3/etc/profile.d/conda.sh
conda activate torchx_env
cd /lustre/grp/cmclab/leiy/torchcraft_api

export TORCHX_ROOT=/lustre/grp/cmclab/leiy/TorchX
export TORCHCRAFT_API_WORK=/lustre/grp/cmclab/leiy/torchcraft_api_jobs
export TORCHCRAFT_ALLOWED_ROOT=/lustre/grp/cmclab
export TORCHCRAFT_MODEL_DIR=/lustre/grp/cmclab/leiy/

export TORCHCRAFT_PARTITION=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
export TORCHCRAFT_GRES=gpu:1
export TORCHCRAFT_CPUS=8
export TORCHCRAFT_MEM=32G
export TORCHCRAFT_TIME=12:00:00

srun python -m uvicorn main:app --host 0.0.0.0 --port 8020