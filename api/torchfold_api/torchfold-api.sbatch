#!/bin/bash
#SBATCH --job-name=torchfold-api
#SBATCH --partition=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/grp/cmclab/leiy/logs/torchfold-%j.out


source /lustre/grp/cmclab/leiy/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/grp/cmclab/leiy/conda_envs/torchx_env
cd /lustre/grp/cmclab/leiy/torchfold_api

export TORCHFOLD_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python
export TORCHFOLD_ROOT=/lustre/grp/cmclab/leiy/TorchX
export TORCHFOLD_API_WORK=/lustre/grp/cmclab/leiy/torchfold_api_jobs
export TORCHFOLD_ALLOWED_ROOT=/lustre/grp/cmclab
export TORCHFOLD_DB_DIR=/lustre/grp/cmclab/share/wangd/af3_data
unset TORCHFOLD_SIF  # run from source via conda python, no apptainer/sif

export TORCHFOLD_PARTITION=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#gpu41,gpu43
export TORCHFOLD_GRES=gpu:1 
export TORCHFOLD_CPUS=8
export TORCHFOLD_MEM=64G
export TORCHFOLD_TIME=12:00:00
export TORCHFOLD_MODEL_DIR=/lustre/grp/cmclab/leiy/

srun --cpu-bind=none  python -m uvicorn main:app --host 0.0.0.0 --port 8040