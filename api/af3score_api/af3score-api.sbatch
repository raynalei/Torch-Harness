#!/bin/bash
#SBATCH --job-name=af3score-api
#SBATCH --partition=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/grp/cmclab/leiy/logs/af3score-%j.out

set -e
source /lustre/grp/cmclab/share/anaconda3/etc/profile.d/conda.sh
conda activate /lustre/grp/cmclab/leiy/conda_envs/af3score
cd /lustre/grp/cmclab/leiy/af3score_api

export AF3SCORE_ROOT=/lustre/grp/cmclab/leiy/AF3Score
export AF3SCORE_API_WORK=/lustre/grp/cmclab/leiy/af3score_api_jobs
export AF3SCORE_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/af3score/bin/python
export AF3SCORE_DB_DIR=/lustre/grp/cmclab/share/wangd/af3_data
export AF3SCORE_MODEL_DIR=/lustre/grp/cmclab/leiy/
export AF3SCORE_ALLOWED_INPUT_ROOTS=/lustre/grp/cmclab/leiy
export AF3_ENV=/lustre/grp/cmclab/leiy/conda_envs/af3score
export PATH="$AF3_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/lustre/grp/cmclab/leiy/AF3Score/src${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH="$AF3_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export AF3SCORE_PARTITION=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
export AF3SCORE_GRES=gpu:1
export AF3SCORE_CPUS=8
export AF3SCORE_MEM=32G
export AF3SCORE_TIME=12:00:00

srun python -m uvicorn main:app --host 0.0.0.0 --port 8030