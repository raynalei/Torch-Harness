#!/bin/bash
#SBATCH --job-name=torchscore-api
#SBATCH --partition=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/grp/cmclab/leiy/logs/torchscore-api-%j.out

set -euo pipefail
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export TORCHSCORE_ROOT=/lustre/grp/cmclab/leiy/TorchX
export TORCHSCORE_API_WORK=/lustre/grp/cmclab/leiy/torchscore_api_jobs
export TORCHSCORE_ALLOWED_INPUT_ROOTS=/lustre/grp/cmclab
export TORCHSCORE_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python

# Set exactly one weight source. This is a deployment path: verify before use.
export TORCHSCORE_MODEL_DIR=/lustre/grp/cmclab/leiy/
unset TORCHSCORE_CHECKPOINT
# For a TorchFold checkpoint, replace the two lines above with:
# unset TORCHSCORE_MODEL_DIR
# export TORCHSCORE_CHECKPOINT=/absolute/path/to/checkpoint.pt

# These resources apply to inference subjobs, independently of this API allocation.
export TORCHSCORE_PARTITION=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#gpu41,gpu43
export TORCHSCORE_GRES=gpu:1
export TORCHSCORE_CPUS=8
export TORCHSCORE_MEM=64G
export TORCHSCORE_TIME=12:00:00

cd /lustre/grp/cmclab/leiy/Torchscore-api
exec "$TORCHSCORE_PYTHON" -u -m uvicorn main:app --host 0.0.0.0 --port 8070
