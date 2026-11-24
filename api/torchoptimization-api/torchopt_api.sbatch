#!/bin/bash
#SBATCH --job-name=torchoptimization-api
#SBATCH --partition=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/grp/cmclab/leiy/logs/torchoptimization-api-%j.out

set -euo pipefail
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

# Python for the API service; model jobs use TORCHOPT_PYTHON below.
API_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python
export TORCHOPT_ROOT=/lustre/grp/cmclab/share/huangjm/Project/TorchOptimization_v2.0.13
export TORCHOPT_PYTHON=/lustre/grp/cmclab/share/wangmm/Optimization/env/torchx_env/bin/python
export TORCHOPT_MODEL_DIR="$TORCHOPT_ROOT/assets/torchcraft"
export TORCHOPT_RDE_DIR="$TORCHOPT_ROOT/assets/RDE-PPI"
export TORCHOPT_API_WORK=/lustre/grp/cmclab/leiy/torchoptimization_api_jobs
export TORCHOPT_ALLOWED_ROOT=/lustre/grp/cmclab
# Default partition for submitted optimization jobs; requests can override it.
export TORCHOPT_PARTITION=ggpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#gpu41,gpu43
cd /lustre/grp/cmclab/leiy/torchoptimization-api
exec "$API_PYTHON" -u -m uvicorn main:app --host 0.0.0.0 --port 8060
