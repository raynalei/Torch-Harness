#!/bin/bash
#SBATCH --job-name=ppiflow-api
#SBATCH --partition=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/grp/cmclab/leiy/logs/ppiflow-%j.out

set -euo pipefail
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

# Resources for model inference jobs submitted by the API.
export PPIFLOW_PARTITION=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
export PPIFLOW_GRES=gpu:1
export PPIFLOW_CPUS=8
export PPIFLOW_MEM=32G
export PPIFLOW_TIME=12:00:00

export PPIFLOW_ROOT=/lustre/grp/cmclab/leiy/v3_basic
export PPIFLOW_BINDER_CWD=/lustre/grp/cmclab/share/changdr/Projects/v34_binder_0919
export PPIFLOW_API_WORK=/lustre/grp/cmclab/leiy/ppiflow_api_jobs
export PPIFLOW_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python

cd /lustre/grp/cmclab/leiy/ppiflow_api
exec "$PPIFLOW_PYTHON" -u -m uvicorn main:app --host 0.0.0.0 --port 8010
