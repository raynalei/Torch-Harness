#!/bin/bash
#SBATCH --job-name=rosetta-api
#SBATCH --partition=gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=rosetta-api-%j.out

set -euo pipefail
cd "${ROSETTA_API_DIR:?Set ROSETTA_API_DIR to the shared directory containing main.py}"

# Option A: export ROSETTA_CONDA_ENV=<your actual env name or absolute env path>
# before submission. ROSETTA_CONDA_SH may specify conda's actual conda.sh path.
# Option B: submit from the already activated PyRosetta conda environment.
if [[ -n "${ROSETTA_CONDA_ENV:-}" ]]; then
    if [[ -z "${ROSETTA_CONDA_SH:-}" ]]; then
        conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
        if [[ -z "$conda_exe" ]]; then
            echo "Set ROSETTA_CONDA_SH or activate your PyRosetta environment before submitting." >&2
            exit 1
        fi
        ROSETTA_CONDA_SH="$("$conda_exe" info --base)/etc/profile.d/conda.sh"
    fi
    set +u
    source "$ROSETTA_CONDA_SH"
    conda activate "$ROSETTA_CONDA_ENV"
    set -u
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the PyRosetta conda environment, or set ROSETTA_CONDA_ENV before submitting." >&2
    exit 1
fi
# Always take Python from the active conda environment, including child jobs.
export ROSETTA_API_PYTHON="$CONDA_PREFIX/bin/python"
export PATH="$CONDA_PREFIX/bin:$PATH"
export ROSETTA_API_WORK="${ROSETTA_API_WORK:-$PWD/jobs}"
export ROSETTA_ALLOWED_INPUT_ROOTS="${ROSETTA_ALLOWED_INPUT_ROOTS:?Set ROSETTA_ALLOWED_INPUT_ROOTS to shared input roots separated by colons}"
export ROSETTA_PARTITION="${ROSETTA_PARTITION:-gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2}"
export ROSETTA_CPUS="${ROSETTA_CPUS:-4}"
export ROSETTA_MEM="${ROSETTA_MEM:-16G}"
export ROSETTA_TIME="${ROSETTA_TIME:-04:00:00}"
export OMP_NUM_THREADS="$ROSETTA_CPUS"
export SLURM_EXPORT_ENV=ALL

mkdir -p "$ROSETTA_API_WORK"
"$ROSETTA_API_PYTHON" -c 'import fastapi, pydantic, uvicorn, pyrosetta; assert int(pydantic.__version__.split(".")[0]) >= 2'
"$ROSETTA_API_PYTHON" -c 'import json, main; result = main.deployment_checks(); print(json.dumps(result)); raise SystemExit(0 if result["ok"] else 1)'
exec srun --ntasks=1 --export=ALL "$ROSETTA_API_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8050
