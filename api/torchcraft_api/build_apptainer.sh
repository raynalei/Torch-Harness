#!/usr/bin/env bash
set -euo pipefail
BASE=/lustre/grp/cmclab/leiy
API="$BASE/torchcraft_api"
OUT="$BASE/containers/torchcraft.sif"
BUILD="$BASE/apptainer_build_torchcraft"

[[ -d "$BASE/TorchX/torchx_gpu" ]] || { echo "Missing $BASE/TorchX" >&2; exit 1; }
[[ -f "$API/requirements-api.txt" && -f "$API/torchcraft.def" ]] || { echo "Missing API build files" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "apptainer not found" >&2; exit 1; }
mkdir -p "$BASE/containers" "$BUILD/tmp" "$BUILD/cache"
export APPTAINER_TMPDIR="$BUILD/tmp"
export APPTAINER_CACHEDIR="$BUILD/cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
# Requires working fakeroot/rootless build support and outbound package downloads.
apptainer build --fakeroot "$OUT" "$API/torchcraft.def"
echo "Created: $OUT"
