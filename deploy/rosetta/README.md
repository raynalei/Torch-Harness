# LEGACY / RETIRED Rosetta HTTP API

**Do not deploy this directory for Torch-Harness.** The autonomous-score-routing
branch now requires direct Slurm PyRosetta for Rosetta evaluation. This legacy
HTTP server implementation is retained only for historical testing and migration;
no active plugin tool or workflow path is permitted to call it.

Use `docs/examples/cluster_deployment.yaml` for the `backend: slurm`
configuration. The supported direct native operation is interface analysis;
other former API operations are **not** implemented by the native worker.
