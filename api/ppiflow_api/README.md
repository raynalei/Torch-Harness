# PPIFLOW Model API (Slurm-backed)

Unified HTTP API for 5 CLI entrypoints: antibody, nanobody, binder, partial_antibody, partial_nanobody. Each request submits an isolated Slurm job. **No modification** of the upstream PPIFLOW scripts is required.

## Setup

```bash
mkdir -p /lustre/grp/cmclab/leiy/ppiflow_api_jobs
# API environment only; inference runs with PPIFLOW_PYTHON (which must have torch and all PPIFLOW dependencies)
#python -m pip install -r requirements.txt
export PPIFLOW_ROOT=/lustre/grp/cmclab/leiy/v3_basic
export PPIFLOW_API_WORK=/lustre/grp/cmclab/leiy/ppiflow_api_jobs
export PPIFLOW_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python
export PPIFLOW_PARTITION=gpu3-12
export PPIFLOW_GRES=gpu:1
export PPIFLOW_CPUS=8
export PPIFLOW_MEM=32G
export PPIFLOW_TIME=12:00:00
uvicorn main:app --host 127.0.0.1 --port 8010
```

**Run the API on a node authorized to run `sbatch`**, not necessarily a GPU node. Inference happens only on allocated Slurm GPU nodes. Check `PPIFLOW_PYTHON`, model weights, filesystem visibility and actual resource needs first. If the original Python module invocation used the active `python` interpreter, point `PPIFLOW_PYTHON` to that same environment.

## Submit

```bash
curl -sS -X POST http://127.0.0.1:8010/v1/designs \
  -H 'Content-Type: application/json' \
  -d '{"mode":"antibody","name":"7n0a","antigen_pdb":"/lustre/grp/cmclab/share/guoly/app_code/test_data/antigen/clean_7n0a.pdb","antigen_chain":"C","specified_hotspots":"C4,C10,C117,C125","framework_dir":"/lustre/grp/cmclab/share/guoly/app_code/test_data/framework_scfv","heavy_chain":"A","light_chain":"B","cdr_length":"CDRH1,8-9,CDRH2,7-8,CDRH3,10-20,CDRL1,7-7,CDRL2,3-3,CDRL3,9-11","samples_per_target":5,"num_timesteps":200}'
```

Returns a `design_id` and `slurm_job_id` (202 Accepted). Then check and retrieve results:

```bash
curl http://127.0.0.1:8010/v1/designs/DESIGN_ID
curl http://127.0.0.1:8010/v1/designs/DESIGN_ID/results
```

Swagger docs: http://127.0.0.1:8010/docs

## Mode-specific fields

- `antibody`: antigen_pdb, antigen_chain, framework_dir, heavy_chain, light_chain, cdr_length; optional specified_hotspots.
- `nanobody`: antigen_pdb, antigen_chain, framework_dir, heavy_chain; optional cdr_length, specified_hotspots.
- `binder`: input_pdb, target_chain, binder_chain, samples_min_length, samples_max_length; optional specified_hotspots.
- `partial_antibody`: complex_pdb, antigen_chain, heavy_chain, light_chain, fixed_positions, cdr_position, start_t; optional specified_hotspots.
- `partial_nanobody`: complex_pdb, antigen_chain, heavy_chain, fixed_positions, cdr_position, start_t; optional specified_hotspots.

Common `name`, `samples_per_target`, and for non-binder modes `num_timesteps`.

The service sets a fixed mode-specific config and model_weights. Defaults must be audited against actual deployment; particularly `nanobody` defaults to `test_antibody.yaml`, and the partial-flow mode uses the antibody-v3 config shown in your examples.

## api functions

/health ['get']
/v1/designs ['post']
/v1/designs/{design_id} ['get']
/v1/designs/{design_id}/results ['get']
