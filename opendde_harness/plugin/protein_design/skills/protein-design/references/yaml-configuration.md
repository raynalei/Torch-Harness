# Torch workflow configuration

Build run_config for protein_design_start, or select a workflow_config YAML
profile. Inspect protein_model_list and the tool's JSON schema for current
request fields. The shared executor's internal InitialRoundConfig class and
initial_round directory name persist for checkpoint compatibility; no dedicated
initial-round command or model option is required.

Use generators keyed by ppiflow and/or torchcraft, with real service registration
and API request fields. When omitted, generators use target C, binder/heavy A, and light B for paired
antibodies, with an identity refold map. Configure evaluation.torchscore, torchfold, rosetta and
compatible torchopt services as required by the legal route. TorchScore supplies
both confidence scores; the generic start path defaults autonomous to an enabled
configuration. Missing required services fail validation rather than falling
back to the retired compute service.

Keep work_dir and path_mappings consistent with shared filesystem paths. Reuse
run_dir and saved remote IDs on recovery. Request sample counts come from the
scientific task and actual model schema; there is no imposed global rounds,
100-sample or 5-20 batch budget.

Legacy design/fold/compute YAML, compute_url, worker pools, GPU placement and
OpenDDE checkpoint settings are not this workflow's configuration. Do not copy
those fields from historical upstream examples. Consult docs/torch_skills.md
and the registered model request schema instead.
