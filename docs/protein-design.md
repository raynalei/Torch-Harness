# Model service deployment

The Harness client uses independent model APIs. Configure endpoints and workflow
defaults in the deployment profile described in [Workflow configuration](workflow-configuration.md).
The [cluster profile](examples/cluster_deployment.yaml) shows the supported Slurm
settings and shared input/output paths. Set these to paths available to the client
and its model jobs.

The example resolves relative `sbatch_script` paths from the deployment YAML.
Keep it at `docs/examples/cluster_deployment.yaml` in the checkout alongside the
repository's `api/` directory. For a wheel installation, point
`plugins.config["protein-design"].deployment_config` at that checkout's YAML.
The scripts still require their configured cluster runtimes and shared paths.

Use [Model Registry](protein-model-tools.md) to register and inspect standalone
model requests. Use [Initial Round](initial-round.md) for workflow preflight,
background launch and resume. See [Autonomous Design Loop](autonomous-feedback.md)
for the available follow-up routes.

## Analyze saved results locally

The analysis modules read files on the client. Install the optional
`protein-design` dependencies for structure parsing and numerical routines.
The PLIP runner also requires the PLIP executable in the local environment.

- `python -m opendde_harness.plugin.protein_design.analysis.epitope_analysis --help`
- `python -m opendde_harness.plugin.protein_design.analysis.epitope_batch --help`
- `python -m opendde_harness.plugin.protein_design.analysis.plip_runner --help`
- `python -m opendde_harness.plugin.protein_design.analysis.evolution_analysis --help`
