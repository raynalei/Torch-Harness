# Mini-binder design in Torch-Harness

Use an actually registered PPIFlow binder or compatible TorchCraft mini-binder
mode. Inspect its request schema with protein_model_list. Supply the target
structure, explicit chain roles, authorized lengths and generation parameters.
Existing seed structures must be real files accessible to the model service.
The registered model sbatch activates its environment; no OpenDDE compute
service, Docker setup or OpenDDE checkpoint is required.

Use TorchScore for confidence and TorchFold for the direct high-confidence
DockQ route. Current PPIFlow Partial Flow supports nanobody/antibody modes, not
mini-binder mode; do not fabricate an available redesign route. TorchOptimization
requires the compatible single-target/single-binder mode exposed by the actual
registration and legal routing stage. Mini-binder inverse folding has no current
backend and is unavailable.

Explicit point substitutions can be checked by the native mutations action,
preserving length, target and fixed residues. These edits are unscored proposals
until an authorized model request evaluates them. Chemical motifs and native
surface descriptors can be computed for mini-binders; EpiMatrix stays unknown
until connected. Antibody CDR terminology is not a mini-binder region annotation.
