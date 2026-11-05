"""Persistent dual-generator initial round, separate from the existing evolution loop."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from opendde_harness.utils.portable_lock import file_lock

from .artifacts import (
    chain_sequences,
    measured,
    metric_records,
    output_files,
    prepare_fold,
    refold_output,
    table_records,
    write_pdb,
)
from .config import InitialRoundConfig
from .services import SlurmService, SubmissionUncertainError
from .storage import Checkpoint, SharedFiles, select, write_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InitialRoundRunner:
    def __init__(self, config: InitialRoundConfig, *, transport=None, attach_jobs: dict[str, str] | None = None):
        self.config = config
        self.files = SharedFiles(config.path_mappings)
        self.root = config.work_dir / config.name
        self.transport = transport
        self.attach_jobs = attach_jobs or {}
        self.services = {}

    def validate_files(self):
        self.input_hashes = {}
        if importlib.util.find_spec("biotite") is None:
            raise RuntimeError("install optional dependencies with uv sync --locked --extra dev --extra initial-round")
        if not self.config.evaluation.dockq_python.is_absolute() or not self.config.evaluation.dockq_python.is_file():
            raise ValueError(
                "evaluation.dockq_python must name an existing absolute interpreter path in an isolated DockQ environment"
            )
        self.files.remote(self.root)
        path_keys = {
            "ppiflow": ["antigen_pdb", "input_pdb", "framework_dir"],
            "torchcraft": ["pdb_path", "target_full_seq", "framework_cif_path"],
        }
        target_sequences = []
        for source, generator in self.config.generators.items():
            if not generator.fold_template.is_absolute() or not generator.fold_template.is_file():
                raise ValueError(f"{source}: fold_template must be an existing absolute local path")
            self.input_hashes[f"{source}/fold_template"] = digest(generator.fold_template)
            for key in path_keys[source]:
                if key in generator.request:
                    path = self.files.local(generator.request[key])
                    if not path.exists():
                        raise ValueError(f"{source}: input {key} is not accessible: {path}")
                    inputs = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
                    for item in inputs:
                        self.input_hashes[f"{source}/{key}/{self.files.remote(item)}"] = digest(item)
            # Validate template shape before submitting expensive GPU jobs.
            template = json.loads(generator.fold_template.read_text())
            proteins = {p["protein"]["id"]: p["protein"]["sequence"] for p in template.get("sequences", [])}
            sequences = {original: proteins[folded] for original, folded in generator.chains.refold_map.items()}
            target_sequences.append(tuple(sorted(sequences[chain] for chain in generator.chains.target)))
            # This checks mapping, dialect and embedded paths without writing the template.
            probe = self.root / f"{source}-template-check.json"
            prepare_fold(generator.fold_template, sequences, generator.chains, "preflight", probe)
            prepared = json.loads(probe.read_text())
            if not self.config.evaluation.torchfold_options.get("run_data_pipeline", False):
                for entity in prepared["sequences"]:
                    protein = entity["protein"]
                    if any(key not in protein and key + "Path" not in protein for key in ("unpairedMsa", "pairedMsa")):
                        raise ValueError(
                            "run_data_pipeline=false requires explicit prepared MSAs (or explicit empty strings)"
                        )
            probe.unlink()
        if len(set(target_sequences)) != 1:
            raise ValueError("both generators must design against the same target chain sequences")

    async def check(self):
        self.validate_files()
        process = await asyncio.create_subprocess_exec(
            str(self.config.evaluation.dockq_python),
            str(Path(__file__).with_name("dockq.py")),
            "--check",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError("DockQ interpreter cannot import DockQ; install the isolated tool environment")
        tools = json.loads(stdout)
        previous = self.checkpoint.data.get("tools")
        if previous is not None and previous != tools and self.checkpoint.data["jobs"]:
            raise ValueError("DockQ tool versions changed; use a new run directory")
        self.checkpoint.data["tools"] = tools
        self.checkpoint.save()
        configs = {name: g.service for name, g in self.config.generators.items()}
        configs.update({name: getattr(self.config.evaluation, name) for name in ("af3score", "torchfold", "rosetta")})
        result = {}
        for name, config in configs.items():
            service = SlurmService(config, self.checkpoint, transport=self.transport)
            self.services[name] = service
            health = await service.get(config.health_path)
            if (
                not isinstance(health, dict)
                or health.get("ok") is False
                or str(health.get("status", "ok")).lower() in {"error", "failed", "unhealthy"}
            ):
                raise RuntimeError(f"{name}: service health check failed")
            result[name] = {"reachable": True}
        return result

    async def job(self, service: str, key: str, route: str, id_field: str, payload: dict):
        return await self.services[service].run(
            key,
            route,
            id_field,
            payload,
            attach_id=self.attach_jobs.get(key),
            status_base="/v1/tasks" if service == "rosetta" else None,
        )

    def save_candidates(self):
        self.checkpoint.save()
        write_json(self.root / "candidates.json", self.checkpoint.data.get("candidates", []))

    async def generate(self):
        candidates = self.checkpoint.data.setdefault("candidates", [])
        sources = self.checkpoint.data.setdefault("generated_sources", [])
        errors = {}
        for source, generator in self.config.generators.items():
            if source in sources:
                continue
            try:
                result = await self.job(source, f"generate/{source}", "/v1/designs", "design_id", generator.request)
                manifest = select(result, generator.candidates_field)
                if manifest is None:
                    paths = [
                        p
                        for p in output_files(result, generator.service, self.files)
                        if p.suffix.lower() in {".pdb", ".cif"} and p.match(generator.structure_glob)
                    ]
                else:
                    if not isinstance(manifest, list):
                        raise ValueError("generator candidates_field must select a list")
                    paths = [self.files.local(item["structure_path"]) for item in manifest]
                if not paths:
                    raise ValueError("generator produced no candidate structures")
                added = []
                for path in sorted(set(paths)):
                    sequences = chain_sequences(path)
                    required = set(generator.chains.target + generator.chains.binder)
                    if set(sequences) != required:
                        raise ValueError(
                            f"{source}: structure chain IDs {sorted(sequences)} do not match {sorted(required)}"
                        )
                    remote = self.files.remote(path)
                    candidate_id = source + "-" + hashlib.sha256(remote.encode()).hexdigest()[:16]
                    added.append(
                        {
                            "candidate_id": candidate_id,
                            "source": source,
                            "sequence": "|".join(sequences[c] for c in generator.chains.binder),
                            "chains": sequences,
                            "generated_structure": remote,
                            "generated_sha256": digest(path),
                            "metrics": {},
                            "status": "generated",
                        }
                    )
                candidates.extend(added)
                sources.append(source)
                self.checkpoint.data.pop("generation_errors", None)
                self.save_candidates()
            except SubmissionUncertainError:
                raise
            except Exception as exc:
                errors[source] = str(exc)
        if errors:
            self.checkpoint.data["generation_errors"] = errors
            self.checkpoint.save()
            raise RuntimeError(f"both generators are required; generation failures: {errors}")
        return candidates

    async def score(self, candidate: dict):
        evaluation = self.config.evaluation
        key = candidate["candidate_id"]
        directory = self.root / key
        directory.mkdir(exist_ok=True)
        input_dir = directory / "af3score_input"
        model = input_dir / f"{key}.pdb"
        original = self.files.local(candidate["generated_structure"])
        if digest(original) != candidate["generated_sha256"]:
            raise ValueError(f"generated artifact changed: {key}")
        if (
            candidate.get("refold_structure")
            and digest(self.files.local(candidate["refold_structure"])) != candidate["refold_sha256"]
        ):
            raise ValueError(f"refold artifact changed: {key}")
        if "af3score_iptm" in candidate["metrics"]:
            return
        write_pdb(original, model)
        result = await self.job(
            "af3score",
            f"score/{key}",
            "/v1/tasks",
            "task_id",
            {
                **evaluation.af3score_options,
                "mode": "pdb",
                "input_pdb_dir": self.files.remote(input_dir),
                "num_samples": 1,
            },
        )
        service = self.services["af3score"]
        job = self.checkpoint.data["jobs"][f"score/{key}"]
        metrics = await service.get(f"/v1/tasks/{job['id']}/metrics")
        write_json(directory / "af3score-metrics.json", metrics)
        field = evaluation.af3score.metric_fields.get("iptm", "iptm")
        candidate["metrics"]["af3score_iptm"] = measured(
            metric_records(metrics, evaluation.af3score), field, confidence=True
        )
        candidate["af3score_results"] = result
        candidate["status"] = "scored"
        self.save_candidates()

    async def refold(self, candidate: dict):
        evaluation = self.config.evaluation
        key = candidate["candidate_id"]
        generator = self.config.generators[candidate["source"]]
        directory = self.root / key
        if "torchfold_iptm" not in candidate["metrics"]:
            input_json = directory / "fold_input.json"
            prepare_fold(generator.fold_template, candidate["chains"], generator.chains, key, input_json)
            result = await self.job(
                "torchfold",
                f"refold/{key}",
                "/v1/folds",
                "fold_id",
                {
                    **evaluation.torchfold_options,
                    "name": key,
                    "json_path": self.files.remote(input_json),
                },
            )
            path, iptm = refold_output(result, evaluation.torchfold, self.files, evaluation.fold_confidence_glob)
            sequences = chain_sequences(path)
            for original, folded in generator.chains.refold_map.items():
                if sequences.get(folded) != candidate["chains"][original]:
                    raise ValueError(f"refold chain {folded} does not match generated chain {original}")
            candidate.update(refold_structure=self.files.remote(path), refold_sha256=digest(path))
            candidate["metrics"]["torchfold_iptm"] = iptm
            self.save_candidates()
        if "design_refold_dockq" not in candidate["metrics"]:
            model = self.files.local(candidate["refold_structure"])
            if digest(model) != candidate["refold_sha256"]:
                raise ValueError("refold artifact changed")
            arguments = {
                "model": str(model),
                "native": str(self.files.local(candidate["generated_structure"])),
                "chain_map": generator.chains.refold_map,
                "targets": generator.chains.target,
                "binders": generator.chains.binder,
            }
            proc = await asyncio.create_subprocess_exec(
                str(evaluation.dockq_python),
                str(Path(__file__).with_name("dockq.py")),
                json.dumps(arguments),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), evaluation.dockq_timeout)
            except BaseException:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
                raise
            (directory / "dockq.log").write_bytes(stderr)
            if proc.returncode:
                raise RuntimeError(f"DockQ failed for {key}; see {directory / 'dockq.log'}")
            scores = json.loads(stdout)
            candidate["metrics"]["design_refold_dockq"] = measured([scores], "dockq", confidence=True)
            write_json(directory / "dockq.json", scores)
            self.save_candidates()
        if "rosetta_interface_dg" not in candidate["metrics"]:
            model = directory / "refold.pdb"
            write_pdb(self.files.local(candidate["refold_structure"]), model)
            chain_map = generator.chains.refold_map
            if any(len(c) != 1 for c in chain_map.values()):
                raise ValueError("Rosetta chain groups require single-character chain IDs")
            result = await self.job(
                "rosetta",
                f"interface/{key}",
                "/v1/tools/interface_analyze",
                "task_id",
                {
                    **evaluation.rosetta_options,
                    "input_pdb": self.files.remote(model),
                    "chain_group_a": "".join(chain_map[c] for c in generator.chains.binder),
                    "chain_group_b": "".join(chain_map[c] for c in generator.chains.target),
                },
            )
            field = evaluation.rosetta.metric_fields.get("interface_dg", "dG_separated")
            records = metric_records(result, evaluation.rosetta)
            if not any(select(row, field) is not None for row in records):
                scores = [
                    p
                    for p in output_files(result, evaluation.rosetta, self.files)
                    if p.match(evaluation.rosetta_scores_glob)
                ]
                records = [row for path in scores for row in table_records(path)]
            candidate["metrics"]["rosetta_interface_dg"] = measured(records, field)
        metrics = candidate["metrics"]
        candidate["status"] = (
            "accepted"
            if (
                metrics["torchfold_iptm"] >= evaluation.min_refold_iptm
                and metrics["design_refold_dockq"] >= evaluation.min_dockq
            )
            else "rejected"
        )
        self.save_candidates()

    def export(self, candidates: list[dict]) -> dict:
        for candidate in candidates:
            candidate.pop("rank", None)
        accepted = sorted(
            (c for c in candidates if c["status"] == "accepted"),
            key=lambda c: (
                -c["metrics"]["torchfold_iptm"],
                -c["metrics"]["design_refold_dockq"],
                c["metrics"]["rosetta_interface_dg"],
                c["candidate_id"],
            ),
        )
        for rank, candidate in enumerate(accepted, 1):
            candidate["rank"] = rank
        failed = [c["candidate_id"] for c in candidates if c["status"] == "failed"]
        result = {
            "status": "partial" if failed else "completed",
            "outcome": "incomplete" if failed else "hits" if accepted else "no_hits",
            "generated": len(candidates),
            "accepted": len(accepted),
            "failed": failed,
            "ranking": [c["candidate_id"] for c in accepted],
            "dockq_reference": "generated_design_pose",
            "ranking_policy": "refold_iptm_desc,dockq_desc,interface_dg_asc",
        }
        write_json(self.root / "summary.json", result)
        write_json(
            self.root / "initial_candidates.json",
            [
                {
                    "candidate_id": c["candidate_id"],
                    "sequence": c["sequence"],
                    "objective": c["metrics"]["torchfold_iptm"],
                    "metrics": {**c["metrics"], "iptm": c["metrics"]["torchfold_iptm"]},
                    "structure_path": c["refold_structure"],
                    "metadata": {
                        "generator": c["source"],
                        "chains": c["chains"],
                        "generated_structure": c["generated_structure"],
                        "dockq_reference": "generated_design_pose",
                    },
                }
                for c in accepted
            ],
        )
        with (self.root / "ranking.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "candidate_id",
                    "source",
                    "status",
                    "rank",
                    "af3score_iptm",
                    "torchfold_iptm",
                    "design_refold_dockq",
                    "rosetta_interface_dg",
                    "error",
                ],
            )
            writer.writeheader()
            for c in sorted(candidates, key=lambda c: (c.get("rank", float("inf")), c["candidate_id"])):
                writer.writerow({k: c.get(k, c["metrics"].get(k)) for k in writer.fieldnames})
        self.save_candidates()
        return result

    async def run(self, *, check_only=False):
        self.files.remote(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.root / "run.lock", blocking=False):
            self.validate_files()
            self.checkpoint = Checkpoint(
                self.root, {"config": self.config.model_dump(mode="json"), "inputs": self.input_hashes}
            )
            write_json(self.root / "run-config.json", self.config.model_dump(mode="json"))
            unknown = set(self.attach_jobs) - set(self.checkpoint.data["jobs"])
            if unknown:
                raise ValueError(f"--attach-job keys must exist in checkpoint: {sorted(unknown)}")
            try:
                health = await self.check()
                if check_only:
                    return {"status": "checked", "services": health, "inference_verified": False}
                candidates = await self.generate()
                for c in candidates:
                    try:
                        c.pop("error", None)
                        await self.score(c)
                    except SubmissionUncertainError:
                        raise
                    except Exception as exc:
                        c.update(status="failed", error=str(exc))
                        self.save_candidates()
                # Deduplicate within each source: preserve both generators' quotas.
                for source, generator in self.config.generators.items():
                    seen = set()
                    selected = 0
                    ordered = sorted(
                        (c for c in candidates if c["source"] == source and "error" not in c),
                        key=lambda c: (-c["metrics"]["af3score_iptm"], c["candidate_id"]),
                    )
                    for c in ordered:
                        if c["metrics"]["af3score_iptm"] < self.config.evaluation.min_af3score_iptm:
                            c["status"] = "filtered"
                        elif c["sequence"] in seen:
                            c["status"] = "duplicate"
                        elif selected >= generator.per_source_refold_limit:
                            c["status"] = "not_selected"
                        else:
                            selected += 1
                            seen.add(c["sequence"])
                            try:
                                await self.refold(c)
                            except SubmissionUncertainError:
                                raise
                            except Exception as exc:
                                c.update(status="failed", error=str(exc))
                        self.save_candidates()
                return self.export(candidates)
            finally:
                for service in self.services.values():
                    await service.close()
