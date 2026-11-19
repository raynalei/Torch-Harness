"""Offline protocol and worker tests using a test-only PyRosetta stand-in."""

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

DEPLOYMENT = Path(__file__).resolve().parents[1] / "deploy" / "rosetta"

spec = importlib.util.spec_from_file_location("rosetta_deployment", (DEPLOYMENT / "main.py"))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


@pytest.fixture(autouse=True)
def scheduler_paths(monkeypatch):
    monkeypatch.setattr(api, "scheduler_command", lambda name: "/test-slurm/" + name)


def complex_pdb(path):
    lines = []
    for i, (chain, name, y) in enumerate([("A", "GLY", 0), ("B", "ALA", 4)], 1):
        lines.append(f"ATOM  {i:5d}  CA  {name} {chain}{1:4d}    {0:8.3f}{y:8.3f}{0:8.3f}  1.00 80.00           C")
        lines.append("TER")
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


FAKE_PYROSETTA = "import os\nimport sys\nfrom collections import defaultdict\nfrom pathlib import Path\nfrom types import ModuleType\n\nclass EMapVector(defaultdict):\n    def __init__(self):\n        super().__init__(float)\nclass LongRangeTwoBodyEnergy:\n    def defines_residue_pair_energy(self, pose, i, j):\n        assert (i,j)==(1,2)\n        return os.environ.get('FAKE_ENERGY_ZERO') != '1'\n    def residue_pair_energy(self, a, b, pose, scorefxn, emap):\n        emap['long'] += -0.5\nclass Edge:\n    def fill_energy_map(self):\n        result = EMapVector()\n        result['short'] = -3.0\n        return result\nclass Graph:\n    def find_energy_edge(self, i, j):\n        assert (i,j) == (1,2)\n        if os.environ.get('FAKE_PAIR_FAILURE') == '1':\n            raise SystemExit(9)\n        return None if os.environ.get('FAKE_ENERGY_ZERO') == '1' else Edge()\nclass Residue:\n    def __init__(self, i): self.i=i\n    def name3(self): return 'GLY' if self.i==1 else 'ALA'\nclass Pose:\n    def total_residue(self): return 2\n    def pdb_info(self): return self\n    def chain(self, i): return 'A' if i==1 else 'B'\n    def number(self, i): return 1\n    def icode(self, i): return ' '\n    def residue(self, i): return Residue(i)\n    def energies(self): return self\n    def energy_graph(self): return Graph()\n    def clone(self): return Pose()\nclass Options:\n    def hbond_options(self): return self\n    def decompose_bb_hb_into_pair_energies(self, enabled): assert enabled\nclass ScoreFunction:\n    def __call__(self, pose): return -30.0\n    def energy_method_options(self): return Options()\n    def set_energy_method_options(self, options): pass\n    def weights(self): return {'short':2.0,'long':0.5}\n    def get_nonzero_weighted_scoretypes(self): return ['short','long']\n    def all_methods(self): return [LongRangeTwoBodyEnergy()]\nclass InterfaceAnalyzerMover:\n    def set_interface(self, interface): assert interface=='B_A'\n    def set_scorefunction(self, sf): pass\n    def set_pack_input(self, value): pass\n    def set_pack_separated(self, value): pass\n    def set_compute_packstat(self, value): assert value is False\n    def apply(self, pose): pass\n    def get_interface_dG(self): return -12.0\n\ndef init(options):\n    if os.environ.get('FAKE_INIT_FAILURE') == '1':\n        raise RuntimeError('fake initialization failed')\ndef create_score_function(weights):\n    assert weights == 'ref2015'\n    return ScoreFunction()\ndef pose_from_pdb(path):\n    assert 'ATOM' in Path(path).read_text()\n    return Pose()\n\nrosetta = ModuleType('pyrosetta.rosetta')\nrosetta.core = ModuleType('pyrosetta.rosetta.core')\nrosetta.core.scoring = ModuleType('pyrosetta.rosetta.core.scoring')\nrosetta.core.scoring.EMapVector = EMapVector\nrosetta.core.scoring.methods = ModuleType('pyrosetta.rosetta.core.scoring.methods')\nrosetta.core.scoring.methods.LongRangeTwoBodyEnergy = LongRangeTwoBodyEnergy\nrosetta.protocols = ModuleType('pyrosetta.rosetta.protocols')\nrosetta.protocols.analysis = ModuleType('pyrosetta.rosetta.protocols.analysis')\nrosetta.protocols.analysis.InterfaceAnalyzerMover = InterfaceAnalyzerMover\nfor module in [rosetta,rosetta.core,rosetta.core.scoring,rosetta.core.scoring.methods,rosetta.protocols,rosetta.protocols.analysis]:\n    sys.modules[module.__name__]=module\n"

HEADER = "SCORE: total pdbid1 pdbid2 pose_id resi1 resi2 restype1 restype2 description\n"
PAIR = "SCORE: -6.25 1A 1B input 1 2 GLY ALA input_1_2\n"
ONEBODY = "SCORE: -1 1A -- input 1 -- GLY onebody input_1_onebody\n"


def test_normalization_preserves_scientific_values_and_binder_orientation(tmp_path):
    src = complex_pdb(tmp_path / "input.pdb")
    raw, out = tmp_path / "raw.sc", tmp_path / "residue_energies.out"
    raw.write_text(HEADER + ONEBODY + PAIR)
    assert api.normalize_residue_energies(raw, out, src, "B", "A") == 1
    assert out.read_text().splitlines() == [
        "ResResE Res1 Res2 total",
        "ResResE ---- ---- -----",
        "ResResE ALA_661 GLY_651 -6.25",
    ]


@pytest.mark.parametrize(
    "body",
    [
        "",
        HEADER + ONEBODY,
        HEADER + PAIR.replace("-6.25", "nan"),
        HEADER + PAIR.replace("1B", "99B"),
        HEADER + PAIR + PAIR.replace("-6.25", "-7"),
        HEADER + PAIR.replace("1B", "1B:A"),
        HEADER + PAIR + "SCORE: broken\n",
    ],
)
def test_normalization_fails_instead_of_inventing_or_losing_scores(tmp_path, body):
    src = complex_pdb(tmp_path / "input.pdb")
    raw, out = tmp_path / "raw.sc", tmp_path / "normalized.out"
    raw.write_text(body)
    with pytest.raises(ValueError):
        api.normalize_residue_energies(raw, out, src, "B", "A")
    assert not out.exists()


@pytest.mark.parametrize(
    "accounting,expected,exit_code",
    [
        ("42|COMPLETED|0:0\n", "COMPLETED", "0:0"),
        ("42|COMPLETED|1:0\n", "FAILED", "1:0"),
        ("42|COMPLETED|0:9\n", "FAILED", "0:9"),
        ("42|FAILED|1:0\n", "FAILED", "1:0"),
        ("42|CANCELLED by 123|0:15\n", "CANCELLED", "0:15"),
        ("42|TIMEOUT|0:0\n", "TIMEOUT", "0:0"),
        ("42.batch|COMPLETED|0:0\n", "UNKNOWN", None),
        ("42|COMPLETED|\n", "UNKNOWN", None),
        ("", "UNKNOWN", None),
    ],
)
def test_only_parent_accounting_and_zero_exit_prove_completion(monkeypatch, accounting, expected, exit_code):
    def run(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=accounting if Path(args[0]).name == "sacct" else "", stderr="")

    monkeypatch.setattr(api.subprocess, "run", run)
    assert api.job_status("42") == {
        "status": expected,
        "state": expected,
        "exit_code": exit_code,
    }


def test_running_state_does_not_use_stale_accounting(monkeypatch):
    def run(args, **kwargs):
        assert Path(args[0]).name == "squeue"
        return SimpleNamespace(returncode=0, stdout="RUNNING\n", stderr="")

    monkeypatch.setattr(api.subprocess, "run", run)
    assert api.job_status("42")["exit_code"] is None


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "WORK", tmp_path / "jobs")
    monkeypatch.setattr(api, "ALLOWED", (tmp_path,))
    support = tmp_path / "fake-python-packages"
    package = support / "pyrosetta"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(FAKE_PYROSETTA)
    monkeypatch.syspath_prepend(str(support))
    monkeypatch.setenv("PYTHONPATH", str(support) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    original_run = subprocess.run
    scripts = []

    def run(args, **kwargs):
        if Path(args[0]).name == "sbatch":
            assert "--export=ALL" in args
            scripts.append(Path(args[-1]))
            return SimpleNamespace(returncode=0, stdout="42;cluster\n", stderr="")
        if Path(args[0]).name == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if Path(args[0]).name == "sacct":
            return SimpleNamespace(returncode=0, stdout="42|COMPLETED|0:0\n", stderr="")
        return original_run(args, **kwargs)

    monkeypatch.setattr(api.subprocess, "run", run)
    api.deployment_checks.cache_clear()
    yield TestClient(api.app), scripts, original_run
    api.deployment_checks.cache_clear()


def test_submission_runs_pyrosetta_worker_and_returns_github_compatible_results(deployment, tmp_path):
    client, scripts, run = deployment
    src = complex_pdb(tmp_path / "complex with spaces.pdb")
    response = client.post(
        "/v1/tools/interface_analyze",
        json={
            "input_pdb": str(src),
            "chain_group_a": "B",
            "chain_group_b": "A",
            "pack_input": False,
            "pack_separated": True,
        },
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert len(scripts) == 1
    snapshot = scripts[0].parent / "input.pdb"
    assert snapshot.read_bytes() == src.read_bytes()
    # Mutating the submitted source must not change the analysis inputs.
    src.write_text("changed original source")
    executed = run(["bash", str(scripts[0])], capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    status = client.get(f"/v1/tasks/{task_id}").json()
    assert status["status"] == status["state"] == "COMPLETED" and status["exit_code"] == "0:0"
    result = client.get(f"/v1/tasks/{task_id}/results")
    assert result.status_code == 200
    result = result.json()
    assert len([p for p in result["files"] if p.endswith(".out")]) == 1
    log = Path(result["residue_energy_files"][0])
    assert "ALA_661 GLY_651 -6.25" in log.read_text()
    assert result["metrics"][0]["dG_separated"] == -12
    assert "residue_pairs.sc" not in result["score_rows"]
    assert set(result["residue_energy_files"]) <= set(result["files"])


def test_pair_command_failure_makes_the_child_job_fail(deployment, tmp_path, monkeypatch):
    client, scripts, run = deployment
    src = complex_pdb(tmp_path / "input.pdb")
    submitted = client.post(
        "/v1/tools/interface_analyze",
        json={"input_pdb": str(src), "chain_group_a": "B", "chain_group_b": "A"},
    )
    assert submitted.status_code == 202
    monkeypatch.setenv("FAKE_PAIR_FAILURE", "1")
    assert run(["bash", str(scripts[0])], capture_output=True).returncode == 9
    assert not (scripts[0].parent / "outputs/residue_energies.out").exists()


def test_results_are_unavailable_without_verified_exit_code(deployment, tmp_path, monkeypatch):
    client, _, _ = deployment
    src = complex_pdb(tmp_path / "input.pdb")
    task_id = client.post(
        "/v1/tools/interface_analyze",
        json={"input_pdb": str(src), "chain_group_a": "B", "chain_group_b": "A"},
    ).json()["task_id"]
    monkeypatch.setattr(
        api,
        "job_status",
        lambda job: {"status": "COMPLETED", "state": "COMPLETED", "exit_code": None},
    )
    assert client.get(f"/v1/tasks/{task_id}/results").status_code == 409


def test_bad_chain_groups_do_not_submit_a_job(deployment, tmp_path):
    client, scripts, _ = deployment
    src = complex_pdb(tmp_path / "input.pdb")
    response = client.post(
        "/v1/tools/interface_analyze",
        json={"input_pdb": str(src), "chain_group_a": "BB", "chain_group_b": "A"},
    )
    assert response.status_code == 422
    assert scripts == []


@pytest.mark.asyncio
async def test_github_runner_consumes_the_api_energy_artifact(deployment, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.config import (
        Chains,
        PathMapping,
        Service,
    )
    from opendde_harness.plugin.protein_design.initial_round.runner import (
        InitialRoundRunner,
    )
    from opendde_harness.plugin.protein_design.initial_round.services import (
        SlurmService,
    )
    from opendde_harness.plugin.protein_design.initial_round.storage import SharedFiles

    client, scripts, run = deployment
    model = complex_pdb(tmp_path / "generated.pdb")
    service = Service(url="https://rosetta.test")
    chains = Chains(target=["A"], binder=["B"], refold_map={"A": "A", "B": "B"})

    async def job(kind, key, route, id_field, payload):
        assert kind == "rosetta" and route == "/v1/tools/interface_analyze"
        submitted = client.post(route, json=payload)
        assert submitted.status_code == 202
        task_id = submitted.json()[id_field]
        executed = run(["bash", str(scripts[-1])], capture_output=True, text=True)
        assert executed.returncode == 0, executed.stderr
        status = client.get(f"/v1/tasks/{task_id}").json()
        adapter = object.__new__(SlurmService)
        adapter.config = service
        assert adapter.completion_evidence(status)
        result = client.get(f"/v1/tasks/{task_id}/results")
        assert result.status_code == 200
        return result.json()

    root = tmp_path / "client-run"
    (root / "candidate").mkdir(parents=True)
    runner = SimpleNamespace(
        root=root,
        job=job,
        save_candidates=lambda: None,
        files=SharedFiles([PathMapping(local=tmp_path, remote=str(tmp_path))]),
        config=SimpleNamespace(
            autonomous=SimpleNamespace(interface_distance=12.0, interface_energy_threshold=-5.0),
            generators={"ppiflow": SimpleNamespace(chains=chains)},
            evaluation=SimpleNamespace(
                rosetta=service,
                score_metric="torchscore_iptm",
                rosetta_options={"pack_input": False, "pack_separated": True},
                rosetta_residue_glob="*.out",
            ),
        ),
    )
    candidate = {
        "candidate_id": "candidate",
        "source": "ppiflow",
        "generated_structure": str(model),
        "metrics": {"torchscore_iptm": 0.7},
    }
    await InitialRoundRunner.score_interface(runner, candidate)
    assert candidate["interface_residue_energies"] == {"B": {"1": -6.25}}
    assert candidate["interface_fixed_positions"] == {"B": [1]}
    assert candidate["metrics"]["rosetta_interface_score"] == -6.25
    assert candidate["metrics"]["rosetta_interface_dg"] == -12.0
    assert candidate["metrics"]["ranking_score"] == 82.0
    assert Path(candidate["rosetta_energy_log"]).is_file()


def test_health_initializes_native_pyrosetta_in_the_same_python(deployment):
    result = api.deployment_checks()
    assert result["ok"] and result["backend"] == "pyrosetta"
    assert result["supported_tools"] == ["interface_analyze", "structure_score"]


def test_pyrosetta_initialization_failure_is_not_healthy(deployment, monkeypatch):
    monkeypatch.setenv("FAKE_INIT_FAILURE", "1")
    result = api.deployment_checks()
    assert not result["ok"] and "initialization failed" in result["error"]


def test_zero_pair_energies_are_preserved_without_false_fixed_points(deployment, tmp_path, monkeypatch):
    client, scripts, run = deployment
    src = complex_pdb(tmp_path / "input.pdb")
    response = client.post(
        "/v1/tools/interface_analyze",
        json={"input_pdb": str(src), "chain_group_a": "B", "chain_group_b": "A"},
    )
    monkeypatch.setenv("FAKE_ENERGY_ZERO", "1")
    executed = run(["bash", str(scripts[0])], capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    result = client.get(f"/v1/tasks/{response.json()['task_id']}/results").json()
    assert "ALA_661 GLY_651 0\n" in Path(result["residue_energy_files"][0]).read_text()


def test_structure_score_uses_pyrosetta_and_keeps_scalar_metrics(deployment, tmp_path):
    client, scripts, run = deployment
    response = client.post(
        "/v1/tools/structure_score",
        json={"input_pdb": str(complex_pdb(tmp_path / "input.pdb"))},
    )
    assert response.status_code == 202
    assert run(["bash", str(scripts[0])], capture_output=True).returncode == 0
    result = client.get(f"/v1/tasks/{response.json()['task_id']}/results").json()
    assert result["metrics"] == [{"total_score": -30.0, "description": "input"}]
    assert result["residue_energy_files"] == []


def test_unimplemented_advanced_protocols_are_explicit(deployment):
    client, scripts, _ = deployment
    assert client.post("/v1/tools/mutation_binding_ddg", json={}).status_code == 501
    assert scripts == []


@pytest.mark.parametrize("activate_by_name", [False, True])
def test_sbatch_uses_the_selected_conda_python(deployment, tmp_path, monkeypatch, activate_by_name):
    import shlex
    import shutil
    import sys

    _, _, run = deployment
    service_dir = tmp_path / "api"
    service_dir.mkdir()
    shutil.copy2((DEPLOYMENT / "main.py"), service_dir / "main.py")
    prefix = tmp_path / "conda-env"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin/python"
    python.write_text("#!/bin/bash\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o700)
    # The fixture package contains only fake scientific interfaces; uvicorn
    # dependency discovery still uses the real API interpreter.
    srun = prefix / "bin/srun"
    srun.write_text('#!/bin/bash\nprintf "%s\\n" "$ROSETTA_API_PYTHON" "$CONDA_PREFIX" "$@" > "$TEST_SRUN_RECORD"\n')
    srun.chmod(0o700)
    record = tmp_path / "srun-record"
    monkeypatch.setenv("TEST_SRUN_RECORD", str(record))
    monkeypatch.setenv("ROSETTA_API_WORK", str(tmp_path / "work"))
    monkeypatch.delenv("ROSETTA_CONDA_ENV", raising=False)
    if activate_by_name:
        hook = tmp_path / "conda.sh"
        hook.write_text('conda() { [[ "$1" == activate ]] || return 1; export CONDA_PREFIX="$2"; }\n')
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("ROSETTA_CONDA_ENV", str(prefix))
        monkeypatch.setenv("ROSETTA_CONDA_SH", str(hook))
    else:
        monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    script = tmp_path / "api.sbatch"
    script.write_text((DEPLOYMENT / "rosetta-api.sbatch").read_text())
    monkeypatch.setenv("ROSETTA_API_DIR", str(service_dir))
    monkeypatch.setenv("ROSETTA_ALLOWED_INPUT_ROOTS", str(tmp_path))
    finished = run(["bash", str(script)], capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    values = record.read_text().splitlines()
    assert values[:2] == [str(python), str(prefix)]
    assert "--export=ALL" in values and "main:app" in values
