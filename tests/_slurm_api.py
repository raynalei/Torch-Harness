"""Scheduler fixture for protocol tests; production code has no direct-HTTP mode."""

from urllib.parse import urlsplit

import pytest

from opendde_harness.plugin.protein_design.initial_round import api_startup
from opendde_harness.plugin.protein_design.initial_round.services import ServiceConnection


@pytest.fixture
def mock_slurm_apis(tmp_path, monkeypatch):
    script = tmp_path / "mock-api.sbatch"
    script.write_text("#!/bin/bash\n# Protocol-test scheduler fixture\n")
    initialize = ServiceConnection.__init__
    original_command = api_startup.slurm_command
    original_health = api_startup.healthy
    jobs = {}

    def initialize_with_scheduler(self, config, **kwargs):
        if config.sbatch_script is None:
            config = config.model_copy(update={"sbatch_script": script, "sbatch_resolve_host": True})
        initialize(self, config, **kwargs)

    async def command(config, *args):
        if config.sbatch_script != script:
            return await original_command(config, *args)
        if args[0] == "sbatch":
            identifier = str(1000 + len(jobs))
            jobs[identifier] = {"name": args[2].split("=", 1)[1], "host": urlsplit(config.url).hostname}
            return identifier
        if args[0] == "squeue":
            job = jobs[args[2]]
            return "RUNNING|" + (job["name"] if args[-1] == "--format=%T|%j" else job["host"])
        if args[0] == "scontrol":
            return args[-1]
        if args[0] == "scancel":
            return ""
        raise AssertionError(f"unexpected scheduler command: {args}")

    async def health(client, path):
        if client.base_url.host.endswith(".test"):
            # These tests exercise model protocols; startup tests cover HTTP readiness.
            api_startup.remote_api_url(client.base_url)
            return True
        return await original_health(client, path)

    monkeypatch.setattr(ServiceConnection, "__init__", initialize_with_scheduler)
    monkeypatch.setattr(api_startup, "slurm_command", command)
    monkeypatch.setattr(api_startup, "healthy", health)
    return jobs
