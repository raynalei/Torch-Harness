#!/usr/bin/env python3
"""Private HTTP-to-RFD4 adapter for a single Slurm GPU allocation."""

import fcntl
import hmac
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

STRUCTURES = tuple(ext + compression for ext in (".cif", ".bcif") for compression in ("", ".gz", ".zst"))
SPECS = (".yaml", ".yml", ".json")
FIELD_NAMES = {"inputs", "num_replicates", "batch_size", "seed", "generation", "partial_trajectory_fraction_steps"}
TASK_ROUTE = re.compile(r"/v1/tasks/([0-9a-f]{32})(/results)?")
PATH_PATTERN = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
MAX_REQUEST_BYTES = 65536


def integer_option(data, key, default, minimum, maximum=None):
    value = data.get(key, default)
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        limit = f"{minimum}–{maximum}" if maximum is not None else f"≥{minimum}"
        raise ValueError(f"{key} must be an integer ({limit})")
    return value


def container_input(name, root, list_item=False):
    if not isinstance(name, str) or not PATH_PATTERN.fullmatch(name):
        raise ValueError("inputs must use relative paths with letters, digits, ., _, -, and /")
    if any(part in (".", "..") for part in name.split("/")):
        raise ValueError("inputs must not contain . or .. path components")
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("inputs must resolve below the configured input directory")
    if candidate.is_dir() and not list_item:
        return "/inputs/" + name
    allowed = STRUCTURES if list_item else STRUCTURES + SPECS
    if not candidate.is_file() or not name.lower().endswith(allowed):
        raise ValueError(f"input missing or unsupported: {name}")
    return "/inputs/" + name


def inference_options(data, root):
    if not isinstance(data, dict) or "inputs" not in data or data.keys() - FIELD_NAMES:
        raise ValueError("expected an object with inputs and supported options")
    inputs = data["inputs"]
    if isinstance(inputs, list):
        if not inputs:
            raise ValueError("inputs list must not be empty")
        inputs = "[" + ",".join(container_input(item, root, list_item=True) for item in inputs) + "]"
    else:
        inputs = container_input(inputs, root)
    generation = data.get("generation", "inference_base")
    if generation not in ("inference_base", "inverse_folding"):
        raise ValueError("generation must be inference_base or inverse_folding")
    replicates = integer_option(data, "num_replicates", 1, 1)
    batch = integer_option(data, "batch_size", 1, 1)
    options = [
        f"inputs={inputs}",
        f"generation={generation}",
        f"num_replicates={replicates}",
        f"batch_size={batch}",
        "~metrics",
        "compile_lengths=null",
    ]
    if data.get("seed") is not None:
        options.append(f"seed={integer_option(data, 'seed', 0, 0, 4294967295)}")
    fraction = data.get("partial_trajectory_fraction_steps")
    if fraction is not None:
        if type(fraction) not in (int, float) or not 0 < fraction <= 1:
            raise ValueError("partial_trajectory_fraction_steps must be in (0, 1]")
        if generation == "inverse_folding":
            raise ValueError("use partial_trajectory_fraction_steps with inference_base")
        options.append(f"generation.args.partial_trajectory_fraction_steps={float(fraction)}")
    return options


def read_state(path):
    return json.loads((path / "task.json").read_text())


def write_state(path, state):
    temporary = path / "task.json.tmp"
    temporary.write_text(json.dumps(state) + "\n")
    temporary.replace(path / "task.json")


class RFD4Service:
    def __init__(self):
        self.inputs = Path(os.environ["RFD4_INPUTS"]).resolve(strict=True)
        self.jobs = Path(os.environ["RFD4_JOBS"]).resolve(strict=True)
        self.weights = Path(os.environ["RFD4_WEIGHTS"]).resolve(strict=True)
        self.sif = Path(os.environ["RFD4_SIF"]).resolve(strict=True)
        self.apptainer = os.environ.get("RFD4_APPTAINER", "apptainer")
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.token = os.environ.get("RFD4_API_TOKEN")
        self.lock = (self.jobs / ".service.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for state_file in self.jobs.glob("*/task.json"):
            state = read_state(state_file.parent)
            if state["status"] in ("queued", "running"):
                state["status"] = "interrupted"
                write_state(state_file.parent, state)
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID")

    def infer(self, path, state, options):
        state = {**state, "status": "running"}
        write_state(path, state)
        command = [
            self.apptainer,
            "exec",
            "--nv",
            "--bind",
            f"{self.weights}:/weights:ro",
            "--bind",
            f"{self.inputs}:/inputs:ro",
            "--bind",
            f"{path}:/job:rw",
            "--pwd",
            "/job",
            str(self.sif),
            "rfd4-generate",
            "hf_ckpt_path=/weights",
            "out_dir=/job/output",
            *options,
        ]
        with (path / "inference.log").open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                state.update(status="succeeded" if result.returncode == 0 else "failed", exit_code=result.returncode)
            except OSError as error:
                print(error, file=log)
                state.update(status="failed", error=str(error))
        write_state(path, state)

    def submit(self, body):
        try:
            payload = json.loads(body)
            options = inference_options(payload, self.inputs)
        except (ValueError, OSError, RuntimeError) as error:
            return 400, {"error": str(error)}
        task_id = uuid4().hex
        path = self.jobs / task_id
        path.mkdir(mode=0o700)
        state = {
            "task_id": task_id,
            "status": "queued",
            "exit_code": None,
            "output_dir": str(path / "output"),
            "log_path": str(path / "inference.log"),
        }
        (path / "request.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")
        write_state(path, state)
        self.worker.submit(self.infer, path, state, options)
        return 202, state

    def get_task(self, task_id, results):
        path = self.jobs / task_id
        try:
            state = read_state(path)
        except FileNotFoundError:
            return 404, {"error": "task not found"}
        if results and state["status"] == "succeeded":
            output = path / "output"
            state["files"] = sorted(str(file.relative_to(output)) for file in output.rglob("*") if file.is_file())
        return 200, state


async def respond(send, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": data})


async def lifespan(receive, send):
    await receive()
    try:
        service = RFD4Service()
    except Exception as error:
        await send({"type": "lifespan.startup.failed", "message": str(error)})
        return
    app.service = service
    await send({"type": "lifespan.startup.complete"})
    await receive()
    service.worker.shutdown(wait=False, cancel_futures=True)
    await send({"type": "lifespan.shutdown.complete"})


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        return await lifespan(receive, send)
    if scope["type"] != "http":
        return

    service = app.service
    method, path = scope["method"], scope["path"]
    if method == "GET" and path == "/health":
        return await respond(send, 200, {"status": "ok", "service": "rfd4", "slurm_job_id": service.slurm_job_id})
    if service.token:
        authorization = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(authorization, f"Bearer {service.token}".encode()):
            return await respond(send, 401, {"error": "invalid or missing bearer token"})

    if method == "POST" and path == "/v1/tasks":
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message["body"])
            if len(body) > MAX_REQUEST_BYTES:
                return await respond(send, 413, {"error": "request body too large"})
            if not message.get("more_body"):
                break
        return await respond(send, *service.submit(body))

    match = TASK_ROUTE.fullmatch(path)
    if method == "GET" and match:
        return await respond(send, *service.get_task(match[1], bool(match[2])))
    return await respond(send, 404, {"error": "unknown route"})
