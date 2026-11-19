"""Native artifact contracts; API data is never interpreted as instructions."""

import re
from pathlib import PurePosixPath


def absolute_artifact(value):
    if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise ValueError("model output must contain an absolute cluster artifact path")
    return value


def normalize_model_results(kind, results):
    if not isinstance(results, dict):
        raise ValueError("model results must be an object")
    if kind == "rfd4":
        root = PurePosixPath(absolute_artifact(results.get("output_dir")))
        relative = results.get("files")
        if not isinstance(relative, list) or not relative:
            raise ValueError("RFD4 completed without an output artifact manifest")
        files = []
        for name in relative:
            if (
                not isinstance(name, str)
                or not name
                or not PurePosixPath(name).parts
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
            ):
                raise ValueError("RFD4 output files must be relative to output_dir without traversal")
            files.append(str(root / name))
        return {**results, "relative_files": relative, "files": files}
    if kind == "proteinmpnn":
        fasta = absolute_artifact(results.get("fasta_path"))
        files = [fasta]
        files.extend(absolute_artifact(results[key]) for key in ("stdout_path", "stderr_path") if results.get(key))
        return {**results, "files": files}
    return results


def parse_fasta(text):
    records = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if records and not records[-1]["sequence"]:
                raise ValueError("ProteinMPNN returned an empty FASTA record")
            records.append({"header": line[1:], "sequence": ""})
        elif not records or not re.fullmatch(r"[A-Z/]+", line):
            raise ValueError("ProteinMPNN returned invalid FASTA sequence data")
        else:
            records[-1]["sequence"] += line
    if not records or not records[-1]["sequence"]:
        raise ValueError("ProteinMPNN returned no FASTA sequences")
    return records
