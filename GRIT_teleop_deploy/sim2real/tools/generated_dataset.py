"""Audit the unpacked PartiGen/RobotMDAR sample-directory contract."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

REQUIRED_FILES = ("qpos.npy", "contact.npy", "prompt.txt", "motion.mp4")
SAMPLE_RE = re.compile(r"sample_(\d{4})$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_prompt(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def audit_directory(root: Path) -> tuple[dict, list[dict]]:
    root = root.resolve()
    dirs = {}
    unexpected_dirs = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        match = SAMPLE_RE.fullmatch(path.name)
        if match:
            sample_id = int(match.group(1))
            if sample_id in dirs:
                raise ValueError(f"duplicate sample ID {sample_id}")
            dirs[sample_id] = path
        else:
            unexpected_dirs.append(str(path))

    rows = []
    errors = []
    qpos_hashes = Counter()
    src_ids = Counter()
    for sample_id, path in sorted(dirs.items()):
        missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            errors.append(f"sample_{sample_id:04d}: missing {missing}")
            continue
        metadata = parse_prompt(path / "prompt.txt")
        try:
            qpos = np.load(path / "qpos.npy", allow_pickle=False)
            contact = np.load(path / "contact.npy", allow_pickle=False)
            fps = float(metadata["fps"])
            src_idx = int(metadata["src_idx"])
            metadata_sample_idx = int(metadata["sample_idx"])
            gen_frames = int(metadata["gen_frames"])
            src_length = int(metadata["src_length"])
            num_primitive = int(metadata["num_primitive"])
            if qpos.ndim != 2 or qpos.shape[1] != 30 or qpos.shape[0] < 2:
                raise ValueError(f"qpos shape is {qpos.shape}, expected [T,30]")
            if contact.shape != (qpos.shape[0], 2):
                raise ValueError(f"contact shape is {contact.shape}, expected {(qpos.shape[0], 2)}")
            if not np.isfinite(qpos).all() or not np.isfinite(contact).all():
                raise ValueError("qpos/contact contains NaN or Inf")
            if metadata_sample_idx != sample_id:
                raise ValueError(f"metadata sample_idx={metadata_sample_idx} does not match directory")
            if gen_frames != qpos.shape[0]:
                raise ValueError(f"metadata gen_frames={gen_frames} != qpos frames={qpos.shape[0]}")
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"invalid fps={fps}")
            norms = np.linalg.norm(qpos[:, 3:7], axis=1)
            if np.any(norms < 1e-6) or np.max(np.abs(norms - 1.0)) > 0.05:
                raise ValueError("invalid/non-unit xyzw root quaternion")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"sample_{sample_id:04d}: {exc}")
            continue

        paths = {name: path / name for name in REQUIRED_FILES}
        hashes = {name: digest(file_path) for name, file_path in paths.items()}
        sizes = {name: file_path.stat().st_size for name, file_path in paths.items()}
        qpos_hashes[hashes["qpos.npy"]] += 1
        src_ids[src_idx] += 1
        rows.append({
            "sample_id": sample_id,
            "sample_name": path.name,
            "src_idx": src_idx,
            "prompt": metadata.get("prompt", ""),
            "src_length": src_length,
            "num_primitive": num_primitive,
            "frames": int(qpos.shape[0]),
            "fps": fps,
            "duration_s": (qpos.shape[0] - 1) / fps,
            "qpos_shape": list(qpos.shape),
            "contact_shape": list(contact.shape),
            "quaternion_norm_min": float(norms.min()),
            "quaternion_norm_max": float(norms.max()),
            "paths": {name: str(file_path.resolve()) for name, file_path in paths.items()},
            "sizes_bytes": sizes,
            "sha256": hashes,
        })

    sample_ids = sorted(dirs)
    expected_ids = list(range(len(dirs)))
    duplicate_qpos = {key: count for key, count in qpos_hashes.items() if count > 1}
    duplicate_src = {str(key): count for key, count in src_ids.items() if count > 1}
    summary = {
        "root": str(root),
        "sample_directories": len(dirs),
        "valid_samples": len(rows),
        "invalid_samples": len(errors),
        "contiguous_sample_ids_from_zero": sample_ids == expected_ids,
        "missing_sample_ids_0_2575": sorted(set(range(2576)).difference(sample_ids)),
        "unexpected_directories": unexpected_dirs,
        "duplicate_qpos_content_groups": len(duplicate_qpos),
        "duplicate_qpos_content_counts": duplicate_qpos,
        "duplicate_src_idx_counts": duplicate_src,
        "fps_values": sorted({row["fps"] for row in rows}),
        "frame_min": min((row["frames"] for row in rows), default=None),
        "frame_max": max((row["frames"] for row in rows), default=None),
        "total_frames": sum(row["frames"] for row in rows),
        "errors": errors,
    }
    manifest = {
        "format": {
            "qpos_layout": "root position xyz [m] + root quaternion xyzw + 23 joint angles [rad]",
            "qpos_shape": "[T,30]",
            "contact_shape": "[T,2]",
            "fps_source": "prompt.txt metadata, cross-checked against generation/export code",
            "mp4_usage": "human review only; never used for MPJPE",
        },
        "summary": summary,
        "samples": rows,
    }
    return manifest, rows


def records_from_manifest(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        records.append({
            "qpos": np.load(row["paths"]["qpos.npy"], allow_pickle=False),
            "fps": row["fps"],
            "src_idx": row["src_idx"],
            "sample_idx": row["sample_id"],
            "text": row["prompt"],
            "source_paths": row["paths"],
            "source_sha256": row["sha256"],
        })
    return records


def write_manifest(output: Path, manifest: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "sample_id", "src_idx", "prompt", "frames", "fps", "duration_s",
        "qpos_path", "contact_path", "prompt_path", "mp4_path",
        "qpos_sha256", "contact_sha256", "prompt_sha256", "mp4_sha256",
    ]
    with (output / "input_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in manifest["samples"]:
            writer.writerow({
                "sample_id": row["sample_id"], "src_idx": row["src_idx"],
                "prompt": row["prompt"], "frames": row["frames"], "fps": row["fps"],
                "duration_s": row["duration_s"],
                **{f"{key}_path": row["paths"][filename] for key, filename in
                   (("qpos", "qpos.npy"), ("contact", "contact.npy"), ("prompt", "prompt.txt"), ("mp4", "motion.mp4"))},
                **{f"{key}_sha256": row["sha256"][filename] for key, filename in
                   (("qpos", "qpos.npy"), ("contact", "contact.npy"), ("prompt", "prompt.txt"), ("mp4", "motion.mp4"))},
            })
