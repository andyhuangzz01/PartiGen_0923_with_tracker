"""Run the requested generated/baseline 500-motion comparison sequentially."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PYTHON = str(Path(sys.executable).absolute())
SCREEN = ROOT / "tools/screen_generated_motions.py"
COMPARE = ROOT / "tools/compare_mpjpe_runs.py"

GENERATED_INPUT = REPO / "TextOpRobotMDAR/gen_test2576_dar0911_512"
BASELINE_INPUT = ROOT.parent / "gen_all2576_dar0908_512.pkl"
GENERATED_OUT = ROOT.parent / "gen_test2576_dar0911_mpjpe_0001_0500"
BASELINE_OUT = ROOT.parent / "baseline_dar0908_mpjpe_0000_0499"
COMPARE_OUT = ROOT.parent / "mpjpe_compare_500"


def run_screen(input_path: Path, output_path: Path, indices: range) -> None:
    command = [
        PYTHON,
        str(SCREEN),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--indices",
        ",".join(str(index) for index in indices),
        "--parallel",
        "1",
    ]
    if (output_path / "run.json").exists():
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    print("[comparison] generated sample IDs 1..500", flush=True)
    run_screen(GENERATED_INPUT, GENERATED_OUT, range(1, 501))
    print("[comparison] baseline indices 0..499", flush=True)
    run_screen(BASELINE_INPUT, BASELINE_OUT, range(500))
    subprocess.run(
        [
            PYTHON,
            str(COMPARE),
            "--generated",
            str(GENERATED_OUT / "summary.json"),
            "--baseline",
            str(BASELINE_OUT / "summary.json"),
            "--output",
            str(COMPARE_OUT),
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"[comparison] report: {COMPARE_OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
