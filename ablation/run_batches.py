from __future__ import annotations

import argparse
import importlib.util
import json
import secrets
from datetime import datetime
from pathlib import Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_run_ablation_module():
    mod_path = Path(__file__).resolve().parent / "run_ablation.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"run_ablation.py not found at {mod_path}")
    spec = importlib.util.spec_from_file_location("ablation_run_ablation", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_ablation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_one_batch(*, output_root: Path, seed: int, workers: int) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    run_ablation = _load_run_ablation_module()
    run_all = getattr(run_ablation, "run_all", None)
    if run_all is None:
        raise RuntimeError("run_all(...) not found in run_ablation.py")
    run_all(base_output_dir=output_root, seed=int(seed), workers=int(workers))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base-dir", type=str, default="./ablation/outputs_batches")
    args = parser.parse_args()

    runs = int(args.runs)
    if runs <= 0:
        raise ValueError("--runs must be positive")

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for i in range(1, runs + 1):
        seed = int(secrets.randbelow(2**31 - 1))
        out_dir = base_dir / f"outputs_{i:02d}_{_timestamp()}"
        print(f"[batch] start run={i}/{runs} seed={seed} output_root={out_dir}", flush=True)
        _run_one_batch(output_root=out_dir, seed=seed, workers=int(args.workers))
        summary.append({"run": i, "seed": seed, "output_root": str(out_dir)})
        print(f"[batch] done run={i}/{runs} seed={seed} output_root={out_dir}", flush=True)

    index_path = base_dir / f"batches_{_timestamp()}.json"
    index_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch] batches_index_written={index_path}", flush=True)


if __name__ == "__main__":
    main()

