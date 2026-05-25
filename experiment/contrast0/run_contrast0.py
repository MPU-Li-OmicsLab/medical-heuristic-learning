from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
script_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hl.config import LLMConfig, RunConfig
from hl.metrics import compute_metrics
from hl.orchestrator import run_heuristic_learning
from hl.utils.io import write_json, write_text


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    label_col: str


@dataclass(frozen=True)
class SplitSpec:
    val_total: int = 1000
    test_total: int = 1000
    pos_value: int = 1
    neg_value: int = 0


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    base_url: str
    api_key_env: str
    model_name: str
    temperature: float
    extra_body: dict | None = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _split_val_test_balanced(
    df: pd.DataFrame,
    label_col: str,
    spec: SplitSpec,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[label_col].astype(int)
    pos_df = df.loc[y == spec.pos_value].copy()
    neg_df = df.loc[y == spec.neg_value].copy()

    n_val_each = spec.val_total // 2
    n_test_each = spec.test_total // 2
    need_each = n_val_each + n_test_each
    if len(pos_df) < need_each or len(neg_df) < need_each:
        raise ValueError(
            f"Not enough samples for balanced splits. Need pos>= {need_each}, neg>= {need_each}, got pos={len(pos_df)}, neg={len(neg_df)}"
        )

    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(pos_df.index.to_numpy(dtype=int))
    neg_idx = rng.permutation(neg_df.index.to_numpy(dtype=int))

    def take(idxs: np.ndarray, start: int, n: int) -> np.ndarray:
        return idxs[start : start + n]

    pos_test = take(pos_idx, 0, n_test_each)
    neg_test = take(neg_idx, 0, n_test_each)
    pos_val = take(pos_idx, n_test_each, n_val_each)
    neg_val = take(neg_idx, n_test_each, n_val_each)

    test_df = pd.concat([pos_df.loc[pos_test], neg_df.loc[neg_test]], axis=0).sample(frac=1.0, random_state=seed)
    val_df = pd.concat([pos_df.loc[pos_val], neg_df.loc[neg_val]], axis=0).sample(frac=1.0, random_state=seed + 1)

    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    used_idx = set(pos_test.tolist()) | set(neg_test.tolist()) | set(pos_val.tolist()) | set(neg_val.tolist())
    train_pool = df.loc[~df.index.isin(list(used_idx))].copy().reset_index(drop=True)
    return train_pool, val_df, test_df


def _sample_train_balanced(
    train_pool: pd.DataFrame,
    *,
    label_col: str,
    train_total: int,
    seed: int,
    spec: SplitSpec,
) -> tuple[pd.DataFrame, dict]:
    if train_total <= 0:
        raise ValueError("train_total must be positive")
    if train_total % 2 != 0:
        raise ValueError("train_total must be even for 1:1 balanced sampling")

    pos_each = train_total // 2
    neg_each = train_total // 2

    y = train_pool[label_col].astype(int)
    pos_pool = train_pool.loc[y == spec.pos_value].copy()
    neg_pool = train_pool.loc[y == spec.neg_value].copy()
    if len(pos_pool) == 0 or len(neg_pool) == 0:
        raise ValueError(f"train_pool has no samples for one class. pos={len(pos_pool)}, neg={len(neg_pool)}")

    pos_replace = len(pos_pool) < pos_each
    neg_replace = len(neg_pool) < neg_each

    pos_df = pos_pool.sample(n=pos_each, replace=pos_replace, random_state=seed + 11)
    neg_df = neg_pool.sample(n=neg_each, replace=neg_replace, random_state=seed + 23)
    train_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=seed + 97).reset_index(drop=True)

    meta = {
        "train_total": int(train_total),
        "pos_target": int(pos_each),
        "neg_target": int(neg_each),
        "pos_available": int(len(pos_pool)),
        "neg_available": int(len(neg_pool)),
        "pos_replace": bool(pos_replace),
        "neg_replace": bool(neg_replace),
    }
    return train_df, meta


def _load_predict_fn(model_path: Path):
    spec = importlib.util.spec_from_file_location("final_heuristic_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict_fn = getattr(module, "predict", None)
    if predict_fn is None:
        raise RuntimeError(f"`predict(features)` not found in {model_path}")
    return predict_fn


def _predict_labels(predict_fn, df: pd.DataFrame, label_col: str) -> np.ndarray:
    feature_cols = [c for c in df.columns if c != label_col]
    preds: list[int] = []
    for _, row in df.iterrows():
        feats = {c: row[c] for c in feature_cols}
        preds.append(int(predict_fn(feats)))
    return np.asarray(preds, dtype=int)


def _read_final_version(model_path: Path) -> str:
    try:
        code_all = model_path.read_text(encoding="utf-8")
        for line in code_all.splitlines():
            if line.startswith("FINAL_VERSION"):
                return str(json.loads(line.split("=", 1)[1].strip()))
    except Exception:
        return ""
    return ""


def _run_one(*, ds: DatasetSpec, ms: ModelSpec, seed: int, output_root: Path) -> dict:
    split_spec = SplitSpec(val_total=1000, test_total=1000)
    train_total = 1000

    df = _load_csv(ds.csv_path)
    if ds.label_col not in df.columns:
        raise ValueError(f"{ds.name}: label_col={ds.label_col} not found")
    df = df.copy()
    df[ds.label_col] = df[ds.label_col].astype(int)

    train_pool, val_df, test_df = _split_val_test_balanced(df, ds.label_col, split_spec, seed=seed)
    train_df, train_meta = _sample_train_balanced(train_pool, label_col=ds.label_col, train_total=train_total, seed=seed + 1000, spec=split_spec)

    out_dir = output_root / ds.name / ms.display_name.replace("/", "_") / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = RunConfig(
        output_dir=out_dir,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=f"Contrast0. Dataset={ds.name}. Model={ms.display_name}. TrainTotal=1000 balanced 1:1.",
        random_seed=int(seed),
    )
    llm_cfg = LLMConfig(
        base_url=ms.base_url,
        api_key_env=ms.api_key_env,
        model_name=ms.model_name,
        temperature=ms.temperature,
        extra_body=ms.extra_body,
    )

    run_heuristic_learning(train_df=train_df, test_df=val_df, label_col=ds.label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)

    model_path = out_dir / "final_heuristic_model.py"
    predict_fn = _load_predict_fn(model_path)

    y_true = test_df[ds.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, test_df, label_col=ds.label_col)
    metrics = compute_metrics(y_true, y_pred)
    final_version = _read_final_version(model_path)

    summary = {
        "dataset": ds.name,
        "csv_path": str(ds.csv_path),
        "label_col": ds.label_col,
        "seed": int(seed),
        "model_display_name": ms.display_name,
        "model_name": ms.model_name,
        "base_url": ms.base_url,
        "api_key_env": ms.api_key_env,
        "train_sampling": train_meta,
        "final_version": final_version,
        "heldout_test_metrics": metrics,
    }
    write_json(out_dir / "heldout_test_summary.json", summary)
    write_text(
        out_dir / "heldout_test_summary.txt",
        "\n".join(
            [
                f"dataset={ds.name}",
                f"model={ms.display_name}",
                f"final_version={final_version}",
                f"train_sampling={train_meta}",
                f"heldout_test_metrics={metrics}",
            ]
        )
        + "\n",
    )

    return {
        "大模型": ms.display_name,
        "数据集": ds.name,
        "ACC": f"{float(metrics.get('ACC')):.3f}" if metrics.get("ACC") is not None else "",
        "F1": f"{float(metrics.get('F1')):.3f}" if metrics.get("F1") is not None else "",
        "Sensitivity": f"{float(metrics.get('Sensitivity')):.3f}" if metrics.get("Sensitivity") is not None else "",
        "Specificity": f"{float(metrics.get('Specificity')):.3f}" if metrics.get("Specificity") is not None else "",
    }


def _run_one_safe(*, ds: DatasetSpec, ms: ModelSpec, seed: int, output_root: Path) -> dict:
    try:
        print(f"[contrast0] start dataset={ds.name} model={ms.display_name}", flush=True)
        r = _run_one(ds=ds, ms=ms, seed=seed, output_root=output_root)
        print(f"[contrast0] done dataset={ds.name} model={ms.display_name}", flush=True)
        return r
    except Exception as e:
        print(f"[contrast0] failed dataset={ds.name} model={ms.display_name} error={e}", flush=True)
        return {
            "大模型": ms.display_name,
            "数据集": ds.name,
            "ACC": "",
            "F1": "",
            "Sensitivity": "",
            "Specificity": "",
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-root", type=str, default=str(script_dir / "outputs"))
    args = p.parse_args()

    seed = int(args.seed)
    workers = int(args.workers)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    deepseek_base = "https://api.deepseek.com/v1"
    deepseek_key_env = "DEEPSEEK_API_KEY"

    router_base = "https://openrouter.ai/api/v1"
    router_key_env = "OPENROUTER_API_KEY"

    vveai_base = "https://api.vveai.com"
    vveai_gemini_key_env = "VVEAI_GEMINI_API_KEY"
    vveai_gpt55_key_env = "VVEAI_GPT55_API_KEY"

    if os.getenv("CONTRAST0_DEEPSEEK_BASE_URL"):
        deepseek_base = os.getenv("CONTRAST0_DEEPSEEK_BASE_URL", deepseek_base)
    if os.getenv("CONTRAST0_DEEPSEEK_KEY_ENV"):
        deepseek_key_env = os.getenv("CONTRAST0_DEEPSEEK_KEY_ENV", deepseek_key_env)

    if os.getenv("CONTRAST0_ROUTER_BASE_URL"):
        router_base = os.getenv("CONTRAST0_ROUTER_BASE_URL", router_base)
    if os.getenv("CONTRAST0_ROUTER_KEY_ENV"):
        router_key_env = os.getenv("CONTRAST0_ROUTER_KEY_ENV", router_key_env)

    if os.getenv("CONTRAST0_VVEAI_BASE_URL"):
        vveai_base = os.getenv("CONTRAST0_VVEAI_BASE_URL", vveai_base)
    if os.getenv("CONTRAST0_VVEAI_GEMINI_KEY_ENV"):
        vveai_gemini_key_env = os.getenv("CONTRAST0_VVEAI_GEMINI_KEY_ENV", vveai_gemini_key_env)
    if os.getenv("CONTRAST0_VVEAI_GPT55_KEY_ENV"):
        vveai_gpt55_key_env = os.getenv("CONTRAST0_VVEAI_GPT55_KEY_ENV", vveai_gpt55_key_env)

    vveai_base = vveai_base.rstrip("/")
    if not vveai_base.endswith("/v1"):
        vveai_base = vveai_base + "/v1"

    models = [
        ModelSpec(
            display_name="deepseek-v4-pro",
            base_url=deepseek_base,
            api_key_env=deepseek_key_env,
            model_name="deepseek-v4-pro",
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        ),
        ModelSpec(
            display_name="deepseek-v4-pro-thinking",
            base_url=deepseek_base,
            api_key_env=deepseek_key_env,
            model_name="deepseek-v4-pro",
            temperature=0.0,
            extra_body={"thinking": {"type": "enabled"}},
        ),
        ModelSpec(
            display_name="deepseek-v4-flash",
            base_url=deepseek_base,
            api_key_env=deepseek_key_env,
            model_name="deepseek-v4-flash",
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        ),
        ModelSpec(
            display_name="qwen/qwen3.7-max",
            base_url=router_base,
            api_key_env=router_key_env,
            model_name="qwen/qwen3.7-max",
            temperature=0.0,
        ),
        ModelSpec(
            display_name="gemini-3.1-pro-preview",
            base_url=vveai_base,
            api_key_env=vveai_gemini_key_env,
            model_name="gemini-3.1-pro-preview",
            temperature=0.0,
        ),
        ModelSpec(
            display_name="gpt-5.5",
            base_url=vveai_base,
            api_key_env=vveai_gpt55_key_env,
            model_name="gpt-5.5",
            temperature=0.0,
        ),
    ]

    datasets = [
        DatasetSpec("UKB", repo_root / "data" / "UKB.csv", "label"),
        DatasetSpec("YHD", repo_root / "data" / "YHD_bicarbonate.csv", "hospital_expire_flag"),
    ]

    tasks = [(ds, ms) for ms in models for ds in datasets]
    results: list[dict] = []
    if workers <= 1:
        for ds, ms in tasks:
            results.append(_run_one_safe(ds=ds, ms=ms, seed=seed, output_root=output_root))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one_safe, ds=ds, ms=ms, seed=seed, output_root=output_root) for ds, ms in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())

    model_order = {m.display_name: i for i, m in enumerate(models)}
    dataset_order = {"UKB": 0, "YHD": 1}

    def sort_key(r: dict) -> tuple:
        m = str(r.get("大模型", ""))
        ds = str(r.get("数据集", ""))
        return (model_order.get(m, 99), dataset_order.get(ds, 99))

    results.sort(key=sort_key)

    out_csv = script_dir / "contrast0.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["大模型", "数据集", "ACC", "F1", "Sensitivity", "Specificity"],
        )
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in writer.fieldnames} for r in results])

    print(f"contrast0_csv={out_csv}", flush=True)


if __name__ == "__main__":
    main()
