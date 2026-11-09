"""验证集类别比例扫描实验（UKB + MHL）。

本脚本完成三件事：

1. 按种子构建固定的、互不重叠的数据划分：
   - 训练集 3000 条（正负各 1500，1:1）；
   - 测试集 1000 条（正负各 500，1:1）；
   - MHL 内部验证集 3000 条（正负各 1500，1:1）；
   - 7 个"验证集类别比例"集合，每个集合 3000 条：
     1:1、1:2、2:1、1:4、4:1、1:8、8:1。
2. 使用 MHL（run_heuristic_learning）训练白盒规则模型；
   训练过程中以 3000 条 1:1 验证集做版本选择。
3. 加载最终 `final_heuristic_model.py`，在 7 个不同类别比例的验证集上
   逐个评估，记录 ACC、F1、Sensitivity、Specificity、TP、FP、FN、TN。

7 个比例验证集各自使用独立的 3000 条新样本（与训练/测试/MHL 验证集以及
彼此之间均互不重叠）。这样保证每个比例下的验证集总量都是 3000，
且不依赖有放回抽样；原始 UKB 数据量足以支撑该设计。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hl.config import LLMConfig, RunConfig
from hl.metrics import compute_metrics
from hl.orchestrator import run_heuristic_learning
from hl.utils.io import write_json, write_text

ROW_ID_COL = "__mhl_row_id__"
DATASET_NAME = "UKB"
CSV_PATH = REPO_ROOT / "data" / "UKB.csv"
LABEL_COL = "label"

FIELD_NAMES = [
    "种子",
    "数据集",
    "验证集比例",
    "验证集总量",
    "正例数",
    "负例数",
    "ACC",
    "F1",
    "Sensitivity",
    "Specificity",
    "TP",
    "FP",
    "FN",
    "TN",
    "status",
    "error",
    "out_dir",
]


@dataclass(frozen=True)
class RatioSpec:
    """正类:负类比例，例如 1:2 表示 1 个正例对应 2 个负例。"""

    pos: int
    neg: int

    @property
    def name(self) -> str:
        return f"{self.pos}:{self.neg}"

    @property
    def slug(self) -> str:
        return f"{self.pos}_{self.neg}"

    def __str__(self) -> str:
        return self.name


DEFAULT_RATIOS = (
    RatioSpec(1, 1),
    RatioSpec(1, 2),
    RatioSpec(2, 1),
    RatioSpec(1, 4),
    RatioSpec(4, 1),
    RatioSpec(1, 8),
    RatioSpec(8, 1),
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    frame = pd.read_csv(CSV_PATH)
    if LABEL_COL not in frame.columns:
        raise ValueError(f"Label column not found: {LABEL_COL}")
    if ROW_ID_COL in frame.columns:
        raise ValueError(f"Reserved column already exists: {ROW_ID_COL}")
    frame = frame.copy()
    frame[LABEL_COL] = frame[LABEL_COL].astype(int)
    frame[ROW_ID_COL] = np.arange(len(frame), dtype=int)
    return frame


def _counts_from_ratio(total: int, ratio: RatioSpec) -> tuple[int, int]:
    """按 total 和 pos:neg 比例求正/负样本目标数，保证二者之和为 total。"""
    denominator = ratio.pos + ratio.neg
    pos = int(round(total * ratio.pos / denominator))
    pos = max(1, min(pos, total - 1))
    return pos, total - pos


def _row_ids_hash(row_ids: list[int]) -> str:
    payload = ",".join(str(int(i)) for i in row_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _block_meta(raw: pd.DataFrame, label_col: str) -> dict:
    ids = raw[ROW_ID_COL].astype(int).tolist()
    y = raw[label_col].astype(int)
    return {
        "rows": int(len(raw)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "source_row_hash": _row_ids_hash(ids),
    }


def _build_split(ids: np.ndarray, source: pd.DataFrame, rand_state: int) -> pd.DataFrame:
    """按原始索引收集行，并打乱后重置索引。"""
    raw = source.loc[ids].copy().sample(frac=1.0, random_state=rand_state)
    return raw.reset_index(drop=True)


def _build_frozen_splits(
    frame: pd.DataFrame,
    *,
    label_col: str,
    seed: int,
    train_total: int,
    test_total: int,
    mhl_val_total: int,
    val_ratio_total: int,
    ratios: tuple[RatioSpec, ...],
) -> dict:
    """构建全部互不重叠的冻结数据划分。

    返回 dict，包含：
    - train_raw / test_raw / mhl_val_raw：带 ROW_ID_COL 的原始 DataFrame；
    - ratio_raw：{ratio.name: DataFrame}；
    - manifest：划分元数据（数量、哈希、无交集校验）。
    """
    if train_total % 2 != 0 or test_total % 2 != 0 or mhl_val_total % 2 != 0:
        raise ValueError("train_total/test_total/mhl_val_total must be even (1:1 balanced)")

    y = frame[label_col].astype(int)
    pos_df = frame.loc[y == 1].copy()
    neg_df = frame.loc[y == 0].copy()

    train_each = train_total // 2
    test_each = test_total // 2
    mhl_val_each = mhl_val_total // 2
    ratio_counts = [(r, *_counts_from_ratio(val_ratio_total, r)) for r in ratios]

    pos_needed = (
        train_each
        + test_each
        + mhl_val_each
        + sum(p for _, p, _n in ratio_counts)
    )
    neg_needed = (
        train_each
        + test_each
        + mhl_val_each
        + sum(n for _, _p, n in ratio_counts)
    )
    if len(pos_df) < pos_needed or len(neg_df) < neg_needed:
        raise ValueError(
            f"Not enough samples: need pos>={pos_needed}, neg>={neg_needed}; "
            f"got pos={len(pos_df)}, neg={len(neg_df)}"
        )

    rng = np.random.default_rng(seed)
    pos_order = rng.permutation(pos_df.index.to_numpy())
    neg_order = rng.permutation(neg_df.index.to_numpy())
    pos_cursor = [0]
    neg_cursor = [0]

    def take_pos(n: int) -> np.ndarray:
        start = pos_cursor[0]
        pos_cursor[0] = start + n
        return pos_order[start : start + n]

    def take_neg(n: int) -> np.ndarray:
        start = neg_cursor[0]
        neg_cursor[0] = start + n
        return neg_order[start : start + n]

    train_raw = _build_split(take_pos(train_each), pos_df, seed + 101)
    train_raw = pd.concat([train_raw, _build_split(take_neg(train_each), neg_df, seed + 102)], axis=0).sample(
        frac=1.0, random_state=seed + 103
    ).reset_index(drop=True)

    test_raw = _build_split(take_pos(test_each), pos_df, seed + 201)
    test_raw = pd.concat([test_raw, _build_split(take_neg(test_each), neg_df, seed + 202)], axis=0).sample(
        frac=1.0, random_state=seed + 203
    ).reset_index(drop=True)

    mhl_val_raw = _build_split(take_pos(mhl_val_each), pos_df, seed + 301)
    mhl_val_raw = pd.concat([mhl_val_raw, _build_split(take_neg(mhl_val_each), neg_df, seed + 302)], axis=0).sample(
        frac=1.0, random_state=seed + 303
    ).reset_index(drop=True)

    ratio_raw: dict[str, pd.DataFrame] = {}
    for idx, (ratio, p_n, n_n) in enumerate(ratio_counts):
        part = _build_split(take_pos(p_n), pos_df, seed + 400 + idx * 10 + 1)
        part = pd.concat([part, _build_split(take_neg(n_n), neg_df, seed + 400 + idx * 10 + 2)], axis=0).sample(
            frac=1.0, random_state=seed + 400 + idx * 10 + 3
        ).reset_index(drop=True)
        ratio_raw[ratio.name] = part

    # 校验所有划分互不重叠
    named_blocks = {
        "train": train_raw,
        "test": test_raw,
        "mhl_val": mhl_val_raw,
        **{f"ratio_{ratio.slug}": ratio_raw[ratio.name] for ratio in ratios},
    }
    all_ids: dict[str, set[int]] = {
        name: set(raw[ROW_ID_COL].astype(int).tolist()) for name, raw in named_blocks.items()
    }
    names = list(all_ids.keys())
    overlaps: dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            inter = all_ids[left] & all_ids[right]
            if inter:
                overlaps[f"{left}∩{right}"] = len(inter)
    if overlaps:
        raise ValueError(f"Frozen splits overlap unexpectedly: {overlaps}")

    manifest = {
        "dataset": DATASET_NAME,
        "csv_path": str(CSV_PATH),
        "label_col": label_col,
        "seed": int(seed),
        "train_total": int(train_total),
        "test_total": int(test_total),
        "mhl_val_total": int(mhl_val_total),
        "val_ratio_total": int(val_ratio_total),
        "ratios": [r.name for r in ratios],
        "blocks": {name: _block_meta(raw, label_col) for name, raw in named_blocks.items()},
        "all_disjoint": True,
        "overlap_count": len(overlaps),
    }
    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "mhl_val_raw": mhl_val_raw,
        "ratio_raw": ratio_raw,
        "manifest": manifest,
    }


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
        code = model_path.read_text(encoding="utf-8")
        for line in code.splitlines():
            if line.startswith("FINAL_VERSION"):
                return str(json.loads(line.split("=", 1)[1].strip()))
    except Exception:
        return ""
    return ""


def _save_split_tables(out_dir: Path, splits: dict) -> None:
    data_dir = out_dir / "data_splits"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.joinpath("train.csv").write_text(splits["train_raw"].to_csv(index=False), encoding="utf-8")
    data_dir.joinpath("test.csv").write_text(splits["test_raw"].to_csv(index=False), encoding="utf-8")
    data_dir.joinpath("mhl_val.csv").write_text(splits["mhl_val_raw"].to_csv(index=False), encoding="utf-8")
    for ratio_name, raw in splits["ratio_raw"].items():
        slug = ratio_name.replace(":", "_")
        data_dir.joinpath(f"ratio_{slug}.csv").write_text(raw.to_csv(index=False), encoding="utf-8")
    write_json(data_dir / "split_manifest.json", splits["manifest"])


def _train_hl(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    seed: int,
    iterations: int,
    llm_cfg: LLMConfig,
) -> Path:
    run_cfg = RunConfig(
        output_dir=out_dir,
        iterations=iterations,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=(
            f"Dataset={DATASET_NAME}. Probes=U1_K1. "
            f"TrainSize=3000 balanced 1:1. TestSize=1000 balanced 1:1. "
            "MHL internal validation set 3000 balanced 1:1. "
            "After training, the final white-box rule model is evaluated on "
            "validation sets with different positive:negative ratios (1:1,1:2,2:1,1:4,4:1,1:8,8:1)."
        ),
        random_seed=int(seed),
        llm_enabled=True,
    )
    run_heuristic_learning(
        train_df=train_df,
        val_df=val_df,
        label_col=label_col,
        run_cfg=run_cfg,
        llm_cfg=llm_cfg,
    )
    model_path = out_dir / "final_heuristic_model.py"
    if not model_path.exists():
        raise RuntimeError(f"MHL did not produce final model: {model_path}")
    return model_path


def _format_metric(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _evaluate_ratio_sets(
    *,
    predict_fn,
    ratio_raw: dict[str, pd.DataFrame],
    label_col: str,
    out_dir: Path,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for ratio in DEFAULT_RATIOS:
        raw = ratio_raw[ratio.name]
        df = raw.drop(columns=[ROW_ID_COL])
        y_true = df[label_col].astype(int).to_numpy()
        y_pred = _predict_labels(predict_fn, df, label_col=label_col)
        metrics = compute_metrics(y_true, y_pred)

        predictions_path = out_dir / f"predictions_ratio_{ratio.slug}.csv"
        pd.DataFrame(
            {
                ROW_ID_COL: raw[ROW_ID_COL].astype(int),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        ).to_csv(predictions_path, index=False)

        rows.append(
            {
                "种子": int(seed),
                "数据集": DATASET_NAME,
                "验证集比例": ratio.name,
                "验证集总量": int(len(raw)),
                "正例数": int((raw[label_col] == 1).sum()),
                "负例数": int((raw[label_col] == 0).sum()),
                "ACC": _format_metric(metrics.get("ACC")),
                "F1": _format_metric(metrics.get("F1")),
                "Sensitivity": _format_metric(metrics.get("Sensitivity")),
                "Specificity": _format_metric(metrics.get("Specificity")),
                "TP": int(metrics["TP"]),
                "FP": int(metrics["FP"]),
                "FN": int(metrics["FN"]),
                "TN": int(metrics["TN"]),
                "status": "ok",
                "error": "",
                "out_dir": str(out_dir),
            }
        )
    return rows


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in FIELD_NAMES} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "UKB MHL validation-ratio sweep: train once on a 3000/1000/3000 "
            "(train/test/MHL-val) frozen split, then evaluate the best rule on "
            "7 ratio-specific 3000-row validation sets."
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--train-total", type=int, default=3000, help="Balanced 1:1 train rows (default 3000)")
    parser.add_argument("--test-total", type=int, default=1000, help="Balanced 1:1 test rows (default 1000)")
    parser.add_argument("--mhl-val-total", type=int, default=3000, help="Balanced 1:1 MHL evolution val rows (default 3000)")
    parser.add_argument("--val-ratio-total", type=int, default=3000, help="Rows in each ratio-specific validation set (default 3000)")
    parser.add_argument("--iterations", type=int, default=10, help="MHL rule evolution iterations (default 10)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Skip MHL training and evaluate this existing final_heuristic_model.py",
    )
    parser.add_argument(
        "--check-split",
        action="store_true",
        help=(
            "Only build and verify the frozen splits (no LLM calls, no training). "
            "Useful to inspect counts/overlap before spending API budget."
        ),
    )
    parser.add_argument("--output-root", type=str, default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--csv-out", type=str, default="", help="CSV output path; default experiment/valratio/valratio_hl_seed<seed>.csv")
    args = parser.parse_args()

    seed = int(args.seed)
    train_total = int(args.train_total)
    test_total = int(args.test_total)
    mhl_val_total = int(args.mhl_val_total)
    val_ratio_total = int(args.val_ratio_total)
    iterations = int(args.iterations)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[valratio] loading dataset {CSV_PATH}", flush=True)
    frame = _load_dataset()
    splits = _build_frozen_splits(
        frame,
        label_col=LABEL_COL,
        seed=seed,
        train_total=train_total,
        test_total=test_total,
        mhl_val_total=mhl_val_total,
        val_ratio_total=val_ratio_total,
        ratios=DEFAULT_RATIOS,
    )
    manifest = splits["manifest"]
    print(
        f"[valratio] splits ok: train={manifest['blocks']['train']['rows']} "
        f"test={manifest['blocks']['test']['rows']} "
        f"mhl_val={manifest['blocks']['mhl_val']['rows']}",
        flush=True,
    )
    for ratio_name, raw in splits["ratio_raw"].items():
        print(
            f"[valratio] ratio {ratio_name}: rows={len(raw)} "
            f"pos={int((raw[LABEL_COL] == 1).sum())} neg={int((raw[LABEL_COL] == 0).sum())}",
            flush=True,
        )

    if args.check_split:
        check_root = output_root / f"seed{seed}" / f"check_{_timestamp()}"
        check_root.mkdir(parents=True, exist_ok=True)
        _save_split_tables(check_root, splits)
        print(f"[valratio] split-check written to {check_root}", flush=True)
        return

    run_dir = output_root / f"seed{seed}" / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_split_tables(run_dir, splits)

    train_df = splits["train_raw"].drop(columns=[ROW_ID_COL])
    mhl_val_df = splits["mhl_val_raw"].drop(columns=[ROW_ID_COL])
    test_df = splits["test_raw"].drop(columns=[ROW_ID_COL])

    llm_cfg = LLMConfig(
        base_url=os.getenv("VALRATIO_HL_BASE_URL", "https://api.deepseek.com/v1"),
        api_key_env=os.getenv("VALRATIO_HL_KEY_ENV", "DEEPSEEK_API_KEY"),
        model_name=os.getenv("VALRATIO_HL_MODEL", "deepseek-v4-pro"),
        temperature=float(os.getenv("VALRATIO_HL_TEMPERATURE", "0.0")),
    )

    if args.model_path:
        model_path = Path(args.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"[valratio] evaluate-only mode, model={model_path}", flush=True)
    else:
        print(f"[valratio] training MHL: train={len(train_df)} val={len(mhl_val_df)}", flush=True)
        model_path = _train_hl(
            train_df=train_df,
            val_df=mhl_val_df,
            label_col=LABEL_COL,
            out_dir=run_dir,
            seed=seed,
            iterations=iterations,
            llm_cfg=llm_cfg,
        )
        print(f"[valratio] MHL training done, model={model_path}", flush=True)

    predict_fn = _load_predict_fn(model_path)
    final_version = _read_final_version(model_path)

    rows = _evaluate_ratio_sets(
        predict_fn=predict_fn,
        ratio_raw=splits["ratio_raw"],
        label_col=LABEL_COL,
        out_dir=run_dir,
        seed=seed,
    )

    # 平衡测试集指标作为额外审计信息（只写入 JSON，不进入比例扫描 CSV）
    test_y_true = test_df[LABEL_COL].astype(int).to_numpy()
    test_y_pred = _predict_labels(predict_fn, test_df, label_col=LABEL_COL)
    test_metrics = compute_metrics(test_y_true, test_y_pred)
    pd.DataFrame(
        {
            ROW_ID_COL: splits["test_raw"][ROW_ID_COL].astype(int),
            "y_true": test_y_true,
            "y_pred": test_y_pred,
        }
    ).to_csv(run_dir / "predictions_test.csv", index=False)

    summary = {
        "dataset": DATASET_NAME,
        "label_col": LABEL_COL,
        "seed": int(seed),
        "train_total": int(train_total),
        "test_total": int(test_total),
        "mhl_val_total": int(mhl_val_total),
        "val_ratio_total": int(val_ratio_total),
        "iterations": int(iterations),
        "model_path": str(model_path),
        "final_version": final_version,
        "data_manifest": manifest,
        "test_metrics": test_metrics,
        "ratio_results": rows,
    }
    write_json(run_dir / "validation_ratio_summary.json", summary)
    write_text(
        run_dir / "validation_ratio_summary.txt",
        "\n".join(
            [
                f"dataset={DATASET_NAME}",
                f"seed={seed}",
                f"train_total={train_total}",
                f"test_total={test_total}",
                f"mhl_val_total={mhl_val_total}",
                f"val_ratio_total={val_ratio_total}",
                f"final_version={final_version}",
                f"model_path={model_path}",
                f"test_metrics={test_metrics}",
                "",
                "ratio_results:",
                *[f"  {row['验证集比例']}: ACC={row['ACC']} F1={row['F1']} "
                  f"Sens={row['Sensitivity']} Spec={row['Specificity']} "
                  f"TP={row['TP']} FP={row['FP']} FN={row['FN']} TN={row['TN']}" for row in rows],
            ]
        )
        + "\n",
    )

    if args.csv_out:
        csv_path = Path(args.csv_out).resolve()
    else:
        csv_path = SCRIPT_DIR / f"valratio_hl_seed{seed}.csv"
    _write_results_csv(csv_path, rows)
    print(f"[valratio] results written to {csv_path}", flush=True)
    print(f"[valratio] run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
