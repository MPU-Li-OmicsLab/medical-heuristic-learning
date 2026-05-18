from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agent.client import ChatMessage, LLMClient
from agent.prompts import get_iteration_prompt, get_rule_generation_prompt
from config import LLMConfig, RunConfig
from evolution.degradation import detect_degradation, format_degradation_warning
from evolution.error_analysis import collect_errors, format_error_report
from evolution.rule_utils import ParsedProposal, extract_function_name, strip_code_fences, validate_python_syntax
from metrics import compute_metrics, generate_metric_description
from probes.knowledge import run_knowledge_probe
from probes.univariate import run_univariate_probe
from utils.data_loader import load_csv
from utils.io import append_text, ensure_dir, write_json, write_text


@dataclass(frozen=True)
class IterationRecord:
    version: str
    error_analysis: str
    metrics: dict[str, float]


def _predict_with_function(fn: Callable[[dict], int], df: pd.DataFrame, label_col: str) -> np.ndarray:
    feature_cols = [c for c in df.columns if c != label_col]
    preds: list[int] = []
    for _, row in df.iterrows():
        feats = {c: row[c] for c in feature_cols}
        try:
            p = int(fn(feats))
        except Exception:
            p = 0
        preds.append(1 if p == 1 else 0)
    return np.asarray(preds, dtype=int)


def _default_rule_v0_code(feature_cols: list[str]) -> tuple[str, str]:
    core = f"""def predict_v0(features: dict) -> int:
    def _get_float(k: str, default: float = 0.0) -> float:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _get_int(k: str, default: int = 0) -> int:
        v = features.get(k, default)
        try:
            if v is None:
                return default
            return int(float(v))
        except Exception:
            return default

    score = 0
    if _get_int("Respiratory_failure") == 1:
        score += 4

    age = _get_float("age")
    if age >= 85:
        score += 2
    elif age >= 75:
        score += 1

    if _get_int("sepsis") == 1:
        score += 2

    sofa = _get_float("SOFA")
    if sofa >= 7:
        score += 2
    elif sofa >= 5:
        score += 1

    if _get_float("wbc") < 8.0:
        score += 1
    if _get_int("Cerebral_infarction") == 1:
        score += 1
    if _get_int("ICH") == 1:
        score += 1
    if _get_float("mbp") < 85.0:
        score += 1
    if _get_float("Sodium") < 136.0:
        score += 1
    if _get_float("hemoglobin") < 9.5:
        score += 1
    if _get_float("bicarbonate") < 18.0:
        score += 1

    if _get_float("Urea_Nitrogen") < 12.0 and _get_float("bicarbonate") >= 20.0 and _get_float("SOFA") <= 6.0:
        score -= 2

    if _get_float("SOFA") >= 7.0 and _get_float("Urea_Nitrogen") < 10.0 and _get_float("bicarbonate") >= 20.0:
        score -= 1

    return 1 if score >= 5 else 0
"""
    tests = """TESTS = [
    {"name": "predict_returns_int", "code": "assert predict_v0({}) in (0, 1)"},
]
"""
    return core, tests


def _load_heuristic_module(path: Path) -> dict:
    ns: dict = {}
    code = path.read_text(encoding="utf-8") if path.exists() else ""
    exec(compile(code, str(path), "exec"), ns, ns)
    return ns


def _append_new_version(path: Path, version_code: str, error_analysis: str, new_tests: list[dict]) -> None:
    block = "\n\n" + version_code.strip() + "\n"
    if error_analysis:
        block += f'\nERROR_ANALYSIS_{extract_function_name(version_code) or ""} = {json.dumps(error_analysis, ensure_ascii=False)}\n'
    if new_tests:
        block += "\nTESTS.extend([\n"
        for t in new_tests:
            name = t.get("name", "")
            code = t.get("code", "")
            block += f"    {{'name': {json.dumps(name, ensure_ascii=False)}, 'code': {json.dumps(code, ensure_ascii=False)}}},\n"
        block += "])\n"
    append_text(path, block)


def _parse_proposal(text: str) -> ParsedProposal:
    raw = strip_code_fences(text)
    data = json.loads(raw)
    return ParsedProposal(
        version=str(data.get("version", "")),
        error_analysis=str(data.get("error_analysis", "")),
        new_policy_code=str(data.get("new_policy_code", "")),
        new_tests=list(data.get("new_tests", []) or []),
        modified_tests=list(data.get("modified_tests", []) or []),
    )


def run_heuristic_learning(run_cfg: RunConfig, llm_cfg: LLMConfig) -> None:
    ensure_dir(run_cfg.output_dir)
    df = load_csv(run_cfg.data_csv_path)

    if run_cfg.label_col not in df.columns:
        raise ValueError(f"label_col={run_cfg.label_col} 不存在于 CSV 列中。")

    train_df, test_df = train_test_split(
        df,
        test_size=run_cfg.test_size,
        random_state=run_cfg.random_seed,
        stratify=df[run_cfg.label_col].astype(int),
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != run_cfg.label_col]

    univariate_df = run_univariate_probe(train_df=train_df, label_col=run_cfg.label_col)
    univariate_path = run_cfg.output_dir / "probe_univariate_results.csv"
    univariate_df.to_csv(univariate_path, index=False)

    topk = min(run_cfg.univariate_top_k, len(univariate_df))
    top_features = univariate_df.head(topk)["feature"].tolist()
    univariate_summary = univariate_df.head(topk)[
        ["rank", "feature", "feature_type", "method", "p_value", "missing_rate", "pointbiserial_r", "mwu_p", "binned_or_q4_rel_to_q1"]
    ].to_string(index=False)

    metric_desc = generate_metric_description(run_cfg.metric_priority)

    heuristic_path = run_cfg.output_dir / "heuristic_system.py"
    evolution_results_path = run_cfg.output_dir / "evolution_results.txt"
    iteration_log_path = run_cfg.output_dir / "iteration_log.json"
    final_model_path = run_cfg.output_dir / "final_heuristic_model.py"
    final_comparison_path = run_cfg.output_dir / "final_comparison.txt"
    knowledge_path = run_cfg.output_dir / "probe_knowledge.md"

    trajectory_lines: list[str] = []
    iteration_log: list[dict] = []
    records: list[IterationRecord] = []

    client: LLMClient | None = None
    if run_cfg.llm_enabled:
        client = LLMClient(
            base_url=llm_cfg.base_url,
            api_key_env=llm_cfg.api_key_env,
            model_name=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
        )

    knowledge_table = ""
    if client is not None:
        knowledge = run_knowledge_probe(client=client, features=top_features[: run_cfg.knowledge_top_k], target=run_cfg.label_col)
        knowledge_table = knowledge.markdown_table
        write_text(knowledge_path, knowledge_table)
    else:
        write_text(knowledge_path, "")

    if not heuristic_path.exists():
        header = "CURRENT_VERSION = 'v0'\n\n"
        default_code, default_tests = _default_rule_v0_code(feature_cols=feature_cols)
        write_text(heuristic_path, header + default_code.strip() + "\n\n" + default_tests.strip() + "\n")

    ns = _load_heuristic_module(heuristic_path)
    fn_v0 = ns.get("predict_v0")
    if fn_v0 is None:
        raise RuntimeError("heuristic_system.py 中未找到 predict_v0")

    y_pred_test_v0 = _predict_with_function(fn_v0, test_df, run_cfg.label_col)
    metrics_v0 = compute_metrics(test_df[run_cfg.label_col].astype(int).to_numpy(), y_pred_test_v0).values
    records.append(IterationRecord(version="v0", error_analysis="default_v0", metrics=metrics_v0))
    append_text(evolution_results_path, f"v0\t{metrics_v0}\n")
    trajectory_lines.append("V0: 初始化默认规则。")

    current_version = "v0"
    current_fn = fn_v0

    for i in range(1, run_cfg.iterations + 1):
        next_version = f"v{i}"
        y_pred_train = _predict_with_function(current_fn, train_df, run_cfg.label_col)
        samples = collect_errors(
            df=train_df,
            label_col=run_cfg.label_col,
            y_pred=y_pred_train,
            max_error_samples=run_cfg.max_error_samples,
            random_seed=run_cfg.random_seed + i,
        )
        error_report = format_error_report(samples)

        degradation_warning = "无退化检测（尚未提案）。"

        current_code = heuristic_path.read_text(encoding="utf-8")

        proposal: ParsedProposal | None = None
        accepted = False
        attempt = 0
        while attempt < 2 and not accepted:
            attempt += 1
            if client is None:
                break

            prompt = get_iteration_prompt(
                current_code=current_code,
                error_report=error_report,
                trajectory="\n".join(trajectory_lines),
                degradation_warning=degradation_warning,
                metric_desc=metric_desc,
                next_version=next_version,
            )
            resp = client.chat_json([ChatMessage(role="user", content=prompt)])
            try:
                proposal = _parse_proposal(resp)
            except Exception as e:
                degradation_warning = f"JSON 解析失败：{e}"
                continue

            if proposal.version != next_version:
                degradation_warning = f"version 不匹配：期望 {next_version}，实际 {proposal.version}"
                continue

            new_code = proposal.new_policy_code
            validate_python_syntax(new_code)
            fn_name = extract_function_name(new_code)
            if fn_name != f"predict_{next_version}":
                degradation_warning = f"函数名不匹配：期望 predict_{next_version}，实际 {fn_name}"
                continue

            tmp_ns: dict = {}
            exec(compile(current_code + "\n\n" + new_code, "<heuristic>", "exec"), tmp_ns, tmp_ns)
            new_fn = tmp_ns.get(f"predict_{next_version}")
            if new_fn is None:
                degradation_warning = "新函数未定义成功。"
                continue

            y_pred_old = y_pred_train
            y_pred_new = _predict_with_function(new_fn, train_df, run_cfg.label_col)
            degr = detect_degradation(
                y_true=train_df[run_cfg.label_col].astype(int).to_numpy(),
                y_pred_old=y_pred_old,
                y_pred_new=y_pred_new,
            )
            if len(degr.degraded_indices) > run_cfg.degradation_threshold:
                degradation_warning = format_degradation_warning(degr.degraded_indices)
                continue

            _append_new_version(
                heuristic_path,
                version_code=new_code,
                error_analysis=proposal.error_analysis,
                new_tests=proposal.new_tests,
            )

            ns = _load_heuristic_module(heuristic_path)
            current_fn = ns.get(f"predict_{next_version}")
            if current_fn is None:
                degradation_warning = "新版本写入后无法加载。"
                continue

            current_version = next_version
            trajectory_lines.append(f"V{i}: {proposal.error_analysis}")
            accepted = True

            y_pred_test = _predict_with_function(current_fn, test_df, run_cfg.label_col)
            m = compute_metrics(test_df[run_cfg.label_col].astype(int).to_numpy(), y_pred_test).values
            records.append(IterationRecord(version=current_version, error_analysis=proposal.error_analysis, metrics=m))
            append_text(evolution_results_path, f"{current_version}\t{m}\n")

            iteration_log.append(
                {
                    "version": current_version,
                    "attempt": attempt,
                    "error_report": error_report,
                    "degradation_warning": degradation_warning,
                    "proposal": asdict(proposal),
                    "test_metrics": m,
                }
            )

        if client is None:
            break

        if not accepted:
            iteration_log.append(
                {
                    "version": next_version,
                    "attempt": attempt,
                    "error_report": error_report,
                    "degradation_warning": degradation_warning,
                    "proposal": None,
                    "test_metrics": None,
                }
            )
            break

    write_json(iteration_log_path, iteration_log)

    final_ns = _load_heuristic_module(heuristic_path)
    final_fn = final_ns.get(f"predict_{current_version}")
    if final_fn is None:
        raise RuntimeError("最终版本函数缺失。")

    export_code = heuristic_path.read_text(encoding="utf-8")
    write_text(final_model_path, export_code)

    if len(records) >= 1:
        v0 = records[0]
        v_last = records[-1]
        write_text(final_comparison_path, f"V0={v0.metrics}\nFINAL({v_last.version})={v_last.metrics}\n")


def main() -> None:
    cfg = RunConfig(
        data_csv_path=Path("/data/yk/HL/data/YHD_bicarbonate.csv"),
        label_col="hospital_expire_flag",
        output_dir=Path("/data/yk/HL/out"),
        iterations=10,
        metric_priority=("F1", "ACC"),
        max_error_samples=200,
        degradation_threshold=10,
        llm_enabled=True,
    )
    llm = LLMConfig()
    run_heuristic_learning(cfg, llm)


if __name__ == "__main__":
    main()

