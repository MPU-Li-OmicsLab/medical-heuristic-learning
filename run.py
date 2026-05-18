from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agent.client import ChatMessage, LLMClient
from agent.prompts import get_iteration_prompt, get_rule_generation_prompt
from config import LLMConfig, RunConfig
from evolution.degradation import collect_degradation_examples, detect_degradation, format_degradation_warning
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


def _pick_report_features(feature_cols: list[str], top_features: list[str]) -> list[str]:
    preferred = [
        "Respiratory_failure",
        "SOFA",
        "resp_rate",
        "sepsis",
        "Kidney_failure",
        "Urea_Nitrogen",
        "heart_rate",
        "age",
        "inr",
        "temperature",
        "ICH",
        "Chloride",
        "bicarbonate",
        "Cerebral_infarction",
        "ARDS",
        "wbc",
        "mbp",
        "Sodium",
        "hemoglobin",
        "platelet",
    ]
    selected: list[str] = []
    for f in list(dict.fromkeys(top_features + preferred)):
        if f in feature_cols:
            selected.append(f)
    return selected


def _run_assert_tests(ns: dict, tests: Iterable[dict]) -> tuple[bool, str]:
    failed: list[str] = []
    for t in tests:
        name = str(t.get("name", ""))
        code = str(t.get("code", ""))
        if not code.strip():
            continue
        try:
            exec(code, ns, ns)
        except Exception as e:
            failed.append(f"{name}: {e}")
            if len(failed) >= 10:
                break
    if failed:
        return False, "；".join(failed)
    return True, ""


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
    y_true_train = train_df[run_cfg.label_col].astype(int).to_numpy()
    y_true_test = test_df[run_cfg.label_col].astype(int).to_numpy()

    univariate_df = run_univariate_probe(train_df=train_df, label_col=run_cfg.label_col)
    univariate_path = run_cfg.output_dir / "probe_univariate_results.csv"
    univariate_df.to_csv(univariate_path, index=False)

    topk = min(run_cfg.univariate_top_k, len(univariate_df))
    top_features = univariate_df.head(topk)["feature"].tolist()
    report_features = _pick_report_features(feature_cols=feature_cols, top_features=top_features)
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
    metrics_v0 = compute_metrics(y_true_test, y_pred_test_v0, y_score=y_pred_test_v0).values
    records.append(IterationRecord(version="v0", error_analysis="default_v0", metrics=metrics_v0))
    append_text(evolution_results_path, f"v0\t{metrics_v0}\n")
    trajectory_lines.append("V0: 初始化默认规则。")

    current_version = "v0"
    current_fn = fn_v0

    for i in range(1, run_cfg.iterations + 1):
        next_version = f"v{i}"
        y_pred_train = _predict_with_function(current_fn, train_df, run_cfg.label_col)
        allowed_degradation = max(run_cfg.degradation_threshold, int(len(train_df) * run_cfg.degradation_rate))
        train_old = compute_metrics(y_true_train, y_pred_train, y_score=y_pred_train).values
        primary = run_cfg.metric_priority[0] if run_cfg.metric_priority else "F1"
        samples = collect_errors(
            df=train_df,
            label_col=run_cfg.label_col,
            y_pred=y_pred_train,
            max_error_samples=run_cfg.max_error_samples,
            random_seed=run_cfg.random_seed + i,
            feature_cols=report_features,
        )
        error_report = format_error_report(samples, max_details=run_cfg.max_error_details)

        degradation_warning = (
            f"本轮约束：允许退化阈值={allowed_degradation}；"
            f"训练集当前指标={train_old}；"
            f"本轮优化优先级首要指标={primary}；"
            f"请避免 Specificity 下降超过 {run_cfg.max_specificity_drop:.3f}，"
            f"避免 ACC 下降超过 {run_cfg.max_acc_drop:.3f}。"
        )

        current_code = heuristic_path.read_text(encoding="utf-8")

        proposal: ParsedProposal | None = None
        accepted = False
        attempt = 0
        last_proposal: ParsedProposal | None = None
        while attempt < run_cfg.max_llm_attempts and not accepted:
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
                last_proposal = proposal
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
                y_true=y_true_train,
                y_pred_old=y_pred_old,
                y_pred_new=y_pred_new,
            )
            train_new = compute_metrics(y_true_train, y_pred_new, y_score=y_pred_new).values
            old_primary = float(train_old.get(primary, float("nan")))
            new_primary = float(train_new.get(primary, float("nan")))

            if np.isfinite(old_primary) and np.isfinite(new_primary) and new_primary + 1e-4 < old_primary:
                degradation_warning = (
                    f"训练集指标未提升：{primary} 从 {old_primary:.6f} 降至 {new_primary:.6f}。"
                    "请尽量最小修改以提高该指标，同时避免引入退化。"
                )
                continue

            old_spec = float(train_old.get("Specificity", float("nan")))
            new_spec = float(train_new.get("Specificity", float("nan")))
            if np.isfinite(old_spec) and np.isfinite(new_spec) and new_spec + 1e-6 < old_spec - run_cfg.max_specificity_drop:
                degradation_warning = (
                    f"训练集 Specificity 降幅过大：从 {old_spec:.6f} 降至 {new_spec:.6f}。"
                    f"请减少假阳性（FP），并保持降幅不超过 {run_cfg.max_specificity_drop:.3f}。"
                )
                continue

            old_acc = float(train_old.get("ACC", float("nan")))
            new_acc = float(train_new.get("ACC", float("nan")))
            if np.isfinite(old_acc) and np.isfinite(new_acc) and new_acc + 1e-6 < old_acc - run_cfg.max_acc_drop:
                degradation_warning = (
                    f"训练集 ACC 降幅过大：从 {old_acc:.6f} 降至 {new_acc:.6f}。"
                    f"请减少整体误差，并保持降幅不超过 {run_cfg.max_acc_drop:.3f}。"
                )
                continue

            if len(degr.degraded_indices) > allowed_degradation:
                examples = collect_degradation_examples(
                    df=train_df,
                    label_col=run_cfg.label_col,
                    degraded_indices=degr.degraded_indices,
                    y_pred_old=y_pred_old,
                    y_pred_new=y_pred_new,
                    feature_cols=report_features,
                    max_samples=run_cfg.degradation_max_examples,
                    random_seed=run_cfg.random_seed + 1000 + i + attempt,
                )
                degradation_warning = (
                    format_degradation_warning(degr.degraded_indices)
                    + f"\n允许退化阈值={allowed_degradation}\n"
                    + f"训练集旧指标={train_old}\n训练集新指标={train_new}\n"
                    + "退化样本示例（JSON）=\n"
                    + json.dumps(examples, ensure_ascii=False)
                )
                continue

            base_tests = tmp_ns.get("TESTS", [])
            combined_tests: list[dict] = []
            if isinstance(base_tests, list):
                combined_tests.extend([t for t in base_tests if isinstance(t, dict)])
            combined_tests.extend([t for t in (proposal.new_tests or []) if isinstance(t, dict)])
            ok, msg = _run_assert_tests(tmp_ns, combined_tests)
            if not ok:
                degradation_warning = f"回归测试失败：{msg}"
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
            m = compute_metrics(y_true_test, y_pred_test, y_score=y_pred_test).values
            records.append(IterationRecord(version=current_version, error_analysis=proposal.error_analysis, metrics=m))
            append_text(evolution_results_path, f"{current_version}\t{m}\n")

            iteration_log.append(
                {
                    "version": current_version,
                    "attempt": attempt,
                    "error_report": error_report,
                    "degradation_warning": degradation_warning,
                    "proposal": asdict(proposal),
                    "train_metrics_old": train_old,
                    "train_metrics_new": train_new,
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
                    "proposal": asdict(last_proposal) if last_proposal is not None else None,
                    "train_metrics_old": train_old,
                    "test_metrics": None,
                }
            )
            break

    write_json(iteration_log_path, iteration_log)

    if not records:
        raise RuntimeError("未生成任何版本记录。")

    def _score_f1(rec: IterationRecord) -> float:
        v = rec.metrics.get("F1", float("-inf"))
        try:
            return float(v)
        except Exception:
            return float("-inf")

    best = max(records, key=_score_f1)
    selected_version = best.version

    final_ns = _load_heuristic_module(heuristic_path)
    selected_fn = final_ns.get(f"predict_{selected_version}")
    if selected_fn is None:
        raise RuntimeError(f"选择版本函数缺失：predict_{selected_version}")

    export_code = heuristic_path.read_text(encoding="utf-8")
    export_code = export_code.rstrip() + "\n\n" + f"SELECTED_VERSION = '{selected_version}'\n\n" + (
        "def predict(features: dict) -> int:\n"
        f"    return predict_{selected_version}(features)\n"
    ) + "\n\n" + (
        "if __name__ == '__main__':\n"
        "    assert predict({}) in (0, 1)\n"
    )
    write_text(final_model_path, export_code + "\n")

    v0 = records[0]
    v_last = records[-1]
    write_text(
        final_comparison_path,
        f"V0={v0.metrics}\nLAST({v_last.version})={v_last.metrics}\nSELECTED_BY_TEST_F1({selected_version})={best.metrics}\n",
    )


def main() -> None:
    cfg = RunConfig(
        data_csv_path=Path("/data/yk/HL/data/YHD_bicarbonate.csv"),
        label_col="hospital_expire_flag",
        output_dir=Path("/data/yk/HL/out2"),
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
