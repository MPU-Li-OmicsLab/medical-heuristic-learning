from __future__ import annotations

import json


def get_knowledge_probe_prompt(features: list[str], target: str) -> str:
    return (
        "你是一个医学知识注入助手。请为给定的医学表格特征生成临床先验知识。\n"
        f"结局/目标变量：{target}\n"
        "你必须返回一个 Markdown 表格，且必须包含以下列：\n"
        "| 特征名 | 单变量强度（摘要） | 临床关联描述 | 建议阈值 | 证据置信度（高/中/低） |\n"
        "要求：\n"
        "- 建议阈值必须给出（若不适用请写“无明确阈值”并说明理由）\n"
        "- 临床关联描述要尽量具体，可用于解释规则\n"
        "- 证据置信度给出高/中/低\n"
        "特征列表如下（JSON）：\n"
        + json.dumps(features, ensure_ascii=False)
    )


def get_rule_generation_prompt(univariate_summary: str, knowledge_table: str, metric_desc: str) -> str:
    return (
        "你是一个医学规则学习智能体。你将生成一个纯 Python 分类规则函数。\n"
        "输入包含：单变量统计摘要、医学知识表、指标优化优先级说明。\n"
        f"{metric_desc}\n\n"
        "[单变量统计摘要]\n"
        f"{univariate_summary}\n\n"
        "[医学知识表]\n"
        f"{knowledge_table}\n\n"
        "请输出严格 JSON，包含字段：\n"
        '- version: "v0"\n'
        "- error_analysis: 本次规则的设计理由（中文）\n"
        "- new_policy_code: 完整的 Python 函数定义，函数名必须为 predict_v0\n"
        "- new_tests: 回归测试列表，每个为 {name, code}，code 为 assert 语句\n"
        "- modified_tests: 需要修改的旧测试列表（v0 为空数组）\n"
        "规则函数签名必须是：def predict_v0(features: dict) -> int:\n"
        "规则必须自包含，只能使用 Python 标准库，不得依赖第三方包。\n"
    )


def get_iteration_prompt(
    current_code: str,
    error_report: str,
    trajectory: str,
    degradation_warning: str,
    metric_desc: str,
    next_version: str,
) -> str:
    return (
        "你是一个医学规则学习智能体。根据下面的信息修改当前分类规则：\n"
        "- 当前完整代码（含所有历史版本）\n"
        "- 本次训练集错误样本分析\n"
        "- 历史迭代轨迹（过去各版本的修改原因）\n"
        "- 指标优化优先级说明\n"
        "- （若有）退化警告：上一版修改导致以下样本退化，请修复\n\n"
        f"{metric_desc}\n\n"
        "[当前完整代码]\n"
        f"{current_code}\n\n"
        "[本次训练集错误样本分析]\n"
        f"{error_report}\n\n"
        "[历史迭代轨迹]\n"
        f"{trajectory}\n\n"
        "[退化警告]\n"
        f"{degradation_warning}\n\n"
        "你必须返回严格 JSON：\n"
        "{\n"
        f'  "version": "{next_version}",\n'
        '  "error_analysis": "...(中文)...",\n'
        '  "new_policy_code": "def predict_...\\n ...",\n'
        '  "new_tests": [{"name": "...", "code": "assert ..."}],\n'
        '  "modified_tests": [{"name": "...", "old_code": "assert ...", "new_code": "assert ..."}]\n'
        "}\n"
        "修改必须最小化，且保持注释清晰。\n"
        "规则必须自包含，只能使用 Python 标准库，不得依赖第三方包。\n"
    )

