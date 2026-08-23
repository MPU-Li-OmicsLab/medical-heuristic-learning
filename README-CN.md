![Medical Heuristic Learning](./supporting_files/medical-heuristic-learning-light.svg)

# Medical Heuristic Learning（MHL）

[![PyPI Version](https://img.shields.io/pypi/v/medical-heuristic-learning?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/medical-heuristic-learning/)
[![PyPI Python Version](https://img.shields.io/pypi/pyversions/medical-heuristic-learning?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/medical-heuristic-learning/)
[![CI](https://img.shields.io/github/actions/workflow/status/MPU-Li-OmicsLab/medical-heuristic-learning/ci.yml?branch=master&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/MPU-Li-OmicsLab/medical-heuristic-learning/actions/workflows/ci.yml)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.16337-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.16337)
[![Apache-2.0 License](https://img.shields.io/badge/License-Apache--2.0-green?style=for-the-badge)](./LICENSE)
[![LI-OMICSLAB](https://img.shields.io/badge/LI--OMICSLAB-00795E?style=for-the-badge)](https://liomicslab.cn/)

**医学启发式学习（Medical Heuristic Learning, MHL）是一种将大语言模型用作白盒规则生成器的医学表格数据预测范式，尤其适用于小样本、严重类别不平衡以及对模型可解释性和可审计性要求较高的场景。**

简体中文 · [English](./README.md) · [论文](https://arxiv.org/abs/2606.16337) · [API 文档](./docs/API-CN.md)

## 概述（Overview）

医学启发式学习（Medical Heuristic Learning, MHL） 是 [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) 范式在结构化医学数据预测中的一种实现。与神经网络把知识吸收到不可见参数中的训练方式不同，MHL 将统计证据、医学先验和验证反馈转化为版本化、可执行、可解释、可审计的纯 Python 决策规则。

**基本适用条件：**

- 医学表格数据分类预测任务；
- 能够通过 API 接入至少一种受支持的大语言模型，如 GPT、Claude 或 DeepSeek。

**优势应用场景：**

- 希望模型的决策过程清晰、可解释且可审计；
- 可用于训练的标注样本较少；
- 数据存在严重乃至极端的类别不平衡。

**MHL 拥有以下核心能力：**

- **白盒产物**：最终模型是确定性的 `predict(features: dict) -> int` 规则函数，而不是难以审计的参数权重。
- **双探针约束**：统计探针提供描述性统计和单变量关联；医学知识探针提供临床解释、候选阈值与证据置信度。
- **受控规则演化**：LLM 生成初始规则，并依据错误病例、退化病例、指标优先级和版本轨迹进行小步代码修订。
- **持续学习**：当特征被新增、删除或重命名时，系统继承上一阶段的探针与规则产物，在新证据下显式修订既有逻辑。
- **测试保障**：核心流程由 `pytest` 测试覆盖，并通过 GitHub Actions 持续验证。

论文全文与实验设计请参阅：[Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules](https://arxiv.org/abs/2606.16337)

## 快速开始

### 1. 使用 pip 安装

运行环境要求 Python 3.11 或更高版本。PyPI 发布后可直接安装：

```bash
python -m pip install medical-heuristic-learning
```

可从 GitHub 或本地源码安装：

```bash
python -m pip install "git+https://github.com/MPU-Li-OmicsLab/medical-heuristic-learning.git"

git clone https://github.com/MPU-Li-OmicsLab/medical-heuristic-learning.git
cd medical-heuristic-learning
python -m pip install -e .
```

默认 LLM 后端使用 OpenAI 兼容接口。建议配置环境变量 `DEEPSEEK_API_KEY` 以使用 DeepSeek API 密钥：

```bash
export DEEPSEEK_API_KEY="your-api-key"
```
更多兼容模型及对应的 API Key 也可在模型配置中更灵活指定，详见 [API 文档](./docs/API-CN.md)。

### 2. 运行 MHL

下面的最小示例使用合成医学表格数据，可直接保存为 `minimal_example.py` 运行。`train_df` 与 `val_df` 均须包含标签列，且移除标签后的特征集合必须一致。

```python
from pathlib import Path

import pandas as pd

from hl import LLMConfig, RunConfig, load_model, run_heuristic_learning

# Synthetic binary classification data for workflow demonstration only; not real clinical data.
data = pd.DataFrame(
    {
        "age": [35, 72, 44, 81, 53, 67, 29, 76, 48, 70, 39, 84],
        "heart_rate": [72, 118, 80, 126, 88, 110, 68, 121, 84, 115, 75, 130],
        "wbc": [6.1, 15.2, 7.4, 17.8, 9.0, 13.6, 5.8, 16.1, 8.2, 14.4, 6.9, 18.3],
        "hospital_expire_flag": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
)
train_df = data.iloc[:8].copy()
val_df = data.iloc[8:].copy()

result = run_heuristic_learning(
    train_df=train_df,
    val_df=val_df,
    label_col="hospital_expire_flag",
    run_cfg=RunConfig(
        output_dir=Path("./mhl_out"),
        iterations=1,
        task_description=(
            "Predict the risk of in-hospital mortality from the available "
            "clinical features."
        ),
    ),
    llm_cfg=LLMConfig(api_key_env="DEEPSEEK_API_KEY"),
)

print(result.final_model_path)

predict = load_model(result.final_model_path)
prediction = predict({"age": 74, "heart_rate": 119, "wbc": 15.0})
print(f"prediction={prediction}")
```

完整可运行示例：[example_training.py](./example_training.py)。

### 3. 重载模型并推理

导出的模型同时支持单行字典输入和 `DataFrame` 批量输入：

```python
from hl import load_batch_model, load_model

predict_one = load_model("./mhl_out/final_heuristic_model.py")
prediction = predict_one({"age": 68, "wbc": 13.2})

predict_batch = load_batch_model("./mhl_out/final_heuristic_model.py")
predictions = predict_batch(feature_dataframe)
```

输入只能包含模型特征；标签剔除和其他预处理由调用方负责。模型文件是可执行 Python 代码，因此只应加载可信来源的产物。完整示例：[example_inference.py](./example_inference.py)。

### 4. 持续学习

持续学习需要提供上一阶段输出目录，并显式描述新增、删除或重命名的特征：

```python
from pathlib import Path

from hl import (
    ContinuousLearningConfig,
    DriftConfig,
    LLMConfig,
    run_continuous_learning,
)

result = run_continuous_learning(
    train_df=new_train_df,
    val_df=new_val_df,
    label_col="hospital_expire_flag",
    llm_cfg=LLMConfig(api_key_env="DEEPSEEK_API_KEY"),
    continuous_cfg=ContinuousLearningConfig(
        output_dir=Path("./mhl_out_continual"),
        task_description=(
            "Adapt the existing clinical rule system to the updated feature schema "
            "while preserving in-hospital mortality prediction performance."
        ),
        drift=DriftConfig(
            dropped_cols=("wbc",),
            added_cols=("new_marker",),
            renamed_cols=(("old_name", "new_name"),),
            change_note=(
                "The wbc feature is no longer available, new_marker has been added, "
                "and old_name has been renamed to new_name."
            ),
            prev_hl_out_dir=Path("./mhl_out"),
        ),
    ),
)
```

完整可运行示例：[example_continuous_learning.py](./example_continuous_learning.py)。

### 5. 运行产物

标准流程默认写入 `./out/{时间戳}/`；持续学习默认写入 `./out/{时间戳}_continuous_learning/`。

| 产物                             | 说明                                        |
| ------------------------------ | ----------------------------------------- |
| `probe_univariate_results.csv` | 统计探针结果与特征排序。                              |
| `probe_knowledge.md`           | 医学知识探针生成的结构化知识表。                          |
| `heuristic_system.py`          | `predict_v0`、`predict_v1` 等全部版本化规则。       |
| `evolution_results.txt`        | 各版本在验证集上的指标轨迹。                            |
| `iteration_log.json`           | 每轮提案、校验、接受状态与退化样本记录。                      |
| **`final_heuristic_model.py`** | 核心的最终规则文件，包含最佳规则版本及稳定的 `predict(...)` 入口。 |
| `final_comparison.txt`         | 初始版本、最佳版本与最后版本的指标对比。                      |

持续学习还会生成 `continuous_learning_context.json`、`probe_univariate_results_prev.csv` 和 `probe_knowledge_prev.md`，用于记录漂移上下文与上一阶段探针快照。

## API 文档

- [中文 API 文档](./docs/API-CN.md)：按照当前 `src/hl` 代码整理的公共入口、配置字段、返回类型、异常与产物契约。
- [英文 API 文档](./docs/API.md)：与中文版对应的英文公共接口文档。

## 流程设计

![MHL 标准流程与持续学习流程](./supporting_files/fig1.jpg)

标准 MHL 由四个步骤组成：

1. **统计探针**：从训练集提取描述性统计、缺失率和单变量关联，为规则生成提供低假设的经验依据。
2. **医学知识探针**：结合特征与任务语义，由 LLM 归纳临床解释、候选阈值和证据置信度。
3. **初始规则生成**：融合双探针结果、任务描述和指标优先级，生成经过结构、语法和函数名校验的 `predict_v0`。
4. **规则迭代**：执行当前规则，分析错误与版本退化，要求 LLM 进行小步修订；候选规则通过校验和评估后才进入版本历史，最终按指标优先级导出最佳版本。

**持续学习**沿用相同的四步闭环，但以先前验证过的探针结果和最终规则作为显式先验。特征空间变化后，系统过滤已删除特征、处理重命名特征、为新增特征补充证据，并生成漂移感知的新 `v0`，随后继续迭代。整个适应过程表现为可见的代码修订，而不是对隐藏参数的覆盖。

## 实验发现

实验围绕样本规模、类别比例、探针消融、LLM 后端和特征演化展开。各实验子目录对应的研究问题如下：

| 实验目录 | 实验内容 | 核心比较 |
| --- | --- | --- |
| [`experiment/contrast0/`](./experiment/contrast0/README.md) | LLM 后端对比实验 | 在相同数据划分和 MHL 流程下，对比不同 LLM 及推理强度生成白盒规则的效果。 |
| [`experiment/contrast1/`](./experiment/contrast1/README.md) | 训练集规模实验 | 在平衡训练集上改变样本量，比较 MHL 与机器学习、深度学习及可解释模型的小样本表现。 |
| [`experiment/contrast2/`](./experiment/contrast2/README.md) | 类别不平衡实验 | 固定训练总量并改变正负样本比例，评估各方法在极端类别失衡下的敏感度与特异度。 |
| [`experiment/ablation/`](./experiment/ablation/README.md) | 探针消融实验 | 组合启用或关闭统计探针与医学知识探针，分析两类信息对规则生成和迭代的贡献。 |
| [`experiment/continuous_learning/`](./experiment/continuous_learning/README.md) | 持续学习实验 | 在 SIRS 被移除、SOFA 被加入且第二阶段样本有限的条件下，比较继承既有状态与从头训练。 |

`experiment/modeling/` 提供多个实验共用的基线模型、预处理和评估实现，`experiment/outputs_rerun/` 用于保存实验产物；二者不属于独立实验。

主要观察如下：

- **总体性能**：MHL 在多个医学表格数据集上取得了与代表性强基线相当的预测性能，同时保留了完整、可执行的决策逻辑。
- **小样本鲁棒性**：在标注样本有限时，医学先验和显式规则结构减少了对大规模参数估计的依赖，使 MHL 保持有竞争力的表现。
- **类别不平衡适应性**：在极端正负比例下，常规模型容易出现近单类别预测；MHL 可借助指标优先级、错误分析和退化反馈显式调整敏感度与特异度之间的权衡。
- **探针互补性**：统计证据与医学知识分别约束数据相关性和临床合理性；消融结果支持二者联合使用，以获得更稳定的规则生成与迭代过程。
- **持续学习能力**：在特征演化与少量新阶段数据下，继承并修订旧规则有助于保留已验证知识，并缓解从头训练或不透明参数覆盖带来的遗忘。
- **跨后端可迁移性**：不同 LLM 后端均可接入同一受约束工作流；最终部署对象仍是确定性的纯 Python 规则，而不是 LLM 本身。

详见 [arXiv 论文](https://arxiv.org/abs/2606.16337)。

## 仓库结构

```text
medical-heuristic-learning/
├── src/hl/
│   ├── agent/                  # OpenAI 兼容客户端与提示模板
│   ├── continuous_learning/    # 特征演化下的持续学习流程
│   ├── evolution/              # 错误分析、退化检测与规则校验
│   ├── orchestrator/           # 标准 MHL 四阶段编排
│   ├── probes/                 # 统计探针与医学知识探针
│   ├── config.py               # LLMConfig 与 RunConfig
│   ├── metrics.py              # 分类指标
│   ├── model.py                # 单行与批量模型加载
│   └── result.py               # 运行产物路径类型
├── docs/                       # API 文档
├── tests/                      # pytest 测试套件
├── experiment/                 # 对比、消融与持续学习实验
├── supporting_files/           # README 题图与流程图
├── example_training.py         # 训练示例
├── example_inference.py        # 模型重载与推理示例
├── example_continuous_learning.py
├── pyproject.toml
└── README-CN.md
```

## TODO

- [ ] 提供兼容 scikit-learn 的 Estimator 接口，包括 `fit`、`predict`、`get_params` 与 `set_params`。
