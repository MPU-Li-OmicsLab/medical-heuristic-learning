# 运行指南（HL Heuristic Learning）

## 1. 前置条件

- Python：按仓库的 `.python-version`（3.11）；`pyproject.toml` 要求 `>=3.11,<3.14`（上限来自 DeepTab 2.0.0）
- 依赖：使用 `uv` 安装（在仓库根目录执行）
- 数据集：默认读取 `./data/YHD_bicarbonate.csv`
- 标签列：`hospital_expire_flag`
- 大模型（可选）：DeepSeek OpenAI 兼容接口，Key 通过环境变量注入

环境变量：

- `DEEPSEEK_API_KEY`：DeepSeek key（启用 LLM 时必填）

## 2. 安装与运行

在仓库根目录执行：

```bash
uv sync
```

`uv` 默认就会安装 `dev` 组；完整对比实验需要的 deeptab、interpret-core[aplr]、
aplr、corels 等包都在 `pyproject.toml` 的 `[dependency-groups].dev` 中。只安装
基础依赖时使用：

```bash
uv sync --no-dev
```

CORELS 1.1.29 是官方源码包，需要 C++ 编译器。若全新 Python 3.11 / NumPy 2
环境因包内旧生成代码构建失败，应使用 dev 组中的 Cython 3 从官方
`corels/_corels.pyx` 重新生成 `_corels.cpp` 后构建 wheel；不要改用非官方 fork。
本仓库当前 `.venv` 已完成该步骤并通过 toy-data 拟合测试。

运行示例入口（推荐）：

```bash
export DEEPSEEK_API_KEY="你的key"
uv run python example_training.py
```

推理示例（需要先运行 `example_training.py` 生成
`./example_out/final_heuristic_model.py`）：

```bash
uv run python example_inference.py
```

持续学习示例（同样需要先运行 `example_training.py`；删除 `wbc` 特征模拟新环境
漂移，输出到 `./example_out_continuous_learning`）：

```bash
uv run python example_continuous_learning.py
```

说明：

- `example_training.py` 当前的数据划分为：
  - 训练集：`data.iloc[:500]`
  - 验证集：`data.iloc[500:1000]`
- 若不配置输出目录，默认输出到 `./out/<时间戳>/`（标准流程）或
  `./out/<时间戳>_continuous_learning/`（持续学习流程）。
- 输出目录与各阶段开关直接在 `example_training.py` 中通过 `RunConfig` 赋值配置
  （示例里已经指向 `./example_out`）。

## 3. 输出文件说明（在 output_dir 下）

- `probe_univariate_results.csv`：单变量统计探针结果（特征相关性/显著性/缺失率等）
- `probe_knowledge.md`：医学知识探针输出（LLM 生成的阈值与解释，若启用）
- `heuristic_system.py`：规则系统源码（包含 `predict_v0/predict_v1/...`）
- `evolution_results.txt`：每个版本在验证集上的指标（ACC/F1/Sensitivity/Specificity）
- `iteration_log.json`：每轮迭代详细日志（错误样本、退化检测、提案与验收原因）
- `final_heuristic_model.py`：最终导出的统一入口 `predict(features)->int`
- `final_comparison.txt`：V0、FINAL、LAST 的对比汇总

持续学习流程还会额外产出：

- `continuous_learning_context.json`
- `probe_univariate_results_prev.csv`
- `probe_knowledge_prev.md`

## 4. Baseline/探针/V0/迭代 的开关（RunConfig）

核心编排在 `hl/orchestrator/main_orchestrator.py: run_heuristic_learning(...)`，
由 `hl/config.py: RunConfig` 控制（持续学习对应
`hl/continuous_learning/main_orchestrator.py` 与 `ContinuousLearningConfig`）。

默认行为（不额外传参）：

- baseline：不跑（`train_baselines=False`，预留字段，主编排器未使用）
- 探针：两种都跑（`run_univariate_probe=True`，`run_knowledge_probe=True`）
- v0 生成：跑（`run_v0_generation=True`）
- 迭代：跑（`run_iterations=True`）

你可以在 `example_training.py` 中改为显式配置，例如：

```python
from hl.config import RunConfig

run_cfg = RunConfig(
    output_dir=output_dir,
    run_univariate_probe=True,
    run_knowledge_probe=True,
    run_v0_generation=True,
    run_iterations=True,
)
```

### 4.2 探针已跑过：不再跑探针，只继续迭代

前提：output_dir 里已有 `probe_univariate_results.csv` / `probe_knowledge.md`（会直接复用读取）。

```python
run_cfg = RunConfig(
    output_dir=output_dir,
    run_univariate_probe=False,
    run_knowledge_probe=False,
    run_iterations=True,
)
```

### 4.3 只跑迭代（完全跳过探针 + 跳过 v0 生成）

前提：output_dir 里已有 `heuristic_system.py`（否则会报错）。

```python
run_cfg = RunConfig(
    output_dir=output_dir,
    run_univariate_probe=False,
    run_knowledge_probe=False,
    run_v0_generation=False,
    run_iterations=True,
)
```

### 4.4 只跑探针（不生成 v0、不迭代）

注意：当前主编排器在 Step 4 总会先加载并评估 `heuristic_system.py`，因此该
模式要求 output_dir 里已经有 `heuristic_system.py`，否则会以
“predict_v0 not found in heuristic_system.py” 报错。若只想得到 probe 文件，
建议正常跑一次完整流程后，再单独重跑探针步骤。

```python
run_cfg = RunConfig(
    output_dir=output_dir,
    run_univariate_probe=True,
    run_knowledge_probe=True,
    run_v0_generation=False,
    run_iterations=False,
)
```

## 5. 关于复用输出目录 / 继续迭代

当前实现的实际行为：

- `run_univariate_probe=False` / `run_knowledge_probe=False` 时会复用 output_dir
  中已有的 probe 缓存文件；
- `heuristic_system.py` 已存在时，v0 阶段会直接复用现有文件；
- 迭代阶段每次都从 `predict_v0` 重新评估开始，按 `predict_v1、predict_v2、...`
  从头追加新版本。它不会读取 `CURRENT_VERSION` 接着上次的版本号续跑，因此对同一
  output_dir 重复跑迭代会在 `heuristic_system.py` 里追加重复版本，并在
  `evolution_results.txt` 里重复写 v0 行。

建议：

- 每次完整实验使用不同的 output_dir，避免混淆；
- 如果只是想“接着迭代”，直接显式指向已有 output_dir 并复用
  probe/`heuristic_system.py`，但要了解上面重复追加的行为（当前没有基于
  `CURRENT_VERSION` 的断点续跑）；
- 如果关闭了 v0 生成（`run_v0_generation=False`）但 output_dir 没有指向已有
  `heuristic_system.py` 的目录，会直接报错，避免误创建新时间戳目录。

## 6. DeepSeek 模型选择（LLMConfig）

在 `hl/config.py: LLMConfig` 中配置：

- `base_url`：`https://api.deepseek.com/v1`（OpenAI 兼容接口路径前缀）
- `model_name`：例如 `deepseek-v4-pro`
- `thinking_mode`：DeepSeek 思考模式开关，默认 `None`（不干预，
  按后端官方默认；DeepSeek 当前默认开启思考、effort 为 `high`）
- `thinking_strength`：思考强度，思考开启时作为 `reasoning_effort` 发送；
  取值 `low` / `medium` / `high` / `xhigh` / `max`，缺省使用
  DeepSeek 服务端默认值（`high`）

如果你想临时改模型，最直接方式是在 `example_training.py` 里：

```python
from hl.config import LLMConfig

llm_cfg = LLMConfig(model_name="deepseek-v4-pro")
```

需要开启思考模式的实验配置示例：

```python
llm_cfg = LLMConfig(
    model_name="deepseek-v4-pro",
    thinking_mode=True,
    thinking_strength="high",
)
```

说明：`thinking_mode=None`（默认）时不发送任何 `thinking` 参数，完全交给
后端官方默认（DeepSeek 当前默认开启思考、effort 为 `high`）；`True`/`False`
才显式发送 `enabled`/`disabled`。单独设置 `thinking_strength` 而未显式指定
`thinking_mode` 时，等价于开启思考并指定强度。若 `extra_body` 已自带
`thinking` 键，则以 `extra_body` 为准，保证旧配置完全兼容。

## 7. 对比实验命令（简要）

当前扩展实验统一使用种子 `36 40 42`，并支持逐任务断点续跑：

```bash
uv run python experiment/contrast1/run_contrast1_balance.py --models all --seeds 36 40 42 --resume
uv run python experiment/contrast2/run_contrast2.py --models all --seeds 36 40 42 --resume
uv run python experiment/continuous_learning/run_continuous_learning_baselines_v2.py --models all --seeds 36 40 42 --resume
uv run python experiment/continuous_learning/run_continuous_learning_hl_v2.py --seeds 36 40 42 --resume
uv run python experiment/continuous_learning/verify_continuous_baselines_v2.py
```

十个普通模型都会重跑；HL 结果尽量复用（持续学习 v2 中 Stage 1 / continual 的
HL 行来自既有实验，仅新跑 direct Stage 2 HL）。详细的数据规模、关闭自动平衡
配置、持续学习三分支设计和输出约定见 `experiment/EXPERIMENT_EXTENSION_PLAN.md`、
`experiment/RERUN_TRAINING_REPORT.md` 与各实验目录下的 README。
