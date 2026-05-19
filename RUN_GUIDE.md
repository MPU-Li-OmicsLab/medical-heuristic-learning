# 运行指南（HL Heuristic Learning）

## 1. 前置条件

- Python：按仓库的 `.python-version`（建议用 uv 管理）
- 依赖：使用 `uv` 安装（项目根目录 `/data/yk/HL`）
- 数据集：默认读取 `/data/yk/HL/data/YHD_bicarbonate.csv`
- 标签列：`hospital_expire_flag`
- 大模型（可选）：DeepSeek OpenAI 兼容接口，Key 通过环境变量注入

环境变量：

- `DEEPSEEK_API_KEY`：DeepSeek key（必填，若启用 LLM）
- `HL_OUTPUT_DIR`：输出目录（可选；若不设且不传 `--output-dir`，默认输出到 `./out/<时间戳>/`）

## 2. 安装与运行

在 `/data/yk/HL` 目录下：

```bash
uv sync
```

运行示例入口（推荐）：

```bash
export DEEPSEEK_API_KEY="你的key"
uv run python example.py
```

说明：

- `example.py` 当前的数据划分为：
  - 训练集：`data.iloc[:500]`
  - 测试集：`data.iloc[500:1000]`
- 若不配置输出目录，默认输出到 `./out/<时间戳>/`（在 `/data/yk/HL/out/<时间戳>/`）。
- 若配置了 `HL_OUTPUT_DIR` 或 `--output-dir`，则输出到指定目录。

也可以显式指定输出目录（推荐在复用/继续迭代时使用）：

```bash
export DEEPSEEK_API_KEY="你的key"
uv run python example.py --output-dir /data/yk/HL/out_example_run1
```

## 3. 输出文件说明（在 output_dir 下）

- `probe_univariate_results.csv`：单变量统计探针结果（特征相关性/显著性/缺失率等）
- `probe_knowledge.md`：医学知识探针输出（LLM 生成的阈值与解释，若启用）
- `heuristic_system.py`：规则系统源码（包含 `predict_v0/predict_v1/...` 与 tests）
- `evolution_results.txt`：每个版本在测试集上的指标（ACC/F1/AUC/Sensitivity/Specificity）
- `iteration_log.json`：每轮迭代详细日志（错误样本、退化检测、提案与验收原因）
- `final_heuristic_model.py`：最终导出的统一入口 `predict(features)->0/1`
- `final_comparison.txt`：V0、FINAL、LAST 与 baseline（若启用）的对比汇总

如果启用了 baseline：

- `baseline_results.json`：baseline 指标
- `baseline_lr.pkl`：逻辑回归模型
- `baseline_dt.pkl`：决策树模型

## 4. Baseline/探针/V0/迭代 的开关（RunConfig）

核心编排在 `run.py: run_heuristic_learning(...)`，由 `config.py: RunConfig` 控制。

默认行为（不额外传参）：

- baseline：不跑（`train_baselines=False`）
- 探针：两种都跑（`run_univariate_probe=True`，`run_knowledge_probe=True`）
- v0 生成：跑（`run_v0_generation=True`）
- 迭代：跑（`run_iterations=True`）

在 `example.py` 入口也提供了等价的命令行开关：

- `--output-dir <path>`：指定输出目录
- `--train-baselines`：启用 baseline
- `--skip-univariate`：跳过单变量探针（复用已有 `probe_univariate_results.csv`，若存在）
- `--skip-knowledge`：跳过知识探针（复用已有 `probe_knowledge.md`，若存在）
- `--skip-v0`：跳过 v0 生成（要求已有 `heuristic_system.py`）
- `--skip-iterations`：跳过迭代

你可以在 `example.py` 中改为显式配置，例如：

```python
from config import RunConfig

run_cfg = RunConfig(
    output_dir=output_dir,
    train_baselines=False,
    run_univariate_probe=True,
    run_knowledge_probe=True,
    run_v0_generation=True,
    run_iterations=True,
)
```

### 4.1 只做对比实验时才跑 baseline

```python
run_cfg = RunConfig(output_dir=output_dir, train_baselines=True)
```

命令行方式：

```bash
uv run python example.py --output-dir /data/yk/HL/out_example_run1 --train-baselines
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

命令行方式（必须指定已有 output_dir）：

```bash
uv run python example.py --output-dir /data/yk/HL/out_example_run1 --skip-univariate --skip-knowledge
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

命令行方式（必须指定已有 output_dir）：

```bash
uv run python example.py --output-dir /data/yk/HL/out_example_run1 --skip-univariate --skip-knowledge --skip-v0
```

### 4.4 只跑探针（不生成 v0、不迭代）

```python
run_cfg = RunConfig(
    output_dir=output_dir,
    run_univariate_probe=True,
    run_knowledge_probe=True,
    run_v0_generation=False,
    run_iterations=False,
)
```

命令行方式：

```bash
uv run python example.py --output-dir /data/yk/HL/out_example_run1 --skip-v0 --skip-iterations
```

## 5. 继续迭代（从已有版本往后跑）

当 output_dir 中已经存在 `heuristic_system.py` 时：

- 程序会读取 `CURRENT_VERSION`（例如 `v2`）
- 后续迭代会从 `v3` 开始命名并追加到同一个 `heuristic_system.py`
- `evolution_results.txt` 若已存在，会复用已有行，避免重复写入 `v0`

建议每次实验使用不同输出目录，避免混淆：

```bash
export HL_OUTPUT_DIR="/data/yk/HL/out_run_$(date +%Y%m%d_%H%M%S)"
uv run python example.py
```

注意：

- 只有当你显式指定 `--output-dir` 或 `HL_OUTPUT_DIR` 指向“已有输出目录”时，才可能复用之前的探针/版本并继续迭代。
- 如果你关闭了任意阶段（例如 `--skip-univariate`），但没有指定输出目录，程序会报错，避免误创建新时间戳目录导致“无法获取之前的部分”。

## 6. DeepSeek 模型选择（LLMConfig）

在 `config.py: LLMConfig` 中配置：

- `base_url`：`https://api.deepseek.com/v1`（OpenAI 兼容接口路径前缀）
- `model_name`：例如 `deepseek-v4-pro`

如果你想临时改模型，最直接方式是在 `example.py` 里：

```python
from config import LLMConfig

llm_cfg = LLMConfig(model_name="deepseek-v4-pro")
```
