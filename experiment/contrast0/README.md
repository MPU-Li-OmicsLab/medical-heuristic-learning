# contrast0：启发式学习对比（不同大模型）

本目录用于比较不同大模型作为启发式学习后端时，对最终分类结果的影响。实验固定使用同一套启发式学习流程，仅替换 LLM 配置。

当前目录下的主脚本是：

- `run_contrast0.py`

## 实验目标

在两个数据集上，固定：

- 启发式学习流程
- 探针设置
- 数据切分方式
- 训练集规模与类别比例

只改变底层使用的大模型，比较最终 held-out test 指标的差异。

## 数据与切分

### 数据集

- `UKB`：`./data/UKB.csv`，标签列 `label`
- `YHD`：`./data/YHD_bicarbonate.csv`，标签列 `hospital_expire_flag`

要求：

- 标签必须是二分类 `0/1`
- 正负样本数量必须足够支撑平衡的 `val/test` 抽样

### 默认随机种子

- `seed=42`

### 切分规则

- `val`：1000 条，按 `1:1` 抽样，即 `500` 正 + `500` 负
- `test`：1000 条，按 `1:1` 抽样，即 `500` 正 + `500` 负
- `train_pool`：除去 `val/test` 后剩余的样本

### 训练集规则

- 训练集总数固定为 `1000`
- 训练集按 `1:1` 平衡采样，即目标为 `500` 正 + `500` 负
- 若某一类样本不足，会对该类启用有放回采样以补足目标数量

## 启发式学习设置

每个实验都调用项目主流程 `run_heuristic_learning(...)`，固定配置为：

- `run_univariate_probe=True`
- `run_knowledge_probe=True`
- `run_v0_generation=True`
- `run_iterations=True`

也就是说，这里比较的是完整的 `U1_K1` 启发式学习流程，而不是消融实验。<mccoremem id="project_memory" />

训练方式为：

- 将 `val_df` 传入 `run_heuristic_learning(..., test_df=val_df, ...)`
- 用验证集结果选择最终导出的 `final_heuristic_model.py`
- 训练结束后，再用该最终模型在 held-out `test_df` 上重新评估
- 把 held-out test 指标写入汇总 CSV 和实验目录下的摘要文件

此外，脚本会把 `seed` 传入 `RunConfig.random_seed`，用于增强流程可复现性。

## 对比的大模型

脚本当前内置 6 个模型配置：

- `deepseek-v4-pro`
- `deepseek-v4-pro-thinking`
- `deepseek-v4-flash`
- `qwen/qwen3.7-max`
- `gemini-3.1-pro-preview`
- `gpt-5.5`

这些模型会分别在两个数据集上运行，因此默认一共执行：

- `6 个模型 × 2 个数据集 = 12` 个实验

### 各模型来源

- DeepSeek：
  - `deepseek-v4-pro`
  - `deepseek-v4-pro-thinking`
  - `deepseek-v4-flash`
- OpenRouter：
  - `qwen/qwen3.7-max`
- vveai：
  - `gemini-3.1-pro-preview`
  - `gpt-5.5`

### DeepSeek 思考模式

`deepseek-v4-pro-thinking` 实际仍调用模型名 `deepseek-v4-pro`，区别在于额外传入：

```json
{"thinking": {"type": "enabled"}}
```

相应地：

- `deepseek-v4-pro` 会显式传入 `{"thinking": {"type": "disabled"}}`
- `deepseek-v4-flash` 也会显式传入 `{"thinking": {"type": "disabled"}}`

## API Key 与 Base URL

### 默认环境变量

- DeepSeek：`DEEPSEEK_API_KEY`
- OpenRouter：`OPENROUTER_API_KEY`
- vveai Gemini：`VVEAI_GEMINI_API_KEY`
- vveai GPT-5.5：`VVEAI_GPT55_API_KEY`

### 默认 Base URL

- DeepSeek：`https://api.deepseek.com/v1`
- OpenRouter：`https://openrouter.ai/api/v1`
- vveai：`https://api.vveai.com/v1`

注意：

- 脚本内部会先读取 `https://api.vveai.com`，再自动规范化为以 `/v1` 结尾的地址

### 可覆盖的环境变量

如需自定义接口地址或 key 变量名，可使用以下覆盖项：

- `CONTRAST0_DEEPSEEK_BASE_URL`
- `CONTRAST0_DEEPSEEK_KEY_ENV`
- `CONTRAST0_ROUTER_BASE_URL`
- `CONTRAST0_ROUTER_KEY_ENV`
- `CONTRAST0_VVEAI_BASE_URL`
- `CONTRAST0_VVEAI_GEMINI_KEY_ENV`
- `CONTRAST0_VVEAI_GPT55_KEY_ENV`

## 指标

汇总表中输出以下 held-out test 指标：

- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`

## 运行方式

请在仓库根目录运行，例如：

```bash
export DEEPSEEK_API_KEY="你的 deepseek key"
export OPENROUTER_API_KEY="你的 openrouter key"
export VVEAI_GEMINI_API_KEY="你的 gemini key"
export VVEAI_GPT55_API_KEY="你的 gpt-5.5 key"

uv run python experiment/contrast0/run_contrast0.py --seed 42 --workers 1
```

参数说明：

- `--seed`：控制 `val/test/train` 抽样以及主流程随机种子
- `--workers`：多进程并发数
- `--output-root`：实验输出根目录，默认位于 `./experiment/contrast0/outputs`

说明：

- 当 `workers > 1` 时，会按 `(模型, 数据集)` 粒度并行执行
- 并发过高可能触发 API 限流，也会更快消耗余额
- 对 LLM 类实验，建议先从 `1` 开始测试，再逐步提升

## 输出说明

### 汇总表

脚本运行结束后会写出：

- `experiment/contrast0/contrast0.csv`

该文件当前列为：

- `大模型`
- `数据集`
- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`

排序顺序为：

- 先按模型在脚本中的定义顺序
- 再按数据集顺序 `UKB -> YHD`

需要注意：

- 汇总表不包含 `status`、`error`、`out_dir` 等字段
- 如果某个实验失败，脚本仍会保留这一行，但对应指标会是空字符串

### 单个实验目录

每个实验会输出到：

- `experiment/contrast0/outputs/<DATASET>/<MODEL>/<TIMESTAMP>/`

如果使用 `--output-root`，则路径变为：

- `<output-root>/<DATASET>/<MODEL>/<TIMESTAMP>/`

其中目录名里的 `<MODEL>` 会把 `/` 替换成 `_`，例如：

- `qwen/qwen3.7-max` 会写成 `qwen_qwen3.7-max`

示例：

- `experiment/contrast0/outputs/UKB/deepseek-v4-pro/20260525_123456/`
- `experiment/contrast0/outputs/YHD/qwen_qwen3.7-max/20260525_123501/`

单个实验目录通常包含：

- `heuristic_system.py`
- `final_heuristic_model.py`
- `final_comparison.txt`
- `evolution_results.txt`
- `iteration_log.json`
- `heldout_test_summary.json`
- `heldout_test_summary.txt`

其中：

- `heldout_test_summary.json` 会记录数据集、模型显示名、实际模型名、`base_url`、`api_key_env`、训练采样信息、最终版本号与 held-out test 指标
- `heldout_test_summary.txt` 是便于快速浏览的文本摘要

## 其他说明

- 脚本会在运行时自动回到仓库根目录加入 `sys.path`，因此可以直接使用 `python experiment/contrast0/run_contrast0.py` 这种按路径执行的方式
- 与 `ablation` 不同，这里没有额外生成 `index.json` 一类的批次级索引文件
