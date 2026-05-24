# contrast0：启发式学习对比（不同大模型）

## 实验目标

在两个数据集（UKB、YHD）上，固定启发式学习流程与探针设置（U1_K1），仅替换大模型配置，对比不同大模型带来的最终分类效果差异。

## 数据与切分

- 数据集
  - UKB：`./data/UKB.csv`，标签列 `label`
  - YHD：`./data/YHD_bicarbonate.csv`，标签列 `hospital_expire_flag`
- 随机种子：默认 `seed=42`
- 验证集/测试集
  - `val`：1000 条，按 1:1 抽样（500 正 + 500 负）
  - `test`：1000 条，按 1:1 抽样（500 正 + 500 负）
- 训练集
  - 总数 1000，按 1:1 抽样（500 正 + 500 负）
  - 若某类样本不足，会在训练集对该类 **with replacement 重采样**，以补足目标数量

## 启发式学习设置

每个实验都会执行与主体训练相同的启发式学习流程（生成/迭代规则），并固定两个探针都开启：

- 单变量探针：开启
- 知识探针：开启

训练时将 `val` 作为 `run_heuristic_learning(..., test_df=val_df)` 的输入，以在验证集上选择 best 并导出 `final_heuristic_model.py`，然后在 held-out `test` 上计算指标写入 `contrast0.csv`。

## 对比的大模型列表

脚本内置 6 种模型配置：

- `deepseek-v4-pro`（非思考，DeepSeek）
- `deepseek-v4-pro-thinking`（思考模式，DeepSeek）
- `deepseek-v4-flash`（DeepSeek）
- `qwen/qwen3.7-max`（OpenRouter）
- `gemini-3.1-pro-preview`（vveai）
- `gpt-5.5`（vveai）

### DeepSeek 思考模式

DeepSeek 的思考模式通过 OpenAI 兼容请求的 `extra_body={"thinking":{"type":"enabled"}}` 打开。参见官方说明：[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)。

### Base URL 与 Key 环境变量

- DeepSeek：`DEEPSEEK_API_KEY`
- OpenRouter（qwen）：`OPENROUTER_API_KEY`
- vveai（Gemini/GPT）：`VVEAI_GEMINI_API_KEY`、`VVEAI_GPT55_API_KEY`

如需自定义，可使用脚本读取的覆盖项：

- `CONTRAST0_DEEPSEEK_BASE_URL` / `CONTRAST0_DEEPSEEK_KEY_ENV`
- `CONTRAST0_ROUTER_BASE_URL` / `CONTRAST0_ROUTER_KEY_ENV`
- `CONTRAST0_VVEAI_BASE_URL` / `CONTRAST0_VVEAI_GEMINI_KEY_ENV` / `CONTRAST0_VVEAI_GPT55_KEY_ENV`

## 指标

输出指标：

- ACC
- F1
- Sensitivity（TPR）
- Specificity（TNR）

## 运行方式

从仓库根目录运行：

```bash
cd /home/yk/medical-heuristic-learning

export DEEPSEEK_API_KEY="你的deepseek-key"
export OPENROUTER_API_KEY="你的openrouter-key"  # 仅用于 qwen/qwen3.7-max
export VVEAI_GEMINI_API_KEY="你的gemini-key"
export VVEAI_GPT55_API_KEY="你的gpt-5.5-key"

uv run python contrast0/run_contrast0.py --seed 42 --workers 1
```

- `--seed`：控制 val/test/train 抽样的随机种子
- `--workers`：并行度（多进程）。建议先从 1 开始；并发过高可能触发 API 限流或快速消耗余额。

## 输出说明

运行完成后会生成：

- `contrast0/contrast0.csv`
  - 列：`大模型, 数据集, ACC, F1, Sensitivity, Specificity`
  - 排序：按模型顺序，再按数据集（UKB → YHD）
- `contrast0/outputs/<DATASET>/<MODEL>/<TIMESTAMP>/...`
  - 每个实验的完整输出目录（包含 `heuristic_system.py`、`final_heuristic_model.py`、`evolution_results.txt`、`final_comparison.txt`、`heldout_test_summary.*` 等）

