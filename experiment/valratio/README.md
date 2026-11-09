# 验证集类别比例扫描实验（UKB + MHL）

本实验只针对 MHL（不运行任何机器学习/深度学习对比模型）。它先在固定的
UKB 数据划分上训练一个 MHL 白盒规则模型，然后对其最终
`final_heuristic_model.py` 在 7 个不同"正类:负类"比例的验证集上分别评估，
记录 Accuracy、F1、Sensitivity、Specificity 以及 TP、FP、FN、TN。

所有命令都应在仓库根目录执行。

## 当前文件与入口

| 文件 | 作用 | 是否直接运行 |
| --- | --- | --- |
| `experiment/valratio/run_valratio_hl.py` | 训练 MHL 并执行验证集比例扫描 | 是 |
| `experiment/valratio/README.md` | 本实验说明 | 否 |

## 运行前准备

### Python 与依赖

项目要求 Python 3.11 及以上。本实验只使用项目运行时依赖：

```bash
uv sync --no-dev
```

### 数据文件

| 数据集 | CSV 路径 | 标签列 |
| --- | --- | --- |
| UKB | `data/UKB.csv` | `label` |

标签会转换为整数 `0/1`。UKB 当前包含约 148k 负例和 37k 正例，
足以支撑本实验的互不重叠抽样。

### MHL 密钥

```bash
export DEEPSEEK_API_KEY="你的密钥"
```

脚本默认使用 `https://api.deepseek.com/v1`、模型 `deepseek-v4-pro`、
温度 `0.0` 和密钥环境变量 `DEEPSEEK_API_KEY`，可通过以下环境变量覆盖：

| 环境变量 | 默认值 |
| --- | --- |
| `VALRATIO_HL_BASE_URL` | `https://api.deepseek.com/v1` |
| `VALRATIO_HL_KEY_ENV` | `DEEPSEEK_API_KEY` |
| `VALRATIO_HL_MODEL` | `deepseek-v4-pro` |
| `VALRATIO_HL_TEMPERATURE` | `0.0` |

## 数据划分设计

每次运行使用固定种子构建全部互不重叠的冻结划分：

| 划分 | 总量 | 正例 | 负例 |
| --- | ---: | ---: | ---: |
| 训练集 | 3000 | 1500 | 1500 |
| 测试集 | 1000 | 500 | 500 |
| MHL 内部验证集 | 3000 | 1500 | 1500 |
| 比例验证集 1:1 | 3000 | 1500 | 1500 |
| 比例验证集 1:2 | 3000 | 1000 | 2000 |
| 比例验证集 2:1 | 3000 | 2000 | 1000 |
| 比例验证集 1:4 | 3000 | 600 | 2400 |
| 比例验证集 4:1 | 3000 | 2400 | 600 |
| 比例验证集 1:8 | 3000 | 333 | 2667 |
| 比例验证集 8:1 | 3000 | 2667 | 333 |

设计要点：

- 训练集和测试集均为 1:1 平衡，与要求一致。
- MHL 在规则生成和版本迭代中看到的是 3000 条 1:1 验证集。
- 7 个比例验证集并不是从同一个 1:1 池子里二次抽样得到的，而是每个比例
  都从原始 UKB 中重新抽取 3000 条新样本；它们与训练集、测试集、MHL
  验证集以及彼此之间均互不重叠。
- 这样处理的原因是：若从一个固定 1500/1500 的池子里构造 `4:1` 或
  `1:8`，样本量必然不足；为保持"每个比例验证集均为 3000 条"且不做
  有放回抽样，必须为每个比例分配独立样本。
- 所有划分的源行标识、类别计数、哈希和两两无交集校验都会写入
  `data_splits/split_manifest.json`。

## 运行方式

### 完整运行（训练 + 比例扫描）

```bash
uv run python experiment/valratio/run_valratio_hl.py --seed 42
```

默认参数：`--train-total 3000`、`--test-total 1000`、
`--mhl-val-total 3000`、`--val-ratio-total 3000`、`--iterations 10`。

### 只检查数据划分（不发 LLM 请求）

```bash
uv run python experiment/valratio/run_valratio_hl.py \
  --seed 42 --check-split --output-root /tmp/valratio_check
```

该模式只构建并校验冻结划分，不调用任何 API。

### 评估已经训练好的模型（跳过训练）

如果已经有一个 MHL `final_heuristic_model.py`，可以只做比例扫描：

```bash
uv run python experiment/valratio/run_valratio_hl.py \
  --seed 42 --model-path /path/to/final_heuristic_model.py
```

脚本会用相同种子重建同一份冻结划分，然后加载给定模型并输出 7 行结果。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed` | `42` | 全部划分共用的随机种子 |
| `--train-total` | `3000` | 1:1 平衡训练集总量 |
| `--test-total` | `1000` | 1:1 平衡测试集总量 |
| `--mhl-val-total` | `3000` | MHL 规则演化验证集总量 |
| `--val-ratio-total` | `3000` | 每个比例验证集的总量 |
| `--iterations` | `10` | MHL 规则迭代轮数 |
| `--model-path` | 无 | 已训练模型路径；设置后跳过训练 |
| `--check-split` | 关闭 | 只构建/校验划分，不发 LLM 请求 |
| `--output-root` | `experiment/valratio/outputs` | 产物根目录 |
| `--csv-out` | `experiment/valratio/valratio_hl_seed<seed>.csv` | 结果 CSV 路径 |

## 输出目录与文件

单个种子的运行目录：

```text
experiment/valratio/outputs/seed<seed>/<时间戳>/
├── data_splits/
│   ├── train.csv
│   ├── test.csv
│   ├── mhl_val.csv
│   ├── ratio_1_1.csv
│   ├── ratio_1_2.csv
│   ├── ratio_2_1.csv
│   ├── ratio_1_4.csv
│   ├── ratio_4_1.csv
│   ├── ratio_1_8.csv
│   ├── ratio_8_1.csv
│   └── split_manifest.json
├── validation_ratio_summary.json
├── validation_ratio_summary.txt
├── predictions_test.csv
├── predictions_ratio_<比例>.csv
└── （MHL 标准产物）
    ├── probe_univariate_results.csv
    ├── probe_knowledge.md
    ├── heuristic_system.py
    ├── evolution_results.txt
    ├── iteration_log.json
    ├── final_heuristic_model.py
    └── final_comparison.txt
```

汇总 CSV（每个种子一个）：

```text
experiment/valratio/valratio_hl_seed<seed>.csv
```

字段为：种子、数据集、验证集比例、验证集总量、正例数、负例数、ACC、
F1、Sensitivity、Specificity、TP、FP、FN、TN、status、error、out_dir。
其中 `TP + FP + FN + TN` 应等于该行验证集总量 3000。

## 注意事项

- 本实验不训练任何对比模型；比例扫描只发生在验证阶段。
- MHL 训练和验证需要远程 LLM 调用，耗时长且产生费用；建议先用
  `--check-split` 确认划分，再单种子完整运行。
- 同一个种子重新运行会生成新的时间戳目录，不会覆盖已有 MHL 产物。
- 汇总 CSV 中 `ACC`、`F1`、`Sensitivity`、`Specificity` 保留 3 位小数，
  `TP/FP/FN/TN` 为整数。
