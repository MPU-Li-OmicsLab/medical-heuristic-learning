# Ablation 实验（UKB + YHD）

本目录用于在不改动主干训练流程逻辑的前提下，批量运行 2 个数据集的探针消融实验，并把每一组实验输出分开存放，便于排查与复现。

## 目标

对两个数据集分别做 2×2 消融：

- 探针 1：单变量统计探针（Univariate probe）
- 探针 2：知识探针（Knowledge probe，依赖 LLM）

对每个数据集运行 4 组：

- U1_K1：univariate=ON，knowledge=ON（完整）
- U1_K0：univariate=ON，knowledge=OFF
- U0_K1：univariate=OFF，knowledge=ON
- U0_K0：univariate=OFF，knowledge=OFF（不给探针上下文）

切分规则：

- val / test 为 1:1 正负样本随机抽取
- val / test 每个集合大小：1000（即 500 正 + 500 负）
- train 不强制 1:1，采用随机抽取（固定随机种子，保证可复现）
- train_size 设置为：3000 / 1000 / 100 / 10
- 训练时把 val 作为 `run_heuristic_learning(..., test_df=val_df, ...)` 传入，用 val 指标选最优并导出 `final_heuristic_model.py`
- 训练完成后，用导出的 `final_heuristic_model.py` 在 held-out test 上再评估一次并写入汇总文件

## 数据要求

默认读取：

- `./data/YHD_bicarbonate.csv`，标签列：`hospital_expire_flag`
- `./data/UKB.csv`，标签列：`label`

要求标签为二分类 0/1，并且正负样本数量足够支撑 3 个集合各 500 正 + 500 负。

## 运行方式

在仓库根目录运行：

```bash
export DEEPSEEK_API_KEY="你的key"
uv run python ablation/run_ablation.py --workers 1
```

说明：

- 如果你用的是 `python ablation/run_ablation.py` 这种“按文件路径运行”的方式，Python 默认只把 `ablation/` 作为导入根路径，可能会找不到仓库根目录下的 `hl/` 包。本脚本已在运行时自动把仓库根目录加入 `sys.path`，因此可以直接按上述命令运行。
- 并行运行示例（多进程）：`--workers 8` 或 `--workers 16`（并发过大可能触发 API 限流/失败重试）。

## 输出结构

输出根目录为：

`ablation/outputs/<DATASET>/<ABLATION>/<TIMESTAMP>/`

其中 `<ABLATION>` 与训练集规模会共同决定路径，实际结构为：

`ablation/outputs/<DATASET>/<ABLATION>/train<TRAIN_SIZE>/<TIMESTAMP>/`

例如：

- `ablation/outputs/YHD/U1_K1/train1000/20260521_153012/`
- `ablation/outputs/UKB/U0_K0/train10/20260521_153045/`

每个实验目录下会包含主流程产物（val 作为 test_df 的那些文件）以及 held-out test 的额外评估结果：

- `final_heuristic_model.py`：训练结束导出的最终模型（内部路由到 best version）
- `final_comparison.txt`：V0/FINAL/LAST 在 val 上的对比（因为 val 被当作 test_df）
- `evolution_results.txt`：每个版本在 val 上的指标轨迹
- `probe_univariate_results.csv`：若 U1，生成/覆盖；若 U0，可能为空或复用（取决于目录是否已有）
- `probe_knowledge.md`：若 K1 且 LLM 可用，生成/覆盖；若 K0，可能为空或复用
- `iteration_log.json`：每轮迭代日志（覆盖）
- `heldout_test_summary.json`：对 held-out test 的最终评估汇总（本 ablation 脚本生成）
- `heldout_test_summary.txt`：同上，便于快速查看

另外会在 `ablation/outputs/` 下生成一个索引文件 `index_<TIMESTAMP>.json`，记录每组实验输出目录路径。
