# Ablation 实验说明

本目录用于批量运行启发式学习系统的消融实验。当前包含 3 个脚本：

- `run_ablation.py`：标准消融主脚本。训练集从剩余样本中随机抽样，不强制类别平衡。
- `run_ablation_balance.py`：平衡训练集版本。训练集按 1:1 正负类采样，必要时允许有放回抽样。
- `run_batches.py`：批量重复运行器。多次调用 `run_ablation.py`，每次使用新的随机种子，并把每一轮完整结果分别写入独立目录。

## 实验目标

对两个数据集分别做 2×2 探针消融：

- 单变量统计探针（Univariate probe）
- 知识探针（Knowledge probe，依赖 LLM）

每个数据集包含 4 组配置：

- `U1_K1`：`run_univariate_probe=True`，`run_knowledge_probe=True`
- `U1_K0`：`run_univariate_probe=True`，`run_knowledge_probe=False`
- `U0_K1`：`run_univariate_probe=False`，`run_knowledge_probe=True`
- `U0_K0`：`run_univariate_probe=False`，`run_knowledge_probe=False`

每组配置默认再枚举 4 个训练集规模：

- `3000`
- `1000`
- `100`
- `10`

因此：

- `run_ablation.py` 或 `run_ablation_balance.py` 默认各会跑 `2 个数据集 × 4 个消融配置 × 4 个训练集规模 = 32` 个实验
- `run_batches.py --runs N` 会重复执行上述整套 32 个实验共 `N` 次

## 数据与切分规则

默认读取两个数据集：

- `./data/YHD_bicarbonate.csv`，标签列为 `hospital_expire_flag`
- `./data/UKB.csv`，标签列为 `label`

通用要求：

- 标签列必须是二分类 `0/1`
- 正负样本数都必须足够支撑验证集和测试集的平衡抽样

`run_ablation.py` 与 `run_ablation_balance.py` 的共同点：

- `val` 与 `test` 都采用 1:1 正负样本抽样
- `val_total=1000`，即 `500` 正样本 + `500` 负样本
- `test_total=1000`，即 `500` 正样本 + `500` 负样本
- 训练时把 `val_df` 传给 `run_heuristic_learning(..., test_df=val_df, ...)`
- 主流程结束后，会加载导出的 `final_heuristic_model.py`，再对 held-out `test_df` 额外评估一次

两者差异：

- `run_ablation.py`：训练集从剩余样本中随机抽取，不保证 1:1 平衡
- `run_ablation_balance.py`：训练集按 1:1 正负类采样；若某一类样本不足，会启用有放回抽样，并把采样信息写入结果

## 脚本用法

请在仓库根目录运行，并先设置 `DEEPSEEK_API_KEY`：

```bash
export DEEPSEEK_API_KEY="你的 key"
```

### 1. 标准消融

```bash
uv run python ablation/run_ablation.py --workers 1
```

常用参数：

- `--workers`：并发进程数，默认 `1`
- `--seed`：随机种子，默认 `42`
- `--output-root`：输出根目录，默认 `./ablation/outputs`
- `--dataset`：可选 `YHD` 或 `UKB`
- `--ablation`：可选 `U1_K1`、`U1_K0`、`U0_K1`、`U0_K0`
- `--train-size`：可选 `3000`、`1000`、`100`、`10`

示例：

```bash
uv run python ablation/run_ablation.py --dataset UKB --ablation U1_K0 --train-size 100 --workers 4
```

### 2. 平衡训练集消融

```bash
uv run python ablation/run_ablation_balance.py --workers 1
```

常用参数与 `run_ablation.py` 基本一致，默认输出目录不同：

- `--output-root` 默认是 `./ablation/output_balance`

### 3. 多次重复实验

```bash
uv run python ablation/run_batches.py --runs 10 --workers 8
```

参数说明：

- `--runs`：重复运行次数，默认 `10`
- `--workers`：每一轮内部执行 `run_ablation.py` 时使用的并发进程数，默认 `1`
- `--base-dir`：批次输出根目录，默认 `./ablation/outputs_batches`

注意：

- `run_batches.py` 当前内部调用的是 `run_ablation.py`，不是 `run_ablation_balance.py`
- 同一轮 batch 内的 32 个实验共享同一个随机种子
- 不同 batch 之间会重新随机生成种子

## 输出结构

### 单次消融输出

对于 `run_ablation.py`，默认输出根目录为：

`ablation/outputs/`

对于 `run_ablation_balance.py`，默认输出根目录为：

`ablation/output_balance/`

单个实验目录结构为：

`<output-root>/<DATASET>/<ABLATION>/train<TRAIN_SIZE>/<TIMESTAMP>/`

例如：

- `ablation/outputs/YHD/U1_K1/train1000/20260521_153012/`
- `ablation/output_balance/UKB/U0_K0/train10/20260521_153045/`

每个实验目录通常会包含：

- `final_heuristic_model.py`：最终导出的模型
- `final_comparison.txt`：主流程在验证集上的版本对比结果
- `evolution_results.txt`：各迭代版本在验证集上的指标轨迹
- `iteration_log.json`：迭代日志
- `heldout_test_summary.json`：本脚本额外生成的 held-out test 汇总
- `heldout_test_summary.txt`：上述汇总的文本版

可能出现的探针相关文件：

- `probe_univariate_results.csv`：仅在开启单变量探针时才有意义
- `probe_knowledge.md`：仅在开启知识探针且 LLM 调用成功时才有意义

由于每次实验都会写入新的时间戳目录，因此不存在“复用旧目录中的探针文件”这一行为。

### 单批次汇总文件

当一次运行结束后，`--output-root` 指定的目录下会额外写出：

- `index.json`：记录本批次的 `seed`、`workers`、`created_at` 与每个实验的执行结果
- `ablation.csv`：聚合本批次所有实验的 held-out test 指标

`ablation.csv` 当前包含的列为：

- `数据集`
- `U`
- `K`
- `训练集规模`
- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`
- `保留的是第几次迭代的结果`
- `status`
- `out_dir`
- `error`

## 批量重复实验的输出

`run_batches.py` 会在 `--base-dir` 下为每一轮创建独立目录：

- `outputs_01_<TIMESTAMP>/`
- `outputs_02_<TIMESTAMP>/`
- ...

每个这样的目录内部，都会包含一次完整 `run_ablation.py` 的输出结果，也就是各自的：

- `index.json`
- `ablation.csv`
- `YHD/...`
- `UKB/...`

全部 batch 完成后，还会在 `--base-dir` 下生成一个总索引：

- `batches_<TIMESTAMP>.json`

它只记录每一轮的：

- `run`
- `seed`
- `output_root`

## 其他说明

- 脚本在运行时会自动把仓库根目录加入 `sys.path`，因此可以直接使用 `python ablation/run_ablation.py` 这种按路径执行的方式
- 若 `--workers` 设置过大，虽然程序仍会尝试并行执行，但更容易遇到 API 限流或外部调用失败
- `run_ablation.py` 和 `run_ablation_balance.py` 都会把异常捕获进结果汇总中；单个实验失败不会阻止整批结果写出 `index.json`
