# 对比实验扩展与统一建模方案

## 1. 文档目的

本文档描述 `contrast1`、`contrast2` 和 `continuous_learning` 的最新扩展方案。为保证所有普通 baseline 使用相同的数据、随机种子、预处理约束和关闭自动平衡后的配置，以下十个普通模型都要重新运行：

- `LogisticRegression`
- `DecisionTree`
- `MLP`
- `XGBoost`
- `LightGBM`

- `FT-Transformer`：改用 DeepTab 的稳定实现，废弃实验脚本内的自实现版本
- `ResNet`：使用 DeepTab 的表格 ResNet 实现
- `EBM`：使用 InterpretML 的 `ExplainableBoostingClassifier`
- `APLR`：使用 InterpretML 的 `APLRClassifier`
- `CORELS`：使用官方 `corels` Python 包

只有 HL 不重跑；最终汇总时复用已经完成的 HL 结果。

三个实验统一使用随机种子：

```text
36、40、42
```

本方案还将建立一个**精简且扁平**的 `experiment/modeling/`。它只统一普通对比模型的参数、构造、训练和预测，不接管数据切分、实验矩阵、结果 CSV 或 HL 流程。

## 2. 总体实施原则

1. `contrast1` 基于 `experiment/contrast1/run_contrast1_balance.py` 修改。
2. `contrast2` 基于 `experiment/contrast2/run_contrast2.py` 修改。
3. `continuous_learning` 继续复用 `continuous_learning_experiment_common.py` 中的数据流和两阶段特征漂移定义。
4. LogisticRegression、DecisionTree、MLP、XGBoost、LightGBM 也要重新运行；只有 HL 不重跑，并在最终汇总时读取既有 HL 结果。
5. 旧的自实现 FT-Transformer 不再作为正式 FT-Transformer 结果；FT-Transformer 必须用 DeepTab 重新训练和测试。
6. 十个重跑模型的原始结果 CSV 保持既有实验的列名和指标格式。
7. 数据集类别数量在训练前确定；模型层和训练框架层不得再次自动平衡。
8. 所有预处理器只能在训练集上拟合，验证集和测试集只允许执行 transform/predict。
9. 测试集不参与超参数选择、early stopping 或分类阈值选择。
10. 保存每次实验的数据清单、模型参数、依赖版本、类别数量和模型产物，以便复现和审计。

## 3. 参考实现与版本选择

### 3.1 DeepTab

DeepTab 2.x 提供 sklearn 风格的 `fit`、`predict`、`predict_proba`、`save` 和 `load` 接口，并将模型结构、预处理和训练参数分别放在三类配置对象中：

- `<Model>Config`
- `PreprocessingConfig`
- `TrainerConfig`

本项目采用以下稳定接口：

```python
from deeptab.configs import (
    FTTransformerConfig,
    PreprocessingConfig,
    ResNetConfig,
    TrainerConfig,
)
from deeptab.models import FTTransformerClassifier, ResNetClassifier
```

参考资料：

- [DeepTab 仓库](https://github.com/OpenTabular/DeepTab)
- [FT-Transformer 文档](https://deeptab.readthedocs.io/en/stable/model_zoo/stable/fttransformer.html)
- [ResNet 文档](https://deeptab.readthedocs.io/en/stable/model_zoo/stable/resnet.html)
- [DeepTab sklearn API](https://deeptab.readthedocs.io/en/stable/core_concepts/sklearn_api.html)
- [DeepTab 模型保存](https://deeptab.readthedocs.io/en/stable/core_concepts/model_operations.html)
- [FT-Transformer 与表格 ResNet 原始论文](https://arxiv.org/abs/2106.11959)

计划固定使用 `deeptab==2.0.0`，避免 DeepTab 2.x 后续接口或默认参数变化影响复现。

### 3.2 InterpretML、EBM 与 APLR

EBM 使用：

```python
from interpret.glassbox import ExplainableBoostingClassifier
```

APLR 使用：

```python
from interpret.glassbox import APLRClassifier
```

InterpretML 的 APLR 是对外部 `aplr` 包的轻量封装，因此 `interpret-core` 和 `aplr` 都作为直接 dev 依赖记录。

参考资料：

- [InterpretML 文档](https://interpret.ml/docs/index.html)
- [EBM API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)
- [APLR 文档](https://interpret.ml/docs/aplr.html)
- [APLR 论文](https://doi.org/10.1007/s00180-024-01475-4)

### 3.3 CORELS

CORELS 按官方 pycorels 仓库推荐方式安装：

```text
pip install corels
```

在本项目中由 `uv` 解析和安装等价的 `corels==1.1.29`。不使用非官方 fork 或行为近似的替代模型。

参考资料：

- [pycorels 官方仓库](https://github.com/corels/pycorels)
- [CorelsClassifier API](https://pycorels.readthedocs.io/en/latest/CorelsClassifier.html)
- [CORELS 论文](https://jmlr.org/papers/v18/17-716.html)

CORELS 的官方 PyPI 包是源码分发，安装时需要 C++ 编译器；GMP 可提升性能但不是必要依赖。安装后必须先执行官方 toy-data 拟合测试。若官方包在当前 Python/NumPy 环境下构建失败，应保留完整错误并停止 CORELS 实验，不得静默切换到其他实现。

实际实施中，1.1.29 sdist 自带的旧生成 C++ 与 Python 3.11 / NumPy 2 不兼容；
因此使用 dev 组的 Cython 3 从官方 `_corels.pyx` 重新生成 C++，再由官方
`setup.py` 构建本机 wheel。未使用 fork 或近似替代实现，且 toy-data 闸门已通过。

## 4. 依赖管理方案

在根目录 `pyproject.toml` 的 `[dependency-groups].dev` 中加入：

```toml
dev = [
    "scikit-learn>=1.8.0",
    "lightgbm>=4.6.0",
    "torch>=2.3.0",
    "xgboost>=2.1.0",
    "deeptab==2.0.0",
    "interpret-core[aplr]==0.7.8",
    "aplr==10.23.0",
    "corels==1.1.29",
    "cython>=3.0,<4",
    "setuptools>=78.1.1,<80",
    "wheel>=0.46.0",
]
```

DeepTab 2.0.0 要求 Python `>=3.10,<3.14`，而当前使用的 uv 0.6.9 不支持给 dev
组单独声明 `requires-python`。实际实现将项目范围收紧为 `>=3.11,<3.14`，并因
DeepTab 2.0.0 的约束将 pandas 固定为 `<3.0.0`。CORELS 的源码构建工具也显式
列入 dev 组。

```toml
[project]
requires-python = ">=3.11,<3.14"
```

依赖修改后执行：

```bash
uv lock
uv sync --group dev
```

验收命令应验证：

```bash
uv run python -c "import deeptab, interpret, aplr, corels; print('imports_ok')"
```

然后打印并保存实际解析版本。`uv.lock` 必须与 `pyproject.toml` 一起提交或保存。

## 5. 精简的统一建模层

### 5.1 对上一版设计的修正

上一版为 `experiment/modeling/` 规划了协议、注册表、平衡策略、预处理、持续学习、产物管理以及多层 adapters，共十余个文件。当前仓库只有三个固定实验入口，不需要插件系统，也没有多套可替换后端，因此这种设计会造成：

- 查找一个模型的完整训练路径时需要跨越多个文件；
- 大量类和协议只会有一个实现；
- `experiment/modeling/` 与各实验脚本的数据、结果职责重叠；
- 为了把 HL 塞进 estimator 接口而产生没有必要的包装层。

修正后的原则是：**只抽取三个实验真正重复的模型代码，其余逻辑留在最了解实验语义的位置。** 不建立 `contracts.py`、`registry.py`、`balance.py`、`preprocessing.py`、`continuous.py`、`artifacts.py`、`adapters/` 或 `experiment/common/`。

### 5.2 最终目录：只保留 4 个 `.py`

```text
experiment/
└── modeling/
    ├── __init__.py
    ├── config.py
    ├── models.py
    └── train_eval.py
```

这是本方案的上限，不再继续按模型家族拆子目录。只有某个文件实际超过约 500 行且已经难以阅读时，才在后续提交中讨论拆分；不为了“可能的扩展”提前拆分。

### 5.3 每个 `.py` 的用途和文件顶部注释

#### `experiment/modeling/__init__.py`

用途：定义建模包唯一的公开入口。实验脚本只从这里导入模型名称、`fit_model`、`predict_model`、`predict_positive_probability` 和 `evaluate_model`，不直接依赖内部文件。该文件不放训练逻辑。

实现时写入以下文件级注释：

```python
"""对比实验普通模型的统一入口。

这里只重导出模型配置、训练、预测和评估函数；数据切分、实验循环、
结果 CSV 与 HL 流程仍由各实验目录负责。
"""
```

#### `experiment/modeling/config.py`

用途：集中保存所有普通对比模型的名称和固定超参数，包括既有的 LogisticRegression、DecisionTree、MLP、XGBoost、LightGBM，以及新增的 FT-Transformer、ResNet、EBM、APLR、CORELS。这里显式关闭模型层自动平衡，并通过 `EXPERIMENT_MODEL_NAMES` 选择本轮需要重跑的全部十个模型。

实现时写入以下文件级注释：

```python
"""对比模型名称与固定超参数。

本文件是三个实验共享的模型配置单一来源。类别比例由实验脚本在训练前
构造，本文件中的模型不得启用 class_weight、sample_weight 或加权采样。
"""
```

配置保持普通常量和字典即可，例如：

```python
ALL_MODEL_NAMES = (
    "LogisticRegression",
    "DecisionTree",
    "MLP",
    "XGBoost",
    "LightGBM",
    "FT-Transformer",
    "ResNet",
    "EBM",
    "APLR",
    "CORELS",
)

NEW_MODEL_NAMES = (
    "FT-Transformer",
    "ResNet",
    "EBM",
    "APLR",
    "CORELS",
)

EXPERIMENT_MODEL_NAMES = ALL_MODEL_NAMES
```

不为每个配置再定义一套 dataclass；只有 DeepTab 官方 API 明确要求的 `FTTransformerConfig`、`ResNetConfig`、`PreprocessingConfig` 和 `TrainerConfig` 在构造时使用。

#### `experiment/modeling/models.py`

用途：根据模型名和 seed 构造模型。函数保持直接、可搜索，例如 `build_model(model_name, seed)`；使用少量 `if/elif` 或简单字典映射，不引入注册器类、工厂类或 Adapter 继承体系。可选依赖在选择对应模型时再导入，并给出明确缺包错误。

实现时写入以下文件级注释：

```python
"""构造对比实验使用的普通模型。

build_model() 统一创建 sklearn、XGBoost、LightGBM、DeepTab、InterpretML
和 CORELS 模型。该模块只负责实例化，不负责切分数据或写实验结果。
"""
```

该文件负责：

- 将统一 seed 传入每个支持随机种子的实现；
- 应用 `config.py` 的固定参数；
- 确保不设置 `class_weight="balanced"`、`scale_pos_weight` 或其它自动平衡项；
- 返回模型对象及一个简单的模型家族标记，例如 `sklearn`、`deeptab` 或 `corels`，供训练函数选择必要分支。

#### `experiment/modeling/train_eval.py`

用途：容纳真正重复的训练、预测和指标计算。仅按三类公开 API 保留三个私有分支：sklearn-compatible、DeepTab、CORELS；不为每个模型单独建立 adapter 文件。

实现时写入以下文件级注释：

```python
"""普通对比模型的共享训练、预测和评估逻辑。

预处理器只在训练集上拟合，验证集只用于 DeepTab 的模型选择，测试集只做
最终评估。所有指标统一调用 hl.metrics.compute_metrics。
"""
```

该文件只提供四个小接口：

```python
fit_model(model_name, train_df, val_df, label_col, seed)
predict_model(fitted_model, data_df)
predict_positive_probability(fitted_model, data_df)
evaluate_model(fitted_model, test_df, label_col)
```

`fit_model` 返回一个轻量 `FittedModel` dataclass，内容只包括模型名、模型家族、已拟合模型、必要的预处理器和训练摘要。这里不设计通用协议或继承树。

- sklearn-compatible 分支：统一训练集拟合的缺失值处理、编码和缩放；EBM/APLR 若官方实现能原生接收合适的 DataFrame，则使用最少必要转换。
- DeepTab 分支：使用 DeepTab 自己的预处理和 Trainer，验证集只用于 early stopping/最佳 epoch。
- CORELS 分支：二值化阈值只从训练集计算，并随 `FittedModel` 保留供验证、测试和持续学习预测复用。
- `evaluate_model`：统一调用 `hl.metrics.compute_metrics`，并返回与旧 CSV 对应的指标字典。

模型保存不是新的抽象层：需要保存时，由 `train_eval.py` 提供一个很小的 `save_fitted_model(...)` 辅助函数，调用各库公开保存方式即可。

### 5.4 明确保留在 `experiment/modeling/` 之外的逻辑

为避免职责再次膨胀，下列内容不迁入该目录：

| 逻辑 | 保留位置 | 原因 |
|---|---|---|
| contrast1 数据切分、训练规模循环、CSV 列格式 | `contrast1/run_contrast1_balance.py` | 只属于 contrast1 |
| contrast2 类别比例采样、混淆矩阵、CSV 列格式 | `contrast2/run_contrast2.py` | 只属于 contrast2 |
| 两阶段数据、SIRS/SOFA 漂移 | `continuous_learning_experiment_common.py` | 只属于持续学习实验 |
| Stage 1 到 Stage 2 的 prior cascade | `run_continuous_learning_baselines.py` | 是实验设计，不是模型能力 |
| HL 训练和持续学习 | `hl/` 与现有 HL 实验入口 | HL 是规则生成流程，不是普通 estimator |
| 结果文件断点续写与排序 | 各实验入口 | 三个实验的任务键和列不同 |
| seeds `36、40、42` | 各实验 CLI/default settings | seed 属于实验矩阵，不属于模型超参数 |

### 5.5 统一范围

`experiment/modeling/` 统一以下十个普通模型：

- LogisticRegression
- DecisionTree
- MLP
- XGBoost
- LightGBM
- FT-Transformer
- ResNet
- EBM
- APLR
- CORELS

HL 不放入该目录。它继续复用 `hl/` 主干和独立实验入口，最终只在结果汇总阶段与上述模型比较。这样既能统一所有常规 baseline，又不会为了表面统一破坏仓库已有的 `hl/`/`experiment/` 边界。

本轮正式运行选择全部十个普通模型：

```text
LogisticRegression, DecisionTree, MLP, XGBoost, LightGBM,
FT-Transformer, ResNet, EBM, APLR, CORELS
```

既有五个普通模型也必须基于统一配置重新训练和测试。特别是当前 `contrast1` 中 LogisticRegression 和 DecisionTree 使用了 `class_weight="balanced"`，重跑时必须将其关闭，不能复用这些旧结果。只有 HL 直接复用既有结果。

### 5.6 调用关系

```text
contrast1 / contrast2 / continuous_learning 实验脚本
                     │
                     ▼
       experiment.modeling（__init__.py）
              │                    │
              ▼                    ▼
    models.py + config.py      train_eval.py
      构造模型与参数          拟合、预测、计算指标
```

没有注册器、adapter 继承树或跨目录回调。阅读一次模型运行只需从实验入口进入 `train_eval.py`，再查看 `models.py` 和 `config.py`。

## 6. 类别平衡策略

### 6.1 唯一允许的平衡位置

类别数量只能在数据准备阶段确定，例如：

- contrast1 在训练前构造 1:1 训练集。
- contrast2 在训练前按目标正负比构造训练集。
- continuous_learning 在训练前构造两个阶段的平衡训练集。

如果原始训练池某一类不足，是否有放回采样也必须在模型训练前完成并记录。模型实际收到的数据长度和正负样本数必须与 manifest 完全一致。

### 6.2 各模型的关闭配置

| 模型 | 必须使用的配置 |
|---|---|
| LogisticRegression | `class_weight=None` |
| DecisionTree | `class_weight=None` |
| MLP | 不传 `sample_weight` |
| XGBoost | `scale_pos_weight=1.0` |
| LightGBM | `class_weight=None`、`is_unbalance=False`、`scale_pos_weight=1.0` |
| FT-Transformer | `class_weight=None`、普通 BCE、普通 shuffle sampler |
| ResNet | `class_weight=None`、普通 BCE、普通 shuffle sampler |
| EBM | 不传 `sample_weight` |
| APLR | 不传 `sample_weight` 或类别权重 |
| CORELS | 不使用额外权重或模型内重采样 |

统一平衡审计应拒绝：

- `class_weight="balanced"`
- `class_weight="balanced_subsample"`
- `is_unbalance=True`
- 非 1 的 `scale_pos_weight`
- PyTorch `pos_weight`
- `WeightedRandomSampler`
- 任何非空 `sample_weight`
- 以不平衡为理由自动启用的 focal loss 或类别 alpha

DeepTab 拟合时显式传入：

```python
model.fit(
    X_train,
    y_train,
    X_val=X_val,
    y_val=y_val,
    loss_fct="bce",
    class_weight=None,
)
```

DeepTab 的 `stratify` 仅用于在没有显式验证集时保持内部划分比例，不属于类别再平衡；本实验始终传入显式验证集。

## 7. 新模型固定配置

本轮不执行超参数搜索。所有配置在正式实验开始前冻结，避免模型之间的调参预算不一致，也避免测试集参与选择。

### 7.1 FT-Transformer

```python
FTTransformerConfig(
    d_model=128,
    n_layers=4,
    n_heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    pooling_method="avg",
)
```

训练配置：

- 数值预处理：`standardization`
- 类别预处理：`int`
- 学习率：`3e-4`
- 最大 epoch：100
- patience：15
- batch size：`max(2, min(128, train_size))`
- weight decay：`1e-6`
- early stopping 指标：`val_loss`
- 损失：未加权 BCE
- 设备：当前正式实验固定 CPU、单设备、32 位精度
- 保存格式：`.deeptab`

### 7.2 ResNet

```python
ResNetConfig(
    layer_sizes=[256, 128, 32],
    num_blocks=3,
    dropout=0.2,
    norm=False,
)
```

训练配置与 FT-Transformer 相同，区别为初始学习率使用 `1e-3`。

### 7.3 EBM

使用 InterpretML 0.7.8 的官方实现和默认算法参数，只覆盖：

```python
ExplainableBoostingClassifier(
    random_state=seed,
    n_jobs=1,
)
```

要求：

- 不传 `sample_weight`。
- 保留原始列名和可解释特征类型。
- 数值缺失沿用 EBM 的独立 missing 处理。
- 保存模型参数、`best_iteration_`、特征名和全局项信息。

### 7.4 APLR

使用：

```python
APLRClassifier(random_state=seed, cv_folds=2, n_jobs=1)
```

要求：

- 通过 InterpretML wrapper 调用，不直接绕过 InterpretML。
- 固定使用两折交叉验证，使 contrast1 最小的 5 正 + 5 负训练集也可运行；该设置
  在正式实验冻结后对所有规模保持一致。
- 数值填补、类别 one-hot 只在训练集拟合。
- 不传样本权重或类别权重。
- 记录 wrapper 和底层 `aplr` 的实际版本及解析参数。
- 预测结果按照 `classes_` 安全映射为整数 0/1。

### 7.5 CORELS

使用：

```python
CorelsClassifier(
    c=0.01,
    n_iter=10000,
    map_type="prefix",
    policy="lower_bound",
    verbosity=[],
    max_card=2,
    min_support=0.01,
)
```

CORELS 只接受二值矩阵，因此需要单独的训练集二值化器：

- 二值列生成值谓词。
- 普通连续列使用训练集四分位点生成互斥区间谓词。
- 类别列使用训练集 one-hot。
- 缺失值生成独立 missing 谓词。
- 重复切点和常量谓词自动删除。
- 输出强制检查为 `uint8` 且只能包含 0/1。
- 切点完全基于 `X_train`，不使用标签选择切点。
- 保存 `predicate_manifest.json`、CORELS 模型和人类可读规则列表。
- 如果搜索达到 `n_iter` 上限，记录实际终止状态，不声称已获得最优性证书。

## 8. 数据切分与随机数

三个实验统一使用：

```python
EXPERIMENT_SEEDS = (36, 40, 42)
```

每个 seed 同时控制：

- train/validation/test 行划分
- 训练集正负类采样
- 模型随机初始化
- DeepTab/PyTorch/Lightning 随机状态
- EBM/APLR 的随机状态

同一实验条件的数据只能生成一次，再提供给十个模型，不能让每个模型自行重新抽样。建议使用 `numpy.random.SeedSequence` 派生数据子种子，并将派生关系写入 manifest。

每个 split manifest 至少记录：

- 数据集路径和文件哈希
- 标签列
- base seed 和派生 seed
- 训练、验证、测试 source row IDs 或其稳定哈希
- 各集合总数和正负样本数
- 是否有放回采样
- 唯一 source row 数
- 特征列顺序
- 数据漂移信息

训练、验证和测试源行必须互不重叠。只有训练集为满足预设数量时允许有放回采样，验证集和测试集禁止重复或相互重叠。

## 9. contrast1 实验方案

### 9.1 基准脚本

基于：

```text
experiment/contrast1/run_contrast1_balance.py
```

重构后该脚本保留数据切分、实验矩阵、CSV 输出和 CLI；模型构造、训练与预测调用 `experiment.modeling` 的四个公开函数。

### 9.2 数据集

- UKB：`data/UKB.csv`，标签 `label`
- YHD：`data/YHD_bicarbonate.csv`，标签 `hospital_expire_flag`

### 9.3 训练集规模

```text
3000、1000、500、100、50、10
```

每个训练集在模型训练前构造为 1:1：

```text
positive = train_size / 2
negative = train_size / 2
```

不足部分只允许在训练集构造阶段有放回采样。

### 9.4 运行矩阵

```text
2 个数据集 × 6 个训练规模 × 10 个模型 × 3 个 seed = 360 行
```

每个 seed 应恰好产生 120 行。

### 9.5 输出格式

原始 CSV 保持现有列：

```text
模型
数据集
训练集数据量
ACC
F1
Sensitivity
Specificity
best_epoch
checkpoint
status
error
```

由于旧格式没有 seed 列，分种子保存：

```text
contrast1_balance_rerun_seed36.csv
contrast1_balance_rerun_seed40.csv
contrast1_balance_rerun_seed42.csv
```

FT-Transformer 和 ResNet 可填写 `best_epoch`、`checkpoint`；其它模型保持空值，不改变列定义。

## 10. contrast2 实验方案

### 10.1 基准脚本

基于：

```text
experiment/contrast2/run_contrast2.py
```

### 10.2 数据集与训练总量

数据集与 contrast1 相同。

训练总量：

```text
1000、3000
```

正:负比例：

```text
1:1
1:2
2:1
1:5
5:1
1:10
10:1
1:50
50:1
```

正负目标数沿用当前脚本规则：

```python
positive = round(total * pos / (pos + neg))
positive = max(1, min(positive, total - 1))
negative = total - positive
```

在 `fit` 前断言实际类别数量等于目标值。任何模型均不得根据当前比例计算或启用额外类别权重。

### 10.3 运行矩阵

```text
2 个数据集 × 2 个训练总量 × 9 个比例 × 10 个模型 × 3 个 seed = 1080 行
```

每个 seed 应恰好产生 360 行。

### 10.4 输出格式

保持现有列：

```text
模型
数据集
训练集数据量
训练集正负比
ACC
F1
Sensitivity
Specificity
TP
FP
FN
TN
status
error
```

分种子保存：

```text
contrast2_rerun_seed36.csv
contrast2_rerun_seed40.csv
contrast2_rerun_seed42.csv
```

## 11. continuous_learning 实验方案

### 11.1 共享数据流

继续使用：

```text
experiment/continuous_learning/continuous_learning_experiment_common.py
```

统一种子保持当前既有值：

```python
DEFAULT_SEEDS = (36, 40, 42)
```

### 11.2 两阶段设置

以当前代码的真实配置为准：

| 阶段 | 训练集 | 验证集 | 测试集 | 特征变化 |
|---|---:|---:|---:|---|
| Stage 1 | 1000 | 500 | 800 | 使用 SIRS，不使用 SOFA |
| Stage 2 | 40 | 500 | 800 | 删除 SIRS，增加 SOFA |

所有集合为 1:1 平衡，Stage 1 和 Stage 2 的六个集合源行两两不重叠。

当前 README 中存在“Stage 2 训练 10 条、验证/测试各 500”的旧描述，实施时应更新为代码实际使用的 Stage 2 训练 40、验证 500、测试 800。

### 11.3 DeepTab 的持续学习限制

DeepTab 高层 estimator API 不能直接从 checkpoint 继续训练，且 `.deeptab` 产物绑定原训练特征 schema。Stage 1 的 SIRS 与 Stage 2 的 SOFA 语义不同，因此不能因为数组维度相同就直接复用最后一列权重，也不能通过 DeepTab 私有 `_task_model` 实现不稳定的隐式迁移。

### 11.4 统一的先验特征级联

五个新模型统一采用 `prior_feature_cascade`：

1. 使用 Stage 1 训练集训练 Stage 1 模型。
2. 在 Stage 1 测试集上评估并保存模型。
3. 为 Stage 2 的 train/val/test 构造 Stage 1 兼容视图。
4. 兼容视图只使用两个阶段的共享特征。
5. Stage 1 所需但 Stage 2 已删除的 SIRS，使用 Stage 1 训练集的中位数或众数填充。
6. Stage 2 的 SOFA 在生成 Stage 1 先验时被忽略。
7. 严禁读取 Stage 2 样本的真实 SIRS 值。
8. 用 Stage 1 模型为 Stage 2 train/val/test 生成先验预测。
9. 将先验作为新特征加入 Stage 2 的 SOFA 特征空间。
10. 在 Stage 2 的 40 条训练数据上训练新的同类模型。

先验特征形式：

- FT-Transformer、ResNet、EBM、APLR：`stage1_prior_probability`
- CORELS：`stage1_prior_prediction`，取值 0/1

Stage 2 结果可使用 `status=continued`，并在 `out_dir` 下保存 `continuation_manifest.json`，内容包括：

- Stage 1 模型路径
- Stage 1/Stage 2 列映射
- SIRS 填充值及其来源
- prior 特征名称和取值范围
- Stage 1 与 Stage 2 source row 审计
- `continuation_strategy="prior_feature_cascade"`

既有的 LogisticRegression、DecisionTree、MLP、XGBoost 和 LightGBM 也重新运行，但继续沿用当前 continuous baseline 脚本为各模型定义的 Stage 1→Stage 2 训练策略；本次只统一模型配置、数据输入、seed 和关闭自动类别平衡。既有模型不强行改成先验特征级联，避免同时改变“模型”和“持续学习方法”两个实验变量。

### 11.5 运行矩阵

```text
1 个数据集 × 2 个阶段 × 10 个模型 × 3 个 seed = 60 行
```

每个 seed 应产生 20 行。

### 11.6 输出格式

新增结果写入独立文件，例如：

```text
experiment/continuous_learning/continuous_baselines_rerun_results.csv
```

保持现有列：

```text
模型
数据集
seed
阶段
ACC
F1
Sensitivity
Specificity
status
error
out_dir
```

现有 `continuous_baseline_results.csv` 和 `continuous_hl_results.csv` 都不覆盖。普通 baseline 写入新的重跑结果文件；HL 不重跑，最终汇总直接读取既有 `continuous_hl_results.csv`。

## 12. YHD 切分可行性修正

当前 YHD 数据共有：

```text
负类：1286
正类：713
```

现有 contrast1/contrast2 脚本要求：

```text
validation：500 正 + 500 负
test：500 正 + 500 负
```

这会要求 1000 个不重复正例，超过当前 YHD 的 713 个正例，因此现有脚本无法在当前数据文件上完整运行 YHD。

推荐修正为按数据集定义 holdout：

| 数据集 | 验证集 | 测试集 | 每个集合类别比例 |
|---|---:|---:|---|
| UKB | 1000 | 1000 | 1:1 |
| YHD | 500 | 500 | 1:1 |

该方案为 YHD 使用 250 个正例做验证、250 个正例做测试，保留 213 个正例进入训练池。训练条件要求更多正例时，可按照原实验协议在训练数据构造阶段有放回采样。

验证集和测试集不得有放回采样，也不得重叠。如果必须保持 YHD 验证集和测试集各 1000 条，则应先提供至少 1000 个正例的数据版本，不能通过复用 holdout 行实现。

## 13. 指标定义

主指标与旧实验保持一致：

- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`

contrast2 额外保存：

- `TP`
- `FP`
- `FN`
- `TN`

统一规则：

- 正类固定为标签 1。
- 支持概率的模型统一使用正类概率 `>=0.5` 得到预测标签。
- 不在验证集或测试集上调分类阈值。
- CORELS 使用其硬分类输出。
- 所有指标统一调用 `hl.metrics.compute_metrics`。
- contrast2 中必须验证 `TP + FP + FN + TN == len(test_df)`。

三随机种子辅助汇总表报告每个条件的均值和标准差；原始逐 seed CSV 仍保持旧格式。只有三个 seed，不建议据此报告不稳定的显著性检验结论。

## 14. 模型产物和运行 manifest

每个任务使用独立目录：

```text
outputs_rerun/
└── <experiment>/
    └── seed<seed>/
        └── <dataset>/
            └── <condition>/
                └── <model>/
```

通用文件：

- `run_manifest.json`
- `resolved_model_config.json`
- `split_manifest.json`
- `metrics.json`
- `predictions.csv`，至少包含 source row ID、真实标签和预测标签
- `error.txt`，仅失败时生成

模型专属文件：

- DeepTab：`model.deeptab`、checkpoint/observability 摘要
- EBM：序列化模型、`best_iteration_`、全局项摘要
- APLR：序列化模型、参数和特征名
- CORELS：序列化模型、`predicate_manifest.json`、`rulelist.txt`
- continuous Stage 2：`continuation_manifest.json`

manifest 至少记录：

- Git commit（如果可获得）和工作区状态
- Python、NumPy、pandas、scikit-learn、torch、DeepTab、InterpretML、APLR、CORELS 版本
- CPU/GPU、精度和线程数
- 完整模型参数
- base seed 和派生 seed
- 训练前实际类别数量
- 自动平衡审计结果
- 训练耗时、最佳 epoch 和终止原因
- 模型保存/重载预测一致性结果

## 15. 输出保护和断点续跑

### 15.1 不覆盖旧结果

- 旧的 contrast1、contrast2、continuous baseline 和 HL 结果保持不动。
- 十个普通模型全部写入带 `rerun` 的独立结果文件。
- HL 不启动新任务；最终汇总读取既有 HL 结果文件。
- 旧自实现 FT-Transformer 结果可保留为历史记录，但正式全模型汇总时必须由 DeepTab FT-Transformer 替换，不能同时作为同名模型重复出现。

### 15.2 任务唯一键

contrast1：

```text
seed + model + dataset + train_size
```

contrast2：

```text
seed + model + dataset + train_total + ratio
```

continuous_learning：

```text
seed + model + dataset + stage
```

### 15.3 保存策略

- 每完成一个任务立即写入结果。
- 通过临时文件和 `os.replace` 原子更新 CSV。
- `--resume` 跳过已存在且 `status=ok/continued` 的任务。
- 失败任务保留错误，不影响其它任务。
- 只有显式 `--retry-errors` 才重试失败行。
- 排序顺序固定，保证重复汇总产生相同文件内容。

## 16. 预期运行规模

| 实验 | 每个 seed | 三个 seed | 计算方式 |
|---|---:|---:|---|
| contrast1 | 120 | 360 | 2 数据集 × 6 规模 × 10 模型 |
| contrast2 | 360 | 1080 | 2 数据集 × 2 总量 × 9 比例 × 10 模型 |
| continuous_learning | 20 | 60 | 1 数据集 × 2 阶段 × 10 模型 |
| 总计 | 500 | 1500 | 十个普通模型全部重跑，不含复用的 HL 行 |

其中 DeepTab FT-Transformer 和 ResNet 各需要 150 次拟合，两个 DeepTab 模型共 300 次拟合。当前环境没有可用 GPU，正式运行默认串行训练 DeepTab；EBM/APLR/CORELS 可有限并发，但各模型内部固定单线程或受控线程，避免进程和内部线程双重超额占用。

## 17. 测试与验收标准

### 17.1 依赖测试

- 四个新增包可以在 `uv run` 环境 import。
- 版本与 lockfile 一致。
- CORELS 可以完成官方 toy-data 拟合和预测。

### 17.2 统一建模层测试

- 十个普通模型均能在小型二分类 DataFrame 上拟合和预测。
- 预测长度、类型和标签集合正确。
- 支持概率的模型概率形状正确且在 `[0,1]` 范围内。
- 模型保存后重新加载，硬预测完全一致；概率在浮点容差内一致。

### 17.3 数据测试

- 同一条件的十个模型使用相同 source row IDs。
- train/val/test 不重叠。
- continuous 两阶段六个集合两两不重叠。
- 所有预处理参数只来自训练集。
- CORELS 的切点只来自训练特征。
- 每次 `fit` 前的正负样本数等于 manifest 目标。

### 17.4 自动平衡审计

- 旧模型统一配置不存在自动类别平衡。
- DeepTab `class_weight is None`。
- DeepTab 使用未加权 BCE。
- DeepTab DataLoader 不使用 `WeightedRandomSampler`。
- EBM/APLR 不接收 `sample_weight`。
- CORELS 不经过额外重采样器。

### 17.5 持续学习测试

- Stage 1 有 SIRS、无 SOFA。
- Stage 2 有 SOFA、无 SIRS。
- Stage 2 prior 生成不访问真实 SIRS。
- prior 由 Stage 1 已拟合模型产生。
- Stage 2 训练列显式包含 prior 特征。
- 所有映射按列名执行，不按数组位置执行。

### 17.6 结果测试

- contrast1 重跑结果恰好 360 行，三个 seed 各 120 行。
- contrast2 重跑结果恰好 1080 行，三个 seed 各 360 行。
- continuous_learning 重跑结果恰好 60 行，三个 seed 各 20 行。
- 原始 CSV 字段与旧实验一致。
- 所有成功行指标非空且在合法范围内。
- contrast2 混淆矩阵可复算全部四个指标。
- 旧结果文件未被修改或覆盖。

## 18. 建议实施顺序

1. 修改 `pyproject.toml` 和 `uv.lock`，执行 `uv sync --group dev`。
2. 完成依赖 import、CORELS 编译和 toy-fit 闸门。
3. 创建只有 `__init__.py`、`config.py`、`models.py`、`train_eval.py` 的扁平 `experiment/modeling/`。
4. 把十个普通模型的构造参数移入 `config.py`，并关闭所有模型层自动平衡。
5. 在 `models.py` 和 `train_eval.py` 中实现 DeepTab、InterpretML 和 CORELS 的构造与三个必要训练分支。
6. 实现数据 manifest、结果原子写入和断点续跑。
7. 重构 contrast1 和 contrast2 入口。
8. 实现 continuous prior-feature cascade。
9. 先运行最小压力测试：contrast1 训练量 10、contrast2 比例 1:50/50:1、continuous Stage 2。
10. 冻结配置后运行三组完整实验，seed 顺序固定为 36、40、42。
11. 校验 1500 行普通模型重跑结果和全部模型产物，再与既有 HL 结果汇总。
12. 更新各实验 README、根 README 和 `RUN_GUIDE.md`。

## 19. 计划运行命令

重构后的 CLI 建议支持模型组、多个 seed 和断点续跑：

```bash
uv run python experiment/contrast1/run_contrast1_balance.py \
  --models all \
  --seeds 36 40 42 \
  --resume
```

```bash
uv run python experiment/contrast2/run_contrast2.py \
  --models all \
  --seeds 36 40 42 \
  --resume
```

```bash
uv run python experiment/continuous_learning/run_continuous_learning_baselines.py \
  --models all \
  --seeds 36 40 42 \
  --resume
```

## 20. 需要明确保留的实验解释边界

- DeepTab 是本方案指定的 FT-Transformer/ResNet 软件实现，不应把它表述成当前脚本的自实现模型。
- continuous_learning 的 DeepTab Stage 2 是公开 API 下的先验特征级联，不应表述成原网络权重微调。
- CORELS 只有在搜索正常完成并确认相应终止状态时，才可表述为获得最优性证书。
- 训练集有放回采样属于预先定义的数据构造，不属于模型自动平衡；必须报告重复行和唯一行数量。
- YHD 使用缩小的无重复 holdout 时，应在最终报告中明确标注，不能声称其测试集大小与 UKB 相同。
- 所有新实验的随机种子固定为 `36、40、42`。
