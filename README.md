# 英语词级错误检测

基于 **PyTorch、GloVe 预训练词向量和一维卷积神经网络（CNN）** 的英语错误检测学习项目。输入一句英文，模型为每个 token 预测“正确 / 疑似错误”，并显示错误概率及位置。

> 当前阶段：训练、评估、绘图和句子检测流程已实现，正在进行类别不平衡与过拟合优化。模型用于实验学习，尚不能保证检测准确；本项目定位为错误检测，暂不生成修改后的句子。

## 当前进度（2026-11-26）

| 状态 | 内容 |
| --- | --- |
| 已实现 | FCE TSV 读取、标签检查、训练集词表扩展、批次补齐 |
| 已实现 | 100 维预训练词向量 + CNN 词级二分类模型 |
| 已实现 | TOML 配置：数据路径、超参数、Dropout、类别权重、Embedding 冻结开关 |
| 已实现 | 独立 `evaluate.py`，共用训练集评估、验证和测试逻辑 |
| 已实现 | 评估使用 `model.eval()` 和无梯度模式；损失与指标排除 PAD |
| 已实现 | 加权交叉熵；训练和评估损失按有效标签权重之和汇总 |
| 已实现 | tqdm 进度条、逐轮 CSV、`best.pt` / `last.pt` 保存 |
| 已实现 | Matplotlib 独立指标曲线、测试混淆矩阵与指标图 |
| 已实现 | `detect.py` 单句及交互式检测 |
| 优化中 | 历史实验出现训练损失下降、验证损失上升；已加入 Dropout、类别权重及冻结 Embedding 配置，仍需实验比较泛化效果 |
| 待完成 | 当前配置的完整实验结论、分词与未登录词处理改进、早停及学习率策略对比 |

已做小样本检查：冻结后词向量不更新而卷积层正常更新；加权损失汇总、PAD 排除及不同评估批大小的结果一致性。**这些检查不代表当前配置已取得更好的测试成绩。**

## 模型结构

```text
英文 token → token ID
    → GloVe Embedding（100 维，可配置冻结）
    → Dropout（0.2）
    → Conv1d（100 → 128，卷积核 3，保持序列长度）
    → ReLU → Dropout（0.3）
    → Linear（128 → 2）
    → 每个 token 的正确 / 错误预测
```

标签 `0` 表示正确词，`1` 表示错误词。输入补齐 ID 为 `0`，标签补齐值为 `-100`。不使用池化，以保留逐词输出。卷积核为 3 时，每个位置使用相邻三个 token 的局部上下文。

## 文件说明

```text
.
├── config.toml       # 中文注释配置文件
├── train.py          # 数据加载、训练、保存检查点及训练后测试
├── model.py          # Embedding + CNN 模型
├── evaluate.py       # 损失、Accuracy、Precision、Recall、F0.5 和混淆矩阵
├── detect.py         # 输入英文句子，标记疑似错误词
├── requirements.txt  # Python 依赖
└── util/
    ├── TSV.py        # FCE TSV 读取
    ├── GloVe.py      # 预训练词向量加载和训练词表扩展
    ├── loss.py       # 加权交叉熵汇总分母
    └── draw.py       # 从训练结果绘制独立图表
```

仓库同时保留两份中文学习文档；文档记录阶段性实现，最新模型结构与参数以代码和 `config.toml` 为准。

## 环境与数据准备

建议 Python 3.11 及以上（使用标准库 `tomllib`）。本地曾在 Python 3.13、PyTorch 2.7、Matplotlib 3.10 环境检查运行。

```bash
python -m pip install -r requirements.txt
```

数据与大文件不随仓库发布，请自行准备并修改配置中的路径：

```text
data/
├── fce-error-detection/tsv/
│   ├── fce-public.train.original.tsv
│   ├── fce-public.dev.original.tsv
│   └── fce-public.test.original.tsv
└── glove-wiki-gigaword-100
```

FCE TSV 每行是 `token<TAB>c` 或 `token<TAB>i`，句子间以空行分隔。程序将 `c` 转为 0、`i` 转为 1。缺词错误按照该数据集约定标在后一个 token 上。

**词向量文件必须带有首行词数和维度**（当前文件首行为 `400000 100`），其余每行是单词与 100 个浮点数。原始不带首行的 GloVe 文本不能直接交给当前加载器，需要先转换为该格式。`embedding_dim` 必须与文件维度相同。

数据集出处：Rei 与 Yannakoudakis（ACL 2016），*Compositional Sequence Labeling Models for Error Detection in Learner Writing*；原始 FCE 数据见 Yannakoudakis、Briscoe 与 Medlock（ACL 2011），*A New Dataset and Method for Automatically Grading ESOL Texts*。使用时请遵守数据来源的使用条件并引用相关论文。

## 配置与运行

路径相对于配置文件所在目录解析，也支持绝对路径。当前关键配置：

```toml
[training]
epochs = 10
batch_size = 32
learning_rate = 0.001
class_weights = [1.0, 2.0] # [正确词, 错误词]

[model]
embedding_dim = 100
freeze_embedding = true
conv_channels = 128
kernel_size = 3
embedding_dropout = 0.2
conv_dropout = 0.3
```

`freeze_embedding = true` 冻结整个词向量层，包括为训练集中新增词初始化的向量；设为 `false` 即允许微调。`class_weights = [1.0, 1.0]` 等同普通交叉熵。错误词权重增加不保证 F0.5 提升，应同时观察 Precision 和 Recall。

在项目根目录运行训练：

```bash
python train.py --config config.toml
```

结果按北京时间保存到 `result/train/YYYY-MM-DD/HH-MM-SS/`，同秒运行自动添加后缀：

```text
一次训练目录/
├── config.toml
├── config.resolved.json
├── results.csv
├── test_metrics.json
├── weights/
│   ├── best.pt
│   └── last.pt
└── plots/
    ├── loss.png
    ├── accuracy.png
    ├── precision.png
    ├── recall.png
    ├── f05.png
    ├── epoch_duration.png
    ├── test_confusion_matrix.png
    └── test_metrics.png
```

每轮记录在线训练损失、固定模型的训练集评估指标、验证集指标、学习率和耗时。`best.pt` 按验证集 F0.5 选择；训练结束后加载最佳权重进行测试。损失曲线包含在线更新损失及本轮训练完成后的评估损失，两者计算时的模型状态不同。

句子检测（默认选结果目录中最近保存的 `best.pt`）：

```bash
python detect.py --text "She go to school every day."
python detect.py --weights result/train/YYYY-MM-DD/HH-MM-SS/weights/best.pt
```

省略 `--text` 会进入交互模式，输入 `q` 退出。上述示例仅演示调用，不保证模型会识别其中错误。首次使用需要先训练，或自行准备兼容检查点。

重新绘图：

```bash
python util/draw.py result/train/YYYY-MM-DD/HH-MM-SS
```

## 已知局限与后续计划

- 历史训练存在过拟合，需要对冻结 / 微调词向量及正则化参数做受控对比；当前不宣称已经解决。
- 当前推理使用轻量规则分词，与训练数据的 RASP 分词可能不同。
- 未登录词映射到 UNK，大小写处理仍有优化空间。
- 当前 CNN 仅建模局部上下文，对需要长距离依赖的语法错误能力有限。
- 尚未实现早停、学习率调度、纠错建议和自动化的完整回归测试套件。
