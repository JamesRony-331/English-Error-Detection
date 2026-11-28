# 英语词级错误检测

基于 **PyTorch、GloVe 预训练词向量和三层空洞卷积神经网络（CNN）** 的英语错误检测学习项目。输入一句英文，模型为每个 token 预测“正确 / 疑似错误”，并显示错误概率及位置。

> 当前阶段：训练、评估、绘图和句子检测流程已实现，正在进行类别不平衡与过拟合优化。模型用于实验学习，尚不能保证检测准确；本项目定位为错误检测，暂不生成修改后的句子。

## 当前进度（2026-11-26）

| 状态 | 内容 |
| --- | --- |
| 已实现 | FCE TSV 读取、标签检查、训练集词表扩展、批次补齐 |
| 已实现 | 100 维预训练词向量 + 三层空洞 CNN 词级二分类模型 |
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
    → Conv1d（100 → 128，卷积核 7，dilation=1）→ ReLU → Dropout
    → Conv1d（128 → 128，卷积核 7，dilation=2）→ ReLU → Dropout
    → Conv1d（128 → 128，卷积核 7，dilation=4）→ ReLU → Dropout
    → Linear（128 → 2）
    → 每个 token 的正确 / 错误预测
```

标签 `0` 表示正确词，`1` 表示错误词。输入补齐 ID 为 `0`，标签补齐值为 `-100`。不使用池化，以保留逐词输出。每层后屏蔽 PAD 中间特征，避免补齐区域影响有效词。当前卷积核为 7、空洞率为 `[1, 2, 4]`，感受野为 43 个 token；卷积核改为 3 时为 15 个。

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

仓库仅提供代码与说明，不附带数据集、词向量、模型权重及运行结果。无需 Git LFS 下载项目数据；请自行准备下述资源。

```bash
git clone https://github.com/JamesRony-331/English-Error-Detection.git
cd English-Error-Detection
python -m pip install -r requirements.txt
```

默认数据目录如下，也可以修改配置中的路径：

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
epochs = 30
batch_size = 32
learning_rate = 0.001
class_weights = [1.0, 2.0] # [正确词, 错误词]

[model]
embedding_dim = 100
freeze_embedding = true
conv_channels = 128
kernel_size = 7
dilations = [1, 2, 4]
embedding_dropout = 0.2
conv_dropout = 0.3
```

`dilations` 的数组长度决定卷积层数，设为 `[1]` 可运行单层对照。三层模型需重新训练；检测器可以读取旧单层检查点，旧权重不会自动变成三层模型。

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

省略 `--text` 会进入交互模式，输入 `q` 退出。上述示例仅演示调用，不保证模型会识别其中错误。仓库附带历史检查点，可用于体验检测，也可以重新训练。历史结果对应各自目录中的配置，不能当作最新配置的统一实验结论。

重新绘图：

```bash
python util/draw.py result/train/YYYY-MM-DD/HH-MM-SS
```

## 已知局限与后续计划

- 历史训练存在过拟合，需要对冻结 / 微调词向量及正则化参数做受控对比；当前不宣称已经解决。
- 当前推理使用轻量规则分词，与训练数据的 RASP 分词可能不同。
- 未登录词映射到 UNK，大小写处理仍有优化空间。
- 当前空洞 CNN 仍仅建模有限范围的上下文，对需要长距离依赖的语法错误能力有限。
- 尚未实现早停、学习率调度和纠错建议；模型回归检查可运行 `python -m unittest discover -s tests -v`，完整端到端测试仍待补充。

## 图片英文识别（OCR）

新增 `OCR.py`，当前为 **macOS 本地 Apple Vision** 版本，无需 API Key。通过 PyObjC 调用系统识别模型；不上传图片到云端。依据 [Apple 官方文档](https://developer.apple.com/documentation/vision/recognizing-text-in-images)，使用 accurate 模式并指定英文。Windows/Linux 暂不支持此后端。

```bash
python -m pip install -r requirements-ocr.txt
python OCR.py                              # 运行附带的四张测试图片
python OCR.py /path/to/essay.jpg            # 单张图片
python OCR.py /path/to/images --output result/ocr/my_test
```

在本项目使用的 Conda 环境中，也可以将 `python` 替换为 `/opt/anaconda3/envs/ML/bin/python`。

默认结果目录为 `result/ocr/YYYY-MM-DD/HH-MM-SS-微秒/`，每张图片输出：

- `.txt`：按行排列的英文文本。
- `.json`：原始识别文本、每行置信度与像素坐标。
- `_boxes.png`：文字区域和行号预览，框不是语法错误标记。

支持常见 PNG/JPEG/BMP/TIFF/WebP 图片，目录批量识别只读取该目录直接包含的图片；多页 TIFF 仅处理首帧。按 EXIF 校正方向，坐标对应校正后的图像；面向单栏作文，暂不处理复杂多栏阅读顺序。

测试数据在 `data/ocr_samples/`，包含印刷体、短作文、带错误句子和合成手写字体图片，每张都有原文可对照。合成手写字体不能替代真人手写验证。运行 `python util/create_ocr_samples.py` 可在 macOS 上重新生成样例。

默认关闭语言纠正以尽量保留学生的原文错误；如需开启，添加 `--language-correction`。关闭后仍可能误读，低置信度不能完全筛出所有错误。当前 OCR 与 `detect.py` 独立，尚未自动将识别结果送入 CNN，建议先人工核对 OCR 文本。

## 网页上传与检测（Flask）

```bash
python -m pip install -r requirements-web.txt
python web_detect.py
# 指定模型或更换端口：
python web_detect.py --weights result/train/YYYY-MM-DD/HH-MM-SS/weights/best.pt --port 5001
```

浏览器访问 `http://127.0.0.1:5000`。选择单张图片后点击“识别并检测”，页面显示校正方向后的图片、OCR 原文、疑似错误词高亮及错误概率。默认在第一次请求时加载最新的 best.pt，之后缓存模型；新增训练结果后重启服务以更新默认模型。

- `GET /`：HTML 页面（`templates/detect.html`）。
- `POST /api/recognize`：multipart/form-data，图片字段名 `image`；返回 `text`、OCR `lines`、检测 `tokens`、预览图片 `image` 和模型路径 `model`。
- 词的 `start`/`end` 对应整篇 OCR 文本，按句子/换行分段调用现有 Detector。
- 单图最大 12 MB、2000 万像素。上传文件仅在内存处理，不永久保存。
- 若 OCR 成功但检测模型不可用，保留 OCR 结果，并返回 `detection_error`；未识别到文字返回提示。
- 当前仅用于本机，绑定 127.0.0.1，关闭调试模式。OCR 依赖 macOS。

测试图片：`data/ocr_samples/`。测试命令：`python -m unittest discover -s tests -v`。

### 摄像头拍照识别

网页点击“打开摄像头”，允许浏览器访问摄像头后，可预览画面并点击“拍照并识别”。拍照后自动关闭摄像头，复用 `/api/recognize` 接口展示 OCR 和错误检测结果；也可手动关闭摄像头。没有音频采集。

电脑摄像头可能因拍摄角度、透视变形或对焦问题产生识别误差，建议用手机正对作文拍照后上传。摄像头需要浏览器授权，使用本机 `http://127.0.0.1:5000` / localhost 或 HTTPS 访问；普通局域网 HTTP 地址通常不能调用摄像头。

## BERT 对比模型与评估

已接入 [sahilnishad/BERT-GED-FCE-FT](https://huggingface.co/sahilnishad/BERT-GED-FCE-FT)。本地文件在 `models/comparison/`，转换后的 `model.pt` 为纯 state_dict，需与该目录的配置、词表一起使用。

```bash
python -m pip install -r requirements-bert.txt
python compare_models.py
python web_detect.py
```

网页默认使用配置：

```toml
[detection]
backend = "bert"                 # 改成 cnn 可切回原模型
bert_dir = "models/comparison"
threshold = 0.30
```

修改后重启网页服务。`--weights` 对应所选后端的权重格式；不要把 BERT 的 model.pt 交给原 CNN Detector。命令行 `detect.py` 保留 CNN 用途，BERT 由 `bert_detect.py` 提供给网页与评估脚本。

`compare_models.py` 在同一 FCE 测试集、同一 30% 阈值下重新评估 BERT 和最新 CNN。结果保存到 `result/comparison/日期/时间/`，包含四张独立图：指标对比、两个模型各自的混淆矩阵、F0.5 随阈值变化曲线，以及指标 JSON、逐词概率 CSV 和阈值指标 CSV。阈值曲线仅用于描述，不能在测试集上挑阈值后声称得到无偏测试成绩。没有训练日志，不生成训练曲线。

BERT 的标签顺序与 CNN 相反：0=错误，1=正确；额外的第 2 个输出未对应有效训练标签。适配器已处理此差异，并对词表与权重进行严格检查。详细来源、对齐方法和评估局限见 `models/comparison/README.md`。

## 实测效果与模型准备

2026-11-27：FCE 测试集 2,720 句、41,477 Token，统一阈值 0.30。

|模型|Accuracy|Precision|Recall|F0.5|
|---|---:|---:|---:|---:|
|BERT|88.18%|71.03%|37.53%|0.6027|
|CNN|69.71%|24.50%|47.71%|0.2714|

CNN 检查点来自 2026-11-26/20-03-37。BERT 误报更少；CNN 召回更高但误报较多。同阈值不等于同校准水平。本地评估并非论文复现，外部模型训练数据与测试集重合关系没有独立核验。

历史类别加权 CNN 的 F0.5=0.3913，P=43.30%，R=28.25%；检查点和判定口径不同，不与上表直接排名。图片识别还受 OCR、角度与清晰度影响，上述指标不是端到端图片准确率。

运行前自行训练 CNN，或从上述 Hugging Face 页面下载 BERT 配套配置、词表和权重，使用 util/convert_bert_weights.py 转换。模型文件及 result 仅保留本地。

### 数据和词向量准备要求

1. 使用 **FCE-public for error detection** 的词级 TSV 版本（Rei 与 Yannakoudakis，2016），保留官方 train/dev/test 划分。不要将原始 XML 直接传给训练程序，也不要混合训练集和测试集。
2. 三个 TSV 文件采用 UTF-8 编码，每行一个词和标签，以制表符分隔；空行分句。标签仅使用 `c`（正确）或 `i`（错误）。文件名和目录见上面的目录树，可在 `config.toml` 的 `[paths]` 中修改。
3. 词向量使用 **GloVe Wikipedia 2014 + Gigaword 5，100 维**，例如 GloVe 6B 中的 `glove.6B.100d.txt`。当前加载器要求 word2vec 文本格式：首行 `词数 维度`，后续每行一个词及 100 个浮点数。标准 400,000 词版本的首行为 `400000 100`；其他词表须写实际词数。
4. 原始无头 GloVe 文件可用下列脚本转换（不修改原文件）。转换后设置 `paths.pretrained_vectors` 指向输出文件，`model.embedding_dim=100`。

```python
from pathlib import Path
source = Path("glove.6B.100d.txt")
target = Path("data/glove-wiki-gigaword-100")
count = 0
with source.open(encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if len(parts) != 101:
            raise ValueError("每行必须包含一个词和100维向量")
        for value in parts[1:]:
            float(value)
        count += 1
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", encoding="utf-8") as out:
    out.write(f"{count} 100\n")
    with source.open(encoding="utf-8") as f:
        for line in f:
            out.write(line)
```

数据与词向量请从其发布方获取，并遵守各自使用条款。训练运行 `python train.py`；网页默认使用 BERT，因此还需按“BERT 对比模型”章节准备本地模型，或切换 `detection.backend="cnn"` 并先训练 CNN。OCR 测试图片可通过 util 下的样例生成脚本在本地生成。
