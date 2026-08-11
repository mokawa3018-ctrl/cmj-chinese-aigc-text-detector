# 中文 AIGC 文本检测模型复现、微调与 ONNX 导出

**Chinese AIGC Text Detector: Fine-tuning, Evaluation and ONNX Export**

本项目围绕已有中文 zh-v3 AIGC 文本分类模型，完成了复现评测、数据构建、多组微调实验、错误分析、ONNX 导出和 PyTorch/ONNX 一致性验证。

标签定义：

```text
0 = human
1 = AI
```

本项目不是从零预训练基础模型。训练模块 `src/training/` 基于 Apache-2.0 许可的上游项目 `YuchuanTian/AIGC_text_detector` 的项目使用版本整理而来，并保留了上游许可证与来源说明。

## 项目亮点

- 统一 PyTorch 与 ONNX 的 CSV 批量评测流程。
- 使用 `pair_id` 做问题级 train/validation 划分，降低问题泄漏风险。
- 构建多生成模型均衡训练方案。
- 对比 1:5 非均衡训练与 1:1 均衡训练。
- 按 `generator` 和 `source` 分析误检与漏检。
- 完成服务器环境下的 PyTorch/ONNX 一致性验证。
- 明确区分代码、数据、模型权重和实验结果的公开边界。

## 仓库结构

```text
aigc-text-detection-evaluation/
├─ configs/
│  └─ train_multimodel_v2_balance.example.json
├─ data/
│  └─ samples/
├─ docs/
├─ models/
├─ results/
│  └─ summary_metrics.csv
├─ src/
│  ├─ training/
│  ├─ check_data.py
│  ├─ check_onnx.py
│  ├─ export_onnx.py
│  ├─ predict_csv.py
│  ├─ predict_csv_onnx.py
│  ├─ evaluation.py
│  ├─ analyze_errors.py
│  └─ train_from_config.py
├─ tests/
├─ LICENSE
├─ THIRD_PARTY_NOTICES.md
├─ requirements.txt
└─ README.md
```

说明：

- `src/training/` 是基于上游 Apache-2.0 代码修改版整理出的真实训练链路。
- `src/train_from_config.py` 将 JSON 实验配置转换为真实训练命令，支持 `--dry-run`。
- `data/samples/` 只包含全新合成示例数据，不包含真实训练集。
- `results/` 只包含聚合指标，不包含逐条预测结果。
- 模型权重和 ONNX 文件不进入普通 Git 仓库。

## 环境安装

```bash
git clone https://github.com/mokawa3018-ctrl/cmj-chinese-aigc-text-detector.git
cd cmj-chinese-aigc-text-detector
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据格式与检查

训练和验证 CSV 采用以下字段：

```text
pair_id
sample_id
source_id
question
answer
label
generator
source
target_char_count
dataset_role
```

模型实际读取的文本列是 `answer`。`label=0` 表示 human，`label=1` 表示 AI。

检查合成示例数据：

```bash
python src/check_data.py \
  --train data/samples/sample_train.csv \
  --validation data/samples/sample_validation.csv
```

Windows PowerShell 单行命令：

```powershell
python src/check_data.py --train data/samples/sample_train.csv --validation data/samples/sample_validation.csv
```

`data/samples/` 中的 CSV 是全新合成数据，只用于演示字段结构、数据校验和最小流程冒烟测试，不能复现实验指标。

## 轻量 CI 与本地测试

本仓库包含轻量自动化测试，覆盖数据检查、配置启动器、公共二分类指标、CSV 推理输入校验、错误分析，以及 ONNX 导出参数和一致性比较逻辑。CI 不安装深度学习完整依赖，不下载模型或数据，也不会执行真实训练、真实模型推理或实际 ONNX 导出。

Windows PowerShell:

```powershell
python -m pip install -r requirements-ci.txt
python -m compileall src tests
ruff check src/check_data.py src/train_from_config.py src/evaluation.py src/analyze_errors.py src/predict_csv.py src/predict_csv_onnx.py src/predict_text.py src/export_onnx.py src/check_onnx.py tests
python -m pytest -q
```

Linux/macOS:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall src tests
ruff check src/check_data.py src/train_from_config.py src/evaluation.py src/analyze_errors.py src/predict_csv.py src/predict_csv_onnx.py src/predict_text.py src/export_onnx.py src/check_onnx.py tests
python -m pytest -q
```

## 单条文本预测

`src/predict_text.py` 使用 Hugging Face PyTorch 模型对单条文本进行二分类预测。

```bash
python src/predict_text.py \
  --model-path models/cmj-chinese-aigc-text-detector \
  --text "这是一段需要检测的中文文本。" \
  --device auto
```

输出字段：

- `predicted_label`：预测标签，`0=human`，`1=AI`
- `human_probability`：人工文本概率
- `ai_probability`：AI 文本概率

`--device` 支持 `auto`、`cpu` 和 `cuda`。如果本机没有 CUDA，建议使用 `auto` 或 `cpu`。

## PyTorch 批量预测

`src/predict_csv.py` 使用 Hugging Face PyTorch 模型目录进行 CSV 批量预测。

示例命令：

```bash
python src/predict_csv.py \
  --model-dir models/base_or_finetuned_model \
  --input data/private/eval.csv \
  --output results/predictions/pytorch_eval.csv \
  --text-column answer \
  --label-column label \
  --max-length 512 \
  --batch-size 16 \
  --device cuda \
  --group-columns generator,source
```

主要输出：

- `pred_label`
- `prob_human`
- `prob_ai`
- 若存在标签列，则输出 `correct`
- 若存在标签列，则计算整体准确率、人工误检率、AI 召回率
- 若同时存在标签列和 `generator` 列，则输出按 `generator` 聚合统计
- 若同时存在标签列和 `source` 列，则输出按 `source` 聚合统计

没有 `label` 列的 CSV 也可以预测；此时脚本只保存预测标签和概率，不计算监督指标。

## ONNX 批量预测

`src/predict_csv_onnx.py` 使用 ONNX 模型目录进行 CSV 批量预测。

示例命令：

```bash
python src/predict_csv_onnx.py \
  --onnx-dir models/onnx_model \
  --input data/private/eval.csv \
  --output results/predictions/onnx_eval.csv \
  --text-column answer \
  --label-column label \
  --group-columns generator,source \
  --batch-size 16 \
  --max-length 512
```

该脚本默认读取 `answer`，也可通过 `--text-column` 指定其他文本列。`label` 列可选：存在时计算准确率、人工误检率、AI 召回率，并在指定分组列存在时输出分组统计；不存在时只保存预测标签和概率。

## 自动错误分析

`src/analyze_errors.py` 读取已生成的预测 CSV，不加载模型。它会保存整体指标、按 `generator` 和 `source` 的指标、固定文本长度段指标，以及按置信度排序的误检和漏检样本。

```bash
python src/analyze_errors.py \
  --input results/predictions/eval.csv \
  --output-dir results/error_analysis \
  --label-column label \
  --prediction-column pred_label \
  --probability-column prob_ai \
  --text-column answer \
  --group-columns generator,source
```

输出包括：`overall_metrics.csv`、`metrics_by_generator.csv`、`metrics_by_source.csv`、`metrics_by_length.csv`、`false_positives.csv` 和 `false_negatives.csv`。长度段固定为 0-50、51-100、101-200、201-500 和 500+ 字。无标签 CSV 可以用于批量预测，但不能用于此监督错误分析。

## 模型训练

真实实验训练链路整理在 `src/training/`。训练模块本身使用命令行参数；`src/train_from_config.py` 可以读取 JSON 配置并转换为真实训练参数。

`configs/train_multimodel_v2_balance.example.json` 用于记录最终 1:1 均衡实验参数。先用 `--dry-run` 预览命令，不会启动训练：

```bash
python src/train_from_config.py --config configs/train_multimodel_v2_balance.example.json --dry-run
```

Windows PowerShell 单行：

```powershell
python src/train_from_config.py --config configs/train_multimodel_v2_balance.example.json --dry-run
```

公开路径示例命令：

```bash
python src/training/train.py \
  --device cuda \
  --max-epochs 1 \
  --batch-size 16 \
  --val-batch-size 8 \
  --max-sequence-length 512 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --seed 0 \
  --local-model models/base \
  --model-name AIGC_detector_zhv3 \
  --local-data . \
  --train-data-file data/private/train.csv \
  --val-data-file data/private/validation.csv \
  --val_file1 data/private/validation.csv \
  --data-name save \
  --mode original_single \
  --aug_min_length 0 \
  --lamb 0.4 \
  --pu_type dual_softmax_dyn_dtrun \
  --prior 0.2 \
  --len_thres 55 \
  --clean 0 \
  --quick_val 0 \
  --log-dir outputs/multimodel_v2_balance
```

基础模型和完整训练数据需要使用者自行准备，并遵守对应许可证和数据授权要求。公开合成示例只适合做格式检查，不适合正式训练。

## ONNX 导出与一致性验证

导出 ONNX：

```bash
python src/export_onnx.py \
  --model-dir models/finetuned_pytorch_model \
  --output-dir models/onnx_model \
  --max-length 512
```

对比 PyTorch 与 ONNX 输出：

```bash
python src/check_onnx.py \
  --pt-model-dir models/finetuned_pytorch_model \
  --onnx-dir models/onnx_model \
  --max-length 512
```

PyTorch 与 ONNX 一致性验证已在原服务器环境完成。本地发布整理环境未重新加载 ONNX 图，也未重复执行推理。ONNX 输入名称、动态轴和 opset 信息来自实际导出脚本配置。

服务器环境记录的 ONNX 导出设置：

```text
inputs: input_ids, attention_mask, token_type_ids
output: logits
opset: 14
dynamic batch: enabled
dynamic sequence length: enabled
```

## 实验结果

### 固定测试集：原始模型 vs 最终模型

| 模型 | 人工文本准确率 | 人工误检率 | AI 召回率 | 混合测试准确率 |
|---|---:|---:|---:|---:|
| 原始 zh-v3 | 89.10% | 10.90% | 90.50% | 89.80% |
| 最终 1:1 均衡模型 | 91.60% | 8.40% | 94.10% | 92.85% |

最终模型在固定混合测试上达到：

```text
固定混合测试准确率：92.85%
人工误检率：8.40%
AI召回率：94.10%
```

### 1:5 与 1:1 训练方案对比

| 模型 | 验证集 F1 | 外部固定人工误检率 | 外部固定 AI 召回率 | 外部固定混合准确率 |
|---|---:|---:|---:|---:|
| 多模型 1:5 非均衡 | 97.05% | 15.50% | 97.30% | 90.90% |
| 多模型 1:1 均衡 | 89.74% | 8.40% | 94.10% | 92.85% |

验证集指标用于训练选择，外部固定测试集用于评估泛化表现，二者不能直接混为同一类指标。1:5 模型验证 F1 更高，但人工误检率上升明显；最终选择 1:1 模型，是因为它在人工误检、AI 召回和整体准确率之间更均衡。

### HC3 中文 7696 条测试

数据组成：

```text
人工文本：4481 条
AI 文本：3215 条
总计：7696 条
```

| 模型 | 整体准确率 | AI 检出率 | 人工误检率 |
|---|---:|---:|---:|
| 原始模型 | 89.54% | 91.14% | 11.60% |
| 最终 1:1 均衡模型 | 91.84% | 92.78% | 8.84% |

变化：

```text
整体准确率：+2.30 个百分点
AI检出率：+1.64 个百分点
人工误检率：-2.76 个百分点
AI正确检出增加：53 条
人工误检约减少：124 条
```

`约减少 124 条` 是根据四舍五入后的比例反推，只作为近似值。

### ONNX 一致性

| 指标 | 数值 |
|---|---:|
| 最大 logits 绝对误差 | 1.43e-6 |
| 最大概率绝对误差 | 2.31e-7 |
| 测试样本预测标签一致性 | 100% |

ONNX 固定测试结果：

| 模型 | 人工文本准确率 | 人工误检率 | AI 召回率 | 混合测试准确率 |
|---|---:|---:|---:|---:|
| ONNX 导出模型 | 91.60% | 8.40% | 94.10% | 92.85% |

## 错误分析

`nlpcc_dbqa` 是当前错误分析中表现最弱的来源类别：

```text
整体准确率约：76.55%
人工误检率约：32.38%
AI召回率约：81.83%
```

法律和金融类别仍有改进空间，但这里不补充未公开汇总中没有依据的具体类别指标。

后续方向包括：

- 增加 `nlpcc_dbqa`、法律和金融类别的高质量配对样本。
- 分析文本长度影响。
- 分析不同生成模型影响。
- 分析数据来源分布偏差。
- 使用独立域外测试集验证泛化能力。

## 模型下载

模型权重与 ONNX 版本已发布至 Hugging Face 和 ModelScope：

```text
Hugging Face:
https://huggingface.co/mokawa3018/cmj-chinese-aigc-text-detector

ModelScope:
https://modelscope.cn/models/mokawa3018/cmj-chinese-aigc-text-detector
```

从 Hugging Face 下载完整模型：

```bash
hf download mokawa3018/cmj-chinese-aigc-text-detector \
  --local-dir models/cmj-chinese-aigc-text-detector
```

从 ModelScope 下载完整模型：

```bash
ms-hub download mokawa3018/cmj-chinese-aigc-text-detector \
  --repo-type model \
  --local-dir models/cmj-chinese-aigc-text-detector
```

PyTorch 加载：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_id = "mokawa3018/cmj-chinese-aigc-text-detector"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
```

说明：

- PyTorch 权重位于模型仓库根目录。
- ONNX 模型位于 `onnx/model.onnx`。
- 标签定义为 `label 0 = human`，`label 1 = AI`。
- 模型文件不直接进入 GitHub 普通 Git 仓库。
- 使用前请阅读模型卡中的许可证、数据来源、局限性和使用边界说明。

模型文件不会直接提交到普通 Git 仓库。`.gitignore` 已排除常见权重文件和本地发布候选目录。

## 文档导航

- [训练流程](docs/training_workflow.md)
- [训练代码修改说明](docs/training_code_changes.md)
- [实验总结](docs/experiment_summary.md)
- [许可证说明](docs/license_notes.md)
- [示例数据说明](data/samples/README.md)
- [结果说明](results/README.md)
- [第三方来源说明](THIRD_PARTY_NOTICES.md)

## 局限性与使用边界

AIGC 检测不是事实鉴定器。模型输出只能作为辅助信号。

对不同领域、文本长度、写作风格和未知生成模型，检测结果可能存在偏差。本文档中的指标只代表当前数据集、划分和模型版本。

不应将检测结果作为处罚、学术不端判断、身份识别或其他高影响决策的唯一依据。

## 开源与致谢

训练代码来源于 `YuchuanTian/AIGC_text_detector`，许可证为 Apache License 2.0。本仓库中的 `src/training/` 是基于上游代码的项目使用版本整理而来，原作者归属和许可证声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [LICENSE](LICENSE)。

基础模型来源包括官方中文 zh-v3 AIGC detector 和 Chinese RoBERTa WWM。HC3 中文数据集用于公开数据来源下的评估汇总。

本项目工作的重点在于：数据构建与清洗、问题级划分、批量评测流程、微调方案对比、错误分析、ONNX 导出和发布边界整理。项目不声称训练框架或基础模型完全从零编写。
