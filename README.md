# 基于语音自监督表征的低资源 ASR 系统

> Low-Resource Automatic Speech Recognition via Self-Supervised Speech Representations

## 项目简介

本项目面向“基于语音自监督表征的低资源 ASR”课程大作业。系统使用预训练语音自监督模型提取语音表示，并通过可替换的 CTC Decoder 完成英文语音转写。

当前第一阶段实现并验证了以下真实链路：

```text
流式 LibriSpeech
  → 冻结的 wav2vec 2.0
  → 连续 hidden states
  → Linear / MLP / Transformer CTC Decoder
  → CTC Greedy Decode
  → WER / CER / RTF
```

项目采用统一接口和组件注册表。后续可以替换 SSL 特征提取器、表示层和 Decoder，而不需要改动训练主循环。

## 当前完成情况

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 流式 LibriSpeech | ✅ 已完成 | 支持有限样本、动态 padding、16 kHz 重采样 |
| wav2vec2 特征提取 | ✅ 已完成 | 支持冻结/解冻和 hidden layer 选择 |
| Linear CTC Decoder | ✅ 已完成 | 第一阶段默认 Decoder |
| MLP CTC Decoder | ✅ 已完成 | 可通过 YAML 配置替换 |
| Transformer CTC Decoder | ✅ 已完成 | 支持 padding mask、位置编码和 registry 配置 |
| 训练与断点恢复 | ✅ 已完成 | 支持梯度累积、余弦调度、Early Stopping 和 scheduler 恢复 |
| WER / CER / RTF | ✅ 已完成 | 额外报告替换、删除、插入和命中数量 |
| 单音频与批量推理 | ✅ 已完成 | 目录模式只加载一次模型，可保存 JSON |
| 批量消融实验 | ✅ 已完成 | YAML 实验矩阵，默认 dry-run，显式确认后训练 |
| 研究 C：SSL 表征层 | ✅ 已完成 | 第 9 层在当前设置下取得最低 WER/CER |
| 研究 A：Decoder 结构 | ✅ 已配置 | 以第 9 层为锚点，对比 Linear、MLP、Transformer |
| 研究 B 附加分析 | ✅ 已配置 | 免标签统计离散单元的熵、perplexity 和死簇比例 |
| CUDA / MPS / CPU | ✅ 已完成 | `auto` 按 CUDA → MPS → CPU 选择 |
| HuBERT / WavLM | ⏳ 第二阶段 | 尚未实现 |
| k-means 离散单元 | ⏳ 第二阶段 | 尚未实现 |

## 已验证的真实 quick test

验证日期：2026-06-19。

运行环境：

- Apple M4，PyTorch MPS；
- 真实 `facebook/wav2vec2-base`；
- Hugging Face 流式 LibriSpeech；
- 64 条训练样本、16 条验证样本；
- batch size 2，训练 1 epoch。

结果：

| 指标 | 数值 |
| --- | ---: |
| Train loss | 4.9636 |
| Validation loss | 3.0465 |
| WER | 1.0000 |
| CER | 1.0000 |
| Evaluation RTF | 约 0.022 |

训练、独立评估、checkpoint 恢复和真实单音频推理均已成功执行并正常退出。

上述 quick test 只用于验证工程链路。训练数据极少且仅训练 1 epoch，模型在测试音频上的 greedy CTC 输出为空字符串，WER/CER 为 100%；这些数值不能作为正式实验性能。正式报告需要增加训练数据和 epoch，并完成对比与消融。

## 目录结构

```text
voice_project/
├── config.py              # YAML 配置加载、校验和命令行覆盖
├── run.py                 # 统一命令入口
├── train.py               # ASR 组合模型与训练/验证循环
├── evaluate.py            # checkpoint 独立评估
├── inference.py           # 单音频与目录批量推理
├── run_experiments.py     # 安全的实验矩阵预览与执行
├── environment.yml        # Conda 环境
├── requirements.txt       # Colab / pip 依赖
├── analysis/
│   └── cluster_usage.py   # 离散单元码本利用率分析
├── configs/
│   ├── quick_test.yaml    # 真实小样本闭环
│   ├── baseline.yaml      # 低资源基线配置
│   └── experiments.yaml   # Decoder、层数和数据规模消融矩阵
├── data/
│   └── dataset.py         # 流式 LibriSpeech、预处理与 collator
├── models/
│   ├── interfaces.py      # 统一组件接口
│   ├── registry.py        # 组件注册表与工厂函数
│   ├── ssl_extractor.py   # wav2vec2 SSL 提取器
│   ├── representations.py # 连续表示适配器
│   └── ctc_decoder.py     # Linear / MLP CTC Decoder
├── utils/
│   ├── tokenizer.py       # 字符级 CTC Tokenizer
│   ├── device.py          # CUDA / MPS / CPU 选择
│   ├── metrics.py         # WER / CER / RTF
│   └── checkpoint.py      # checkpoint 保存与恢复
├── tests/                 # 不下载大模型的离线测试
├── results/               # 可追踪的轻量实验结果、表格和图片
├── outputs/               # 实验输出，不提交 Git
└── hf_cache/              # Hugging Face 缓存，不提交 Git
```

## 已追踪的研究结果

完整训练输出、checkpoint 和缓存默认写入 `outputs/`，不提交 Git。用于报告的轻量结果会整理到 `results/`，可以随仓库追踪。

### 研究 C：SSL 表征层选择

研究 C 固定连续表征、MLP CTC decoder、3600 条 LibriSpeech 训练样本和 500 条验证样本，只改变取用的 wav2vec2 hidden layer。

| 实验 | 层号 | 最佳 epoch | WER | CER | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hidden_layer_6` | 6 | 10 | 0.4138 | 0.1186 | 0.4604 |
| `hidden_layer_9` | 9 | 10 | 0.2954 | 0.0768 | 0.3088 |
| `baseline_mlp` | 12 | 10 | 0.8596 | 0.3426 | 1.1925 |

结论：第 9 层表征在当前设置下效果最好，后续研究 A / B / D 建议以 `ssl.layer: 9` 作为默认锚点。

可追踪文件见：

- `results/research_c/summary.csv`
- `results/research_c/epoch_metrics.csv`
- `results/research_c/wer_by_layer.png`
- `results/research_c/wer_by_epoch.png`

## 环境安装

### Conda（推荐）

```bash
conda env create -f environment.yml
conda activate low-resource-asr
```

验证环境：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

### Google Colab

Colab 已预装 PyTorch。克隆仓库后运行：

```bash
pip install -r requirements.txt
```

确认 GPU：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

训练代码会自动优先使用 CUDA。建议将 `runtime.output_dir` 指向 Google Drive，避免运行时断开后丢失 checkpoint。

## 快速开始

### 1. 运行真实 quick test

首次运行会下载 wav2vec2 和少量 LibriSpeech 数据：

```bash
python run.py train --config configs/quick_test.yaml
```

输出目录：

```text
outputs/quick_test/
├── best.pt
├── last.pt
├── config.yaml
└── epoch_1.json
```

流式读取统一使用 `num_workers: 1`：既让 Hugging Face/PyArrow 数据读取与主训练进程隔离，也避免多个 worker 重复消费同一份有限数据流。

PyTorch 当前没有原生 MPS CTC loss kernel。本项目会让 wav2vec2 和 Decoder 保持在 MPS 上运行，并仅将 CTC loss 计算回退到 CPU；CUDA 不需要该回退。

### 2. 运行基线

```bash
python run.py train --config configs/baseline.yaml
```

`baseline.yaml` 的训练量明显大于 quick test，请根据 GPU 显存和时间调整样本数、batch size 与 epoch。

### 3. 覆盖配置

无需修改 YAML，可使用多个 `--set`：

```bash
python run.py train \
  --config configs/quick_test.yaml \
  --set training.epochs=3 \
  --set runtime.device=cuda
```

支持的设备值：

- `auto`：CUDA → MPS → CPU；
- `cuda`：NVIDIA GPU；
- `mps`：Apple Silicon GPU；
- `cpu`：CPU。

若显式指定不可用设备，程序会直接报错，不会静默切换。

### 4. 独立评估

```bash
python run.py evaluate \
  --config configs/quick_test.yaml \
  --checkpoint outputs/quick_test/best.pt
```

输出 WER、CER、loss、RTF 和样本数量，并保存到：

```text
outputs/quick_test/evaluation.json
```

### 5. 单音频与批量推理

```bash
python run.py transcribe \
  --checkpoint outputs/quick_test/best.pt \
  --audio path/to/audio.wav
```

支持 WAV 和 FLAC。输出包括转写文本、音频时长、推理耗时、RTF 和实际设备。

也可以输入一个目录。模型和 checkpoint 只加载一次，目录中的
WAV、FLAC、MP3、OGG 文件按文件名顺序处理：

```bash
python run.py transcribe \
  --checkpoint outputs/quick_test/best.pt \
  --audio path/to/audio_dir \
  --output outputs/transcriptions.json
```

### 6. 断点续训

```bash
python run.py train \
  --config configs/baseline.yaml \
  --resume outputs/baseline_wav2vec2_mlp/last.pt
```

断点中会同时保存并恢复模型、优化器、scheduler 和 Early Stopping 状态。
每轮训练日志还会记录参数量：

- `parameters.total`：完整 ASR 模型参数量，包括冻结的 SSL 模型；
- `parameters.trainable`：实际参与训练的参数量；
- `parameters.decoder`：Decoder 自身参数量；
- `parameters.decoder_trainable`：Decoder 中实际参与训练的参数量。

### 7. 预览和运行实验矩阵

先预览，不会下载模型、读取数据或开始训练：

```bash
python run.py experiments --matrix configs/experiments.yaml
```

只预览指定实验：

```bash
python run.py experiments \
  --matrix configs/experiments.yaml \
  --name decoder_transformer
```

确认配置和输出目录无误后，显式加入 `--execute`：

```bash
python run.py experiments \
  --matrix configs/experiments.yaml \
  --name decoder_transformer \
  --execute
```

`configs/experiments.yaml` 当前包含 Linear、MLP、Transformer Decoder，
hidden layer 和训练样本规模对比。离散单元尚未可靠实现，因此实验加载器会
主动拒绝离散表示配置，避免生成名义上离散、实际上仍使用连续特征的无效结果。

研究 A 建议使用研究 C 得到的第 9 层作为锚点，而不是最后一层：

```bash
python run.py experiments \
  --matrix configs/experiments.yaml \
  --name decoder_linear_layer9 \
  --name decoder_mlp_layer9 \
  --name decoder_transformer_layer9 \
  --execute
```

这三组分别对应：

| 实验 | Decoder | 作用 |
| --- | --- | --- |
| `decoder_linear_layer9` | Linear | 检查 SSL 表征是否接近线性可分 |
| `decoder_mlp_layer9` | MLP | 衡量逐帧非线性带来的收益 |
| `decoder_transformer_layer9` | Transformer | 衡量跨帧上下文建模带来的收益 |

研究 A 的 layer9 实验默认设置了 `training.save_checkpoints: false`，
因此只保留 `config.yaml` 和每轮 `epoch_*.json`，不会生成较大的 `.pt`
checkpoint。报告所需的 WER、CER、RTF 和参数量都在 JSON 中。

### 8. 研究 B 附加分析：码本利用率

phone purity 需要帧级音素标签，LibriSpeech 默认没有这类标注；若不引入
Montreal Forced Aligner 等强制对齐工具，推荐先做免标签的码本利用率分析。

给定 k-means 产生的离散 token 序列，可以统计：

- entropy bits：簇使用分布熵；
- normalized entropy：除以 `log2(K)` 后的归一化熵；
- perplexity：等效活跃簇数；
- dead cluster ratio：死簇比例；
- most used frequency：最高频簇占比。

输入文件可以是 `.txt`、`.csv`、`.json`、`.npy` 或 `.npz`。文本文件中 token
可用空格、换行或逗号分隔；JSON 可以是嵌套 list。

```bash
python run.py cluster-usage \
  --assignments outputs/units_k128.txt outputs/units_k256.txt outputs/units_k512.txt \
  --codebook-size 128 \
  --codebook-size 256 \
  --codebook-size 512 \
  --output-dir results/research_b_cluster_usage
```

输出包括：

```text
results/research_b_cluster_usage/
├── cluster_usage_summary.json
├── cluster_usage_summary.csv
└── cluster_usage_by_k.svg
```

如果后续有离散 ASR 的下游结果，可以准备一个 CSV：

```csv
codebook_size,wer,cer
128,0.52,0.18
256,0.49,0.16
512,0.50,0.17
```

然后通过 `--downstream-metrics` 合并，便于分析“码本利用率”和 WER 是否一致：

```bash
python run.py cluster-usage \
  --assignments outputs/units_k128.txt outputs/units_k256.txt outputs/units_k512.txt \
  --codebook-size 128 \
  --codebook-size 256 \
  --codebook-size 512 \
  --downstream-metrics results/discrete_wer.csv \
  --output-dir results/research_b_cluster_usage
```

若希望把“几乎不用”的低频簇也视为死簇，可设置阈值。例如使用频率低于
0.1% 的簇都算死簇：

```bash
python run.py cluster-usage \
  --assignments outputs/units_k512.txt \
  --codebook-size 512 \
  --dead-min-frequency 0.001 \
  --output-dir results/research_b_cluster_usage_k512
```

## 配置说明

YAML 配置分为：

| 配置块 | 用途 |
| --- | --- |
| `experiment` | 实验名称与随机种子 |
| `data` | 数据集、split、样本数和采样率 |
| `tokenizer` | 字符级 CTC 词表 |
| `ssl` | SSL 类型、模型 ID、冻结策略和层 |
| `representation` | 连续或后续离散表示 |
| `decoder` | Decoder 类型与参数 |
| `training` | epoch、batch、学习率和 DataLoader |
| `runtime` | 设备、输出目录和缓存目录 |

训练控制字段：

| 字段 | 含义 |
| --- | --- |
| `gradient_accumulation_steps` | 累积多少个 batch 后更新一次参数；最后不足一组也会更新 |
| `scheduler` | `none` 或 `cosine` |
| `min_learning_rate` | 余弦调度最低学习率 |
| `early_stopping_patience` | 连续多少个 epoch 未改善 WER 后停止；`0` 表示关闭 |
| `save_checkpoints` | 是否保存 `best.pt` 和 `last.pt`；关闭后仍保存 `config.yaml` 和每轮 `epoch_*.json` |

## 模块替换方法

### 新增 SSL 特征提取器

1. 在 `models/ssl_extractor.py` 中继承 `SSLExtractor`；
2. 输入 padding 波形和原始长度；
3. 返回 `SpeechFeatures(values, lengths, feature_dim, metadata)`；
4. 使用 `@register_ssl("名称")` 注册；
5. 在 YAML 中设置 `ssl.type`。

### 新增表示层

1. 在 `models/representations.py` 中继承 `RepresentationAdapter`；
2. 输入和输出统一使用 `SpeechFeatures`；
3. 使用 `@register_representation("名称")` 注册。

后续 k-means 离散单元会放在这一层，不侵入 SSL 或训练循环。

### 新增 Decoder

1. 在 `models/ctc_decoder.py` 中继承 `CTCDecoder`；
2. 输入 `[batch, frames, input_dim]` 和有效长度；
3. 返回 `DecoderOutput(logits, lengths)`；
4. 使用 `@register_decoder("名称")` 注册；
5. 在 YAML 中设置 `decoder.type`。

Transformer 示例：

```yaml
decoder:
  type: transformer
  model_dim: 256
  num_heads: 4
  num_layers: 2
  feedforward_dim: 1024
  dropout: 0.1
```

`model_dim` 必须能被 `num_heads` 整除。

## 指标

| 指标 | 含义 |
| --- | --- |
| WER | Word Error Rate，词错误率 |
| CER | Character Error Rate，字符错误率 |
| RTF | 推理耗时 / 音频时长 |
| Substitutions / Deletions / Insertions | WER 的替换、删除和插入错误数量 |
| Hits | 正确识别的词数量 |
| Token Rate | 离散单元模式下每秒 token 数，第二阶段 |
| Bitrate | 离散表示信息率，第二阶段 |
| Codebook Size | 离散单元词表大小，第二阶段 |

## 测试

离线测试不会下载预训练模型：

```bash
pytest -q
```

测试覆盖配置、设备选择、Tokenizer、动态 padding、模型 shape、CTC loss、指标、checkpoint 和 CLI。

真实验收需要网络，并通过 `configs/quick_test.yaml` 完成。

## 分支与 PR 协作规范

### 基本规则

- 禁止直接向 `main` 推送日常修改；
- 开始任务前，从最新 `main` 创建独立分支；
- 一个分支只处理一项任务；
- 所有修改通过 Pull Request 合并；
- PR 必须通过至少 1 位成员审核、Codex 审核以及仓库启用的自动检查；
- 处理完有效审核意见后才能合并；
- 推荐使用 **Squash and merge**，合并后删除工作分支。

### 分支命名

| 类型 | 格式 | 示例 |
| --- | --- | --- |
| 新功能 | `feature/<名称>` | `feature/hubert-extractor` |
| Bug 修复 | `fix/<名称>` | `fix/ctc-length` |
| 文档 | `docs/<名称>` | `docs/colab-guide` |
| 配置或杂项 | `chore/<名称>` | `chore/update-deps` |

### 新手操作流程

同步 `main`：

```bash
git switch main
git pull origin main
```

创建分支：

```bash
git switch -c feature/example
```

修改后检查、暂存和提交：

```bash
git status
git diff
git add <文件名>
git diff --staged
git commit -m "feat: 简要说明修改"
```

首次推送：

```bash
git push -u origin feature/example
```

然后在 GitHub 页面点击 **Compare & pull request**，确认 base 是 `main`，compare 是工作分支。

PR 描述建议：

```markdown
## 修改内容

-

## 修改原因

-

## 验证方法

- [ ] 已运行：

## 需要重点审核的内容

- 无 / 请重点检查：
```

根据审核意见继续修改时，在原分支提交并执行：

```bash
git push
```

不需要重新创建 PR。审核和检查全部通过后使用 **Squash and merge**。

合并后清理本地分支：

```bash
git switch main
git pull origin main
git branch -d feature/example
git fetch --prune
```

## 小组成员

| 姓名 | 学号 |
| --- | --- |
| 待填写 | 待填写 |
| 待填写 | 待填写 |
| 待填写 | 待填写 |
