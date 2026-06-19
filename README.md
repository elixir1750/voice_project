# 基于语音自监督表征的低资源 ASR 系统

> Low-Resource Automatic Speech Recognition via Self-Supervised Speech Representations

## 项目简介

本项目面向“基于语音自监督表征的低资源 ASR”课程大作业。系统使用预训练语音自监督模型提取语音表示，并通过可替换的 CTC Decoder 完成英文语音转写。

当前第一阶段实现并验证了以下真实链路：

```text
流式 LibriSpeech
  → 冻结的 wav2vec 2.0
  → 连续 hidden states
  → Linear / MLP CTC Decoder
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
| 训练与断点恢复 | ✅ 已完成 | 保存 `best.pt`、`last.pt` 和完整配置 |
| WER / CER / RTF | ✅ 已完成 | 训练验证和独立评估均会输出 |
| 单音频推理 | ✅ 已完成 | 支持 WAV/FLAC、自动重采样 |
| CUDA / MPS / CPU | ✅ 已完成 | `auto` 按 CUDA → MPS → CPU 选择 |
| HuBERT / WavLM | ⏳ 第二阶段 | 尚未实现 |
| k-means 离散单元 | ⏳ 第二阶段 | 尚未实现 |
| Transformer Decoder | ⏳ 第二阶段 | 尚未实现 |
| 批量消融实验 | ⏳ 第二阶段 | 尚未实现和运行 |

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
├── inference.py           # 单音频推理
├── environment.yml        # Conda 环境
├── requirements.txt       # Colab / pip 依赖
├── configs/
│   ├── quick_test.yaml    # 真实小样本闭环
│   └── baseline.yaml      # 低资源基线配置
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
├── outputs/               # 实验输出，不提交 Git
└── hf_cache/              # Hugging Face 缓存，不提交 Git
```

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

### 5. 单音频推理

```bash
python run.py transcribe \
  --checkpoint outputs/quick_test/best.pt \
  --audio path/to/audio.wav
```

支持 WAV 和 FLAC。输出包括转写文本、音频时长、推理耗时、RTF 和实际设备。

### 6. 断点续训

```bash
python run.py train \
  --config configs/baseline.yaml \
  --resume outputs/baseline_wav2vec2_mlp/last.pt
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

## 指标

| 指标 | 含义 |
| --- | --- |
| WER | Word Error Rate，词错误率 |
| CER | Character Error Rate，字符错误率 |
| RTF | 推理耗时 / 音频时长 |
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
