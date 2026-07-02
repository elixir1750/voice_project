# Kaggle Notebook 跑研究 D：5400 / 7200 样本

本文档用于在 Kaggle Notebook 上继续运行研究 D 的后两个数据规模实验：

- `train_samples_5400_layer9`
- `train_samples_7200_layer9`

研究 D 固定：

- SSL 模型：`facebook/wav2vec2-base`
- SSL 层：第 9 层
- 表征：continuous hidden states
- Decoder：MLP CTC decoder
- 验证集：500 条 LibriSpeech validation 样本

变量只有训练样本数。

## 1. 新建 Kaggle Notebook

在 Kaggle 新建 Notebook 后，右侧设置：

- Accelerator：选择 GPU，例如 T4 / P100；
- Internet：打开；
- Persistence：保持默认即可。

先运行：

```bash
!nvidia-smi
```

如果能看到 GPU 信息，说明硬件可用。

## 2. 拉取仓库和分支

```bash
%cd /kaggle/working
!git clone https://github.com/elixir1750/voice_project.git
%cd /kaggle/working/voice_project
!git checkout codex/research-c-experiments
```

确认分支：

```bash
!git branch --show-current
```

应该输出：

```text
codex/research-c-experiments
```

## 3. 安装依赖

```bash
!pip install -q -r requirements.txt
```

确认 PyTorch 能看到 GPU：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 4. 先 dry-run 检查实验名

这一步不会训练，只确认 Kaggle 拉到的代码里有这两个实验：

```bash
!python run.py experiments \
  --matrix configs/experiments.yaml \
  --name train_samples_5400_layer9 \
  --name train_samples_7200_layer9
```

如果看到两个实验的 JSON 预览，说明配置正确。

## 5. 正式训练

建议先一个一个跑，降低 Kaggle 会话中断后损失过大的风险。

### 5.1 跑 5400 样本

```bash
!python run.py experiments \
  --matrix configs/experiments.yaml \
  --name train_samples_5400_layer9 \
  --execute
```

### 5.2 跑 7200 样本

```bash
!python run.py experiments \
  --matrix configs/experiments.yaml \
  --name train_samples_7200_layer9 \
  --execute
```

如果想一次跑完，也可以：

```bash
!python run.py experiments \
  --matrix configs/experiments.yaml \
  --name train_samples_5400_layer9 \
  --name train_samples_7200_layer9 \
  --execute
```

## 6. 检查输出

训练结束后检查：

```bash
!find outputs/train_samples_5400_layer9 outputs/train_samples_7200_layer9 -maxdepth 1 -type f | sort
```

正常应看到每组：

```text
config.yaml
epoch_1.json
...
epoch_10.json
```

报告主要需要 `epoch_10.json` 或最佳 epoch 中的：

- WER
- CER
- loss
- RTF
- num_samples

## 7. 打包结果

只打包轻量结果，不需要 checkpoint：

```bash
!mkdir -p report_results
!for d in train_samples_5400_layer9 train_samples_7200_layer9; do \
  mkdir -p report_results/$d; \
  cp outputs/$d/*.json report_results/$d/; \
  cp outputs/$d/config.yaml report_results/$d/; \
done
!zip -r research_d_5400_7200_results.zip report_results
```

Kaggle Notebook 右侧的 Output 区域里会出现：

```text
research_d_5400_7200_results.zip
```

可以直接下载。

## 8. 快速查看最终指标

```bash
!python - <<'PY'
import json
from pathlib import Path

for name in ["train_samples_5400_layer9", "train_samples_7200_layer9"]:
    path = Path("outputs") / name / "epoch_10.json"
    data = json.loads(path.read_text())
    val = data["validation"]
    print(name)
    print("  WER:", val["wer"])
    print("  CER:", val["cer"])
    print("  loss:", val["loss"])
    print("  RTF:", val["rtf"])
PY
```

## 9. 常见问题

### Unknown experiments

如果出现：

```text
ValueError: Unknown experiments: train_samples_5400_layer9
```

说明分支不是最新。运行：

```bash
!git pull
```

或重新确认分支：

```bash
!git checkout codex/research-c-experiments
```

### Hugging Face 504 / 网络超时

这是 Hugging Face streaming 偶发网络问题。通常重新跑同一个实验即可。

如果频繁出现，可以改成一个实验一个实验跑。

### Kaggle 会话中断

当前研究 D 实验会保存 checkpoint，若输出目录里有 `last.pt`，可以续训：

```bash
!python run.py train \
  --config outputs/train_samples_5400_layer9/config.yaml \
  --resume outputs/train_samples_5400_layer9/last.pt
```

7200 对应：

```bash
!python run.py train \
  --config outputs/train_samples_7200_layer9/config.yaml \
  --resume outputs/train_samples_7200_layer9/last.pt
```

如果不想下载 checkpoint，最终打包时只复制 JSON 和 config 即可。

## 10. 研究 D 最终合并方式

已有或计划结果：

| 样本数 | 实验名 |
| ---: | --- |
| 900 | `train_samples_900_layer9` |
| 1800 | `train_samples_1800_layer9` |
| 3600 | 可复用 `hidden_layer_9` / `decoder_mlp_layer9` |
| 5400 | `train_samples_5400_layer9` |
| 7200 | `train_samples_7200_layer9` |

最后将五个点汇总到 `results/research_d/`，画 WER–训练样本数曲线。
