# 研究 B：Colab 运行说明

这份文档按 Colab notebook 的 cell 来写，可以直接复制运行。

研究 B 比较：

```text
连续 wav2vec2 hidden states
vs.
k-means 离散 token → trainable embedding → MLP CTC decoder
```

当前固定：

- 分支：`codex/research-c-experiments`
- SSL 模型：`facebook/wav2vec2-base`
- 表征层：第 9 层
- 训练数据：LibriSpeech `train.100` 的前 3600 条样本
- 验证数据：LibriSpeech `validation` 的前 500 条样本
- decoder：MLP CTC decoder
- 训练轮数：10 epoch
- 离散实验：`kmeans_k100`、`kmeans_k500`、`kmeans_k1000`
- 选做实验：`kmeans_k500_dedup`

## 0. Colab 设置

先在 Colab 菜单里选：

```text
运行时 → 更改运行时类型 → T4 GPU
```

不要选 TPU。当前代码是 PyTorch + CUDA 路径。

## 1. 拉取正确分支

新建一个 Colab cell，运行：

```python
%cd /content
![ -d voice_project/.git ] || git clone -b codex/research-c-experiments --single-branch https://github.com/elixir1750/voice_project.git
%cd /content/voice_project
!git fetch origin codex/research-c-experiments
!git checkout codex/research-c-experiments
!git pull origin codex/research-c-experiments
```

检查当前分支：

```python
!git branch --show-current
```

应该输出：

```text
codex/research-c-experiments
```

## 2. 安装依赖

Colab 通常已经自带 CUDA 版 `torch` / `torchaudio`，所以这里不重装它们，只安装项目额外依赖：

```python
!apt-get -qq update
!apt-get -qq install -y ffmpeg libsndfile1
!grep -v -E '^(torch|torchaudio)' requirements.txt > /tmp/voice_project_colab_requirements.txt
!pip install -q -r /tmp/voice_project_colab_requirements.txt
```

检查 GPU：

```python
!python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

如果 `cuda: False`，说明 Colab 没切到 GPU，需要回到第 0 步。

## 3. 可选：挂载 Google Drive

如果你担心 Colab 断开，建议挂载 Drive，用来保存结果 zip 和 k-means codebook 备份：

```python
from google.colab import drive
drive.mount("/content/drive")
```

创建保存目录：

```python
!mkdir -p /content/drive/MyDrive/voice_project_outputs
```

## 4. 拟合 k-means codebook

离散实验必须先生成 codebook。这里只使用训练 split 的第 9 层 hidden states，不使用 validation/test，避免信息泄露。

正式实验建议抽 50 万帧：

```python
!python run.py fit-kmeans \
  --config configs/baseline.yaml \
  --set ssl.layer=9 \
  --codebook-size 100 \
  --codebook-size 500 \
  --codebook-size 1000 \
  --frame-sample-limit 500000 \
  --output-dir artifacts/kmeans
```

检查文件：

```python
!find artifacts/kmeans -maxdepth 1 -type f | sort
```

应该看到：

```text
artifacts/kmeans/wav2vec2_layer9_k100.pkl
artifacts/kmeans/wav2vec2_layer9_k500.pkl
artifacts/kmeans/wav2vec2_layer9_k1000.pkl
```

如果已经挂载 Drive，建议备份一份：

```python
!mkdir -p /content/drive/MyDrive/voice_project_outputs/artifacts
!cp -r artifacts/kmeans /content/drive/MyDrive/voice_project_outputs/artifacts/
```

如果 Colab 断开后重新开始，可以把 Drive 里的 codebook 拷回来，不必重新 fit：

```python
!mkdir -p artifacts
!cp -r /content/drive/MyDrive/voice_project_outputs/artifacts/kmeans artifacts/
```

## 5. 跑连续基线 B0

B0 连续基线可以复用之前研究 C/A 的第 9 层 MLP 结果。如果需要重新跑：

```python
!python run.py experiments \
  --matrix configs/experiments.yaml \
  --name decoder_mlp_layer9 \
  --execute
```

输出目录：

```text
outputs/decoder_mlp_layer9/
```

## 6. 跑离散主实验 B1/B2/B3

一次性跑 K=100、500、1000：

```python
!python run.py experiments \
  --matrix configs/experiments.yaml \
  --name kmeans_k100 \
  --name kmeans_k500 \
  --name kmeans_k1000 \
  --execute
```

如果时间不够，分开跑也可以：

```python
!python run.py experiments --matrix configs/experiments.yaml --name kmeans_k100 --execute
!python run.py experiments --matrix configs/experiments.yaml --name kmeans_k500 --execute
!python run.py experiments --matrix configs/experiments.yaml --name kmeans_k1000 --execute
```

当前这三个实验都在 `configs/experiments.yaml` 中固定为 10 epoch。

## 7. 可选：跑 dedup 加分实验

`kmeans_k500_dedup` 会合并连续相同 token，用来观察 token rate / bitrate 下降后 WER 是否明显变差：

```python
!python run.py experiments \
  --matrix configs/experiments.yaml \
  --name kmeans_k500_dedup \
  --execute
```

## 8. 打包下载结果

只打包 `epoch_*.json` 和 `config.yaml`，不打包 checkpoint：

```bash
%%bash
mkdir -p report_results/research_b

for d in decoder_mlp_layer9 kmeans_k100 kmeans_k500 kmeans_k1000 kmeans_k500_dedup; do
  if [ -d "outputs/$d" ]; then
    mkdir -p "report_results/research_b/$d"
    cp outputs/$d/*.json "report_results/research_b/$d/"
    cp outputs/$d/config.yaml "report_results/research_b/$d/"
  fi
done

zip -r research_b_results.zip report_results/research_b
```

下载到本机：

```python
from google.colab import files
files.download("research_b_results.zip")
```

或者保存到 Google Drive：

```python
!cp research_b_results.zip /content/drive/MyDrive/voice_project_outputs/
```

## 9. 快速检查结果

查看每个实验最后一轮的 WER/CER：

```python
!python - <<'PY'
import json
from pathlib import Path

for name in ["decoder_mlp_layer9", "kmeans_k100", "kmeans_k500", "kmeans_k1000", "kmeans_k500_dedup"]:
    path = Path("outputs") / name / "epoch_10.json"
    if not path.exists():
        print(name, "not found")
        continue
    data = json.loads(path.read_text())
    val = data["validation"]
    print(
        name,
        "WER=", round(val["wer"], 4),
        "CER=", round(val["cer"], 4),
        "token_rate=", round(val.get("token_rate", 0), 2),
        "bitrate=", round(val.get("bitrate", 0), 2),
        "K=", val.get("codebook_size"),
    )
PY
```

## 10. 常见问题

### `Unknown experiments: kmeans_k100`

说明 Colab 拉到的不是 `codex/research-c-experiments` 分支，或者分支还没更新。重新运行第 1 节。

### `No such file or directory: artifacts/kmeans/...pkl`

说明还没有 fit k-means，或者 Colab 断开后本地 `artifacts/` 消失了。重新运行第 4 节，或者从 Drive 拷回：

```python
!mkdir -p artifacts
!cp -r /content/drive/MyDrive/voice_project_outputs/artifacts/kmeans artifacts/
```

### `cuda: False`

说明没有启用 GPU。去 Colab 菜单里选择 T4 GPU，然后重新运行 notebook。

### fit k-means 太慢

可以先用小帧数做 smoke test：

```python
!python run.py fit-kmeans \
  --config configs/baseline.yaml \
  --set ssl.layer=9 \
  --codebook-size 100 \
  --frame-sample-limit 20000 \
  --output-dir artifacts/kmeans_smoke
```

正式结果仍建议用 50 万帧。

## 11. 报告要记录的指标

每个实验的 `epoch_10.json` 里重点看：

- `wer`
- `cer`
- `loss`
- `rtf`
- `token_rate`
- `bitrate`
- `codebook_size`

建议报告图：

- WER / CER 对比柱状图
- bitrate–WER 权衡曲线
- K=100/500/1000 的 codebook size–WER 曲线

解释口径：

- 连续 B0 是性能上界参考，因为它保留完整 hidden states。
- K 越大，离散表示容量越强，理论上 WER 更可能接近连续表征，但 bitrate 也会上升。
- 如果 dedup 后 token rate / bitrate 明显下降而 WER 变化不大，说明连续重复 token 中存在冗余。
