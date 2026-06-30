# 研究 C：SSL 表征层选择

本目录保存可追踪的研究 C 结果。原始训练输出仍位于 `outputs/`，不会提交到 Git；这里仅保留报告和复现讨论需要的轻量文件。

## 实验设置

- 数据：Hugging Face `librispeech_asr`，`clean/train.100` 抽取 3600 条训练样本，`validation` 抽取 500 条验证样本。
- SSL 模型：`facebook/wav2vec2-base`，冻结参数。
- 表征：连续 hidden states。
- Decoder：MLP CTC decoder。
- 训练：10 epoch，batch size 8。
- 变量：取用的 wav2vec2 hidden layer。

## 汇总结果

| 实验 | 层号 | 最佳 epoch | WER | CER | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hidden_layer_6` | 6 | 10 | 0.4138 | 0.1186 | 0.4604 |
| `hidden_layer_9` | 9 | 10 | 0.2954 | 0.0768 | 0.3088 |
| `baseline_mlp` | 12 | 10 | 0.8596 | 0.3426 | 1.1925 |

## 结论

在固定连续表征、MLP CTC decoder 和 3600 条训练样本的条件下，第 9 层 wav2vec2 表征取得最低 WER/CER。该结果支持“中间层 SSL 表征往往比最后一层更适合低资源 ASR”的预期。

后续研究 A / B / D 建议将 `ssl.layer: 9` 作为默认锚点。

## 文件说明

- `summary.csv`：每个实验的最佳 epoch、最终 epoch 和核心指标。
- `epoch_metrics.csv`：每个 epoch 的 validation WER、CER、loss、RTF 和词级错误分解。
- `wer_by_layer.png`：WER–层号曲线。
- `wer_by_epoch.png`：三个实验的 WER 训练曲线。
