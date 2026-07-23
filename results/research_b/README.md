# 研究 B：连续 vs 离散表示

本目录保存研究 B 的结果。实验固定 wav2vec2 第 9 层、3600 条训练样本、500 条验证样本和 MLP CTC decoder，对比连续 hidden states 与 k-means 离散单元。

连续基线 `continuous_layer9_mlp` 复用研究 A/C 的第 9 层 MLP 结果；其 `token_rate` 和 `bitrate` 是用同一验证集上的 SSL 帧率估算得到，因为原始连续实验早于 token-rate 日志加入。离散实验的 `token_rate`、`bitrate` 和 `codebook_size` 来自对应验证 JSON。

## 文件结构

```text
results/research_b/
├── continuous_layer9_mlp/
├── kmeans_k100/
├── kmeans_k500/
├── kmeans_k1000/
├── kmeans_k500_dedup/
├── summary.csv
├── epoch_metrics.csv
├── wer_by_representation.svg
└── bitrate_wer_tradeoff.svg
```

## 汇总

| 实验 | 表示 | K | dedup | 最佳 epoch | WER | CER | Loss | token rate | bitrate |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuous_layer9_mlp` | continuous | - | - | 10 | 0.2954 | 0.0768 | 0.3088 | 49.90/s | 1226224.2 bit/s |
| `kmeans_k100` | k-means | 100 | 否 | 3 | 1.0000 | 0.8142 | 2.0766 | 49.90/s | 331.5 bit/s |
| `kmeans_k500` | k-means | 500 | 否 | 1 | 0.9811 | 0.5312 | 1.5957 | 49.90/s | 447.3 bit/s |
| `kmeans_k1000` | k-means | 1000 | 否 | 2 | 0.9564 | 0.4558 | 1.4217 | 49.90/s | 497.2 bit/s |
| `kmeans_k500_dedup` | k-means | 500 | 是 | 1 | 0.9710 | 0.4692 | 1.6234 | 35.70/s | 320.1 bit/s |

## 结果解读

- 连续表征明显最好：`continuous_layer9_mlp` 的 WER 为 0.2954，而所有 k-means 离散实验的 WER 都在 0.9564 以上。说明当前离散路径在 10 epoch 预算下远未接近连续 hidden states。
- K 增大带来一定改善：K=100 的 WER 为 1.0000，K=500 为 0.9811，K=1000 为 0.9564。趋势符合“更大码本保留更多信息”的预期，但改善幅度仍不足以弥合与连续表征的差距。
- dedup 显著降低 token rate / bitrate：K=500 非 dedup 的 token rate 为 49.90/s、bitrate 为 447.3 bit/s；dedup 后 token rate 降到 35.70/s、bitrate 降到 320.1 bit/s。WER 从 0.9811 到 0.9710，没有明显变差，说明连续重复 token 中存在冗余。
- 需要谨慎解释：离散系统多了随机初始化的 embedding 表，且本组实验都被 early stopping 提前停止在 4–5 epoch，可能存在明显欠训。因此报告中不宜写“离散表示本质不可用”，更合理的表述是“在当前 10 epoch / early-stopping / 简单 embedding-MLP 设置下，离散表示性能显著落后；后续可通过更长训练、预训练 embedding、BPE/dedup、时长建模或更强 decoder 改善”。

## 可直接写进报告的一句话

在第 9 层 wav2vec2 表征上，连续 hidden states 的 ASR 性能显著优于 k-means 离散单元；增大 codebook size 能降低 WER，但在当前低资源和 10 epoch 预算下仍无法接近连续表征。dedup 能显著降低 token rate 和 bitrate，且没有带来额外明显性能损失，表明离散 token 序列中存在较多连续重复冗余。
