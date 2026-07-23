# 研究 A：Decoder 结构对比（第 9 层锚点）

本目录集中保存研究 A 的三组结果。实验固定 wav2vec2 第 9 层连续表征、3600 条训练样本、500 条验证样本和 10 个 epoch，只改变 CTC decoder 结构。

## 文件结构

```text
results/research_a/
├── decoder_linear_layer9/       # Linear decoder 原始 epoch JSON
├── decoder_mlp_layer9/          # MLP decoder；复用研究 C 的 hidden_layer_9 结果
├── decoder_transformer_layer9/  # Transformer decoder 原始 epoch JSON
├── summary.csv                  # 每组最佳/最终指标汇总
├── epoch_metrics.csv            # 每个 epoch 的指标明细
└── wer_cer_by_decoder.svg       # WER/CER 柱状图
```

## 汇总结果

| 实验 | Decoder | 参数量 | 最佳 epoch | WER | CER | Loss | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decoder_linear_layer9` | Linear | 23,070 | 10 | 0.6027 | 0.1846 | 0.6456 | 0.0102 |
| `decoder_mlp_layer9` | MLP | 410,654 | 10 | 0.2954 | 0.0768 | 0.3088 | 0.0111 |
| `decoder_transformer_layer9` | Transformer | 1,784,094 | 10 | 0.3163 | 0.0845 | 0.3922 | 0.0104 |

## 结果解读与报告写法

三个 decoder 的排序很清晰：MLP 最好，Transformer 略差，Linear 明显最差。

Linear 到 MLP 的 WER 从 0.6027 降到 0.2954，说明第 9 层 SSL 表征虽然质量较高，但并不是完全线性可分；加入逐帧非线性映射后，decoder 能显著提升识别性能。因此，A1→A2 可以归因为“逐帧非线性带来的主要收益”。

MLP 到 Transformer 的 WER 从 0.2954 小幅升到 0.3163，说明在当前固定训练预算下，额外跨帧建模没有带来进一步收益。一个合理解释是：wav2vec2 本身已经是多层 Transformer，第 9 层 hidden states 已经包含较充分的上下文信息；在其后再叠加 Transformer decoder，可能存在一定上下文建模冗余。另一个因素是低资源训练条件：Transformer decoder 参数量约 178 万，明显大于 MLP 的 41 万，在 3600 条样本和 10 epoch 的固定预算下更难优化充分。

Loss 排序也支持这个解释：Linear 0.6456 > Transformer 0.3922 > MLP 0.3088，与 WER/CER 排序一致。Transformer 的 loss 高于 MLP，说明它的劣势更像是固定预算下优化不足，而不是“训练 loss 很低但泛化很差”的过拟合现象。

RTF 三者都约为 0.010，几乎没有随 decoder 参数量显著变化。尽管 Linear 和 Transformer 的 decoder 参数量相差约 77 倍，推理速度仍非常接近，说明冻结 wav2vec2 的前向计算主导了整体推理耗时，轻量 decoder 的计算开销在当前设置下不是瓶颈。

报告中可以采用较稳妥的表述：

> 在固定第 9 层 SSL 表征和相同训练预算下，逐帧非线性 decoder 带来显著提升；跨帧 Transformer decoder 未进一步超过 MLP，可能是因为 SSL 表征已包含上下文信息，同时更大的 decoder 在低资源固定预算下更难优化。该结论不表示 Transformer decoder 本质更差，而是说明在当前实验预算和超参数下，MLP 具有更好的效果与参数效率。

## 初步结论

- Linear decoder 的 WER 明显高于 MLP，说明第 9 层 SSL 表征并非完全线性可分，逐帧非线性带来显著收益。
- Transformer decoder 参数量最大，但本次结果略差于 MLP，说明在 3600 条低资源设置下，额外跨帧上下文并没有稳定转化为识别收益，可能受到训练数据规模、训练轮数或 decoder 容量过大的影响。
- 当前设置下 MLP decoder 是性价比最高的选择：参数量适中，WER/CER 最低。

报告归因可以写作：A1→A2 体现逐帧非线性收益；A2→A3 在本设置下没有带来额外增益，提示跨帧 decoder 需要更多数据或更细调参。
