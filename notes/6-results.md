# 6 Results

## 6.1 Machine Translation

> On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.

在 WMT 2014 英语到德语翻译任务上，big transformer 模型（表 2 中的 Transformer (big)）以超过 2.0 BLEU 的优势超越了此前报告的最佳模型（包括集成模型），确立了 28.4 这一新的最先进 BLEU 分数。

> The configuration of this model is listed in the bottom line of Table 3.

该模型的配置列于表 3 的最后一行。

> Training took 3.5 days on 8 P100 GPUs.

训练在 8 块 P100 GPU 上耗时 3.5 天。

> Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

即使我们的 base 模型也超越了所有先前发布的模型和集成模型，而其训练成本仅为任何有竞争力模型的一小部分。

> On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model.

在 WMT 2014 英语到法语翻译任务上，我们的 big 模型取得了 41.0 的 BLEU 分数，超越了所有先前发布的单一模型，训练成本不到此前最先进模型的 1/4。

> The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.

为英语到法语训练的 Transformer (big) 模型使用了 dropout 率 P_drop = 0.1，而不是 0.3。

> For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals.

对于 base 模型，我们使用通过对最后 5 个检查点（每 10 分钟保存一次）求平均得到的单一模型。

> For the big models, we averaged the last 20 checkpoints.

对于 big 模型，我们对最后 20 个检查点求平均。

> We used beam search with a beam size of 4 and length penalty α = 0.6 [38].

我们使用束大小为 4、长度惩罚 α = 0.6 的束搜索 [38]。

> These hyperparameters were chosen after experimentation on the development set.

这些超参数是在开发集上实验后选定的。

> We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

我们将推理期间的最大输出长度设置为输入长度 + 50，但在可能的情况下提前终止 [38]。

> Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature.

表 2 总结了我们的结果，并将我们的翻译质量和训练成本与文献中的其他模型架构进行了比较。

> We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.

我们通过将训练时间、使用的 GPU 数量以及每块 GPU 的持续单精度浮点运算能力的估计值相乘，来估计训练一个模型所使用的浮点运算次数。

*（脚注 5：我们为 K80、K40、M40 和 P100 分别使用了 2.8、3.7、6.0 和 9.5 TFLOPS 的值。）*

**Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.**

| Model | BLEU (EN-DE) | BLEU (EN-FR) | Training Cost (FLOPs) EN-DE | Training Cost (FLOPs) EN-FR |
|---|---|---|---|---|
| ByteNet [18] | 23.75 | | | |
| Deep-Att + PosUnk [39] | | 39.2 | | 1.0 · 10²⁰ |
| GNMT + RL [38] | 24.6 | 39.92 | 2.3 · 10¹⁹ | 1.4 · 10²⁰ |
| ConvS2S [9] | 25.16 | 40.46 | 9.6 · 10¹⁸ | 1.5 · 10²⁰ |
| MoE [32] | 26.03 | 40.56 | 2.0 · 10¹⁹ | 1.2 · 10²⁰ |
| Deep-Att + PosUnk Ensemble [39] | | 40.4 | | 8.0 · 10²⁰ |
| GNMT + RL Ensemble [38] | 26.30 | 41.16 | 1.8 · 10²⁰ | 1.1 · 10²¹ |
| ConvS2S Ensemble [9] | 26.36 | 41.29 | 7.7 · 10¹⁹ | 1.2 · 10²¹ |
| Transformer (base model) | 27.3 | 38.1 | 3.3 · 10¹⁸ | |
| Transformer (big) | 28.4 | 41.8 | 2.3 · 10¹⁹ | |

## 6.2 Model Variations

> To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013.

为了评估 *Transformer* 不同组件的重要性，我们以不同方式修改 base 模型，衡量其在英语到德语翻译开发集 newstest2013 上的性能变化。

> We used beam search as described in the previous section, but no checkpoint averaging.

我们使用了上一节描述的束搜索，但没有使用检查点平均。

> We present these results in Table 3.

我们在表 3 中展示了这些结果。

**Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.**

| | N | d_model | d_ff | h | d_k | d_v | P_drop | ε_ls | train steps | PPL (dev) | BLEU (dev) | params (×10⁶) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) | | | | 1 | 512 | 512 | | | | 5.29 | 24.9 | |
| | | | | 4 | 128 | 128 | | | | 5.00 | 25.5 | |
| | | | | 16 | 32 | 32 | | | | 4.91 | 25.8 | |
| | | | | 32 | 16 | 16 | | | | 5.01 | 25.4 | |
| (B) | | | | | 16 | | | | | 5.16 | 25.1 | 58 |
| | | | | | 32 | | | | | 5.01 | 25.4 | 60 |
| (C) | 2 | | | | | | | | | 6.11 | 23.7 | 36 |
| | 4 | | | | | | | | | 5.19 | 25.3 | 50 |
| | 8 | | | | | | | | | 4.88 | 25.5 | 80 |
| | | 256 | | | 32 | 32 | | | | 5.75 | 24.5 | 28 |
| | | 1024 | | | 128 | 128 | | | | 4.66 | 26.0 | 168 |
| | | | 1024 | | | | | | | 5.12 | 25.4 | 53 |
| | | | 4096 | | | | | | | 4.75 | 26.2 | 90 |
| (D) | | | | | | | 0.0 | | | 5.77 | 24.6 | |
| | | | | | | | 0.2 | | | 4.95 | 25.5 | |
| | | | | | | | | 0.0 | | 4.67 | 25.3 | |
| | | | | | | | | 0.2 | | 5.47 | 25.7 | |
| (E) | positional embedding instead of sinusoids | | | | | | | | | 4.92 | 25.7 | |
| big | 6 | 1024 | 4096 | 16 | | | 0.3 | | 300K | 4.33 | 26.4 | 213 |

> In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2.

在表 3 第 (A) 行中，我们改变注意力头的数量以及注意力键和值的维度，同时保持计算量不变，如第 3.2.2 节所述。

> While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

虽然单头注意力比最佳设置差 0.9 BLEU，但头数过多时质量也会下降。

> In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality.

在表 3 第 (B) 行中，我们观察到减小注意力键的尺寸 d_k 会损害模型质量。

> This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial.

这表明确定兼容性并不容易，且比点积更复杂的兼容性函数可能更有益。

> We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting.

我们进一步在第 (C) 和 (D) 行中观察到，正如预期的那样，更大的模型更好，而 dropout 在避免过拟合方面非常有效。

> In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

在第 (E) 行中，我们用可学习的位置嵌入 [9] 替换了正弦位置编码，观察到与 base 模型几乎相同的结果。

## 6.3 English Constituency Parsing

> To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing.

为了评估 *Transformer* 能否泛化到其他任务，我们在英文成分句法分析上进行了实验。

> This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input.

该任务提出了特定的挑战：输出受到强烈的结构约束，且比输入长得多。

> Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

此外，RNN 序列到序列模型在小数据条件下一直未能取得最先进的结果 [37]。

> We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences.

我们在 Penn Treebank [25] 的 Wall Street Journal (WSJ) 部分上训练了一个 d_model = 1024 的 4 层 transformer，约有 40000 个训练句子。

> We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from [37] with approximately 17M sentences.

我们还在半监督环境下训练它，使用规模更大的、来自 [37] 的高置信度和 BerkleyParser 语料库，约有 1700 万个句子。

> We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

对于仅 WSJ 的设置，我们使用 16K 词元的词汇表；对于半监督设置，使用 32K 词元的词汇表。

> We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model.

我们只进行了少量实验，在 Section 22 开发集上选择 dropout（包括注意力和残差，见第 5.4 节）、学习率和束大小，所有其他参数与英语到德语 base 翻译模型保持不变。

> During inference, we increased the maximum output length to input length + 300.

在推理期间，我们将最大输出长度增加到输入长度 + 300。

> We used a beam size of 21 and α = 0.3 for both WSJ only and the semi-supervised setting.

对于仅 WSJ 和半监督两种设置，我们均使用束大小 21 和 α = 0.3。

**Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)**

| Parser | Training | WSJ 23 F1 |
|---|---|---|
| Vinyals & Kaiser et al. (2014) [37] | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) [29] | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) [40] | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) [8] | WSJ only, discriminative | 91.7 |
| Transformer (4 layers) | WSJ only, discriminative | 91.3 |
| Zhu et al. (2013) [40] | semi-supervised | 91.3 |
| Huang & Harper (2009) [14] | semi-supervised | 91.3 |
| McClosky et al. (2006) [26] | semi-supervised | 92.1 |
| Vinyals & Kaiser et al. (2014) [37] | semi-supervised | 92.1 |
| Transformer (4 layers) | semi-supervised | 92.7 |
| Luong et al. (2015) [23] | multi-task | 93.0 |
| Dyer et al. (2016) [8] | generative | 93.3 |

> Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

我们在表 4 中的结果表明，尽管缺乏针对任务的调优，我们的模型表现仍然出奇地好，取得了比所有先前报告的模型更好的结果，唯一的例外是循环神经网络语法 [8]。

> In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the BerkeleyParser [29] even when training only on the WSJ training set of 40K sentences.

与 RNN 序列到序列模型 [37] 相比，即使仅在 40000 个句子的 WSJ 训练集上训练，*Transformer* 也超越了 BerkeleyParser [29]。
