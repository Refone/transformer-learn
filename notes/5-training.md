# 5 Training

> This section describes the training regime for our models.

本节描述我们模型的训练机制。

## 5.1 Training Data and Batching

> We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.

我们在标准的 WMT 2014 英语-德语数据集上训练，该数据集包含约 450 万个句子对。

> Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens.

句子使用 **字节对编码** [3] 进行编码，该编码具有约 37000 个词元的共享源-目标词汇表。

> For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38].

对于英语-法语，我们使用了规模大得多的 WMT 2014 英语-法语数据集，包含 3600 万个句子，并将词元划分为 32000 个 word-piece 词汇表 [38]。

> Sentence pairs were batched together by approximate sequence length.

句子对按大致序列长度分批。

> Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

每个训练批次包含一组句子对，其中大约包含 25000 个源词元和 25000 个目标词元。

## 5.2 Hardware and Schedule

> We trained our models on one machine with 8 NVIDIA P100 GPUs.

我们在配备 8 块 NVIDIA P100 GPU 的一台机器上训练我们的模型。

> For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.

对于使用本文所述超参数的 base 模型，每个训练步骤耗时约 0.4 秒。

> We trained the base models for a total of 100,000 steps or 12 hours.

我们总共训练 base 模型 100000 步，即 12 小时。

> For our big models, (described on the bottom line of table 3), step time was 1.0 seconds.

对于我们的 big 模型（见表 3 最后一行），每步耗时为 1.0 秒。

> The big models were trained for 300,000 steps (3.5 days).

big 模型训练了 300000 步（3.5 天）。

## 5.3 Optimizer

> We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10⁻⁹.

我们使用 Adam 优化器 [20]，其中 β1 = 0.9、β2 = 0.98、ε = 10⁻⁹。

> We varied the learning rate over the course of training, according to the formula:

我们在训练过程中根据以下公式调整学习率：

$$
lrate = d_{\text{model}}^{-0.5} \cdot \min(step\_num^{-0.5},\; step\_num \cdot warmup\_steps^{-1.5}) \qquad (3)
$$

> This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.

这对应于在前 warmup_steps 个训练步骤中线性增加学习率，此后按步数的倒数平方根比例降低学习率。

> We used warmup_steps = 4000.

我们使用 warmup_steps = 4000。

## 5.4 Regularization

> We employ three types of regularization during training:

我们在训练期间采用三种类型的正则化：

> Residual Dropout: We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized.

**残差 Dropout**：我们将 dropout [33] 应用于每个子层的输出，然后将其加到子层输入并进行归一化。

> In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.

此外，我们在编码器和解码器堆叠中，对嵌入和位置编码的和应用 dropout。

> For the base model, we use a rate of Pdrop = 0.1.

对于 base 模型，我们使用 P_drop = 0.1 的比率。

> Label Smoothing: During training, we employed label smoothing of value εls = 0.1 [36].

**标签平滑**：在训练期间，我们采用了值 ε_ls = 0.1 的标签平滑 [36]。

> This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

这会损害困惑度，因为模型学会了更加不确定，但提高了准确率和 BLEU 分数。
