# Abstract

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.

目前主流的序列转换模型基于复杂的 **循环神经网络** 或 **卷积神经网络**，它们都包含了一个 **编码器** 和一个 **解码器**。

> The best performing models also connect the encoder and decoder through an attention mechanism.

其中性能最佳的模型，还通过一个 **注意力机制** 把编码器和解码器连接起来。

> We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutionsentirely.

我们提出了一种新的简单网络架构 *Transformer*，它完全基于注意力机制，彻底摒弃了循环和卷积的架构。

> Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantlyless time to train.

在两个机器翻译任务上的实验表明，这些模型在质量上更优，同时具有更高的可并行性，可并行性可以大幅减少训练时间。

> Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

我们的模型在 WMT 2014 英语到德语翻译任务上取得了 28.4 BLEU 的分数，比现有的最佳结果（包括集成模型）提高了超过 2 个 BLEU。

- BLEU（Bilingual Evaluation Understudy），机器翻译经典评价指标

> On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

在 WMT 2014 英语到法语翻译任务上，我们的模型在八块 GPU 上训练 3.5 天后，取得了 41.8 BLEU 的分数，属于是第一梯队分数，而训练成本，根据文献，仅为第一名模型训练成本的一小部分。

> We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

我们证明，*Transformer* 能够很好地泛化到其他任务上。怎么证明的？它能成功完成英语成分句法分析，无论是在大规模训练数据还是有限训练数据上，均表现良好。