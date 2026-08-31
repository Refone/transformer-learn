# Abstract

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.

目前主流的序列转换模型基于复杂的 **循环神经网络** 或 **卷积神经网络**，它们都包含了一个 **编码器** 和一个 **解码器**。

> The best performing models also connect the encoder and decoder through an attention mechanism.

其中性能最佳的模型，还通过一个 **注意力机制** 把编码器和解码器连接起来。

> We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutionsentirely.

我们提出了一种新的简单网络架构 *Transformer*，它完全基于注意力机制，彻底摒弃了循环和卷积的架构。

> Experiments on two machine translation tasks show these models tobe superior in quality while being more parallelizable and requiring significantlyless time to train.

在两个机器翻译任务上的实验表明，这些模型在质量上更优，同时具有更高的可并行性，并且训练所需时间大幅减少。