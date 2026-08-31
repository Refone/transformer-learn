# 2 Background

> The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.

减少顺序计算的目标也构成了 Extended Neural GPU [16]、ByteNet [18] 和 ConvS2S [9] 的基础，它们都使用 **卷积神经网络** 作为基本构建块，为所有输入和输出位置并行计算隐藏表示。

> In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet.

在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数量，会随着位置之间距离的增长而增长，对于 ConvS2S 是线性增长，对于 ByteNet 是对数增长。

> This makes it more difficult to learn dependencies between distant positions [12].

这使得学习远距离位置之间的依赖关系变得更加困难 [12]。

> In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

而在 *Transformer* 中，这一数量被降低到常数级别的操作，尽管代价是由于对注意力加权位置取平均而导致有效分辨率降低，我们通过第 3.2 节中描述的 **多头注意力** 来抵消这一影响。

> Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

**自注意力**（有时称为内部注意力）是一种将单个序列中不同位置关联起来、以计算该序列表示的注意力机制。

> Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

自注意力已被成功应用于多种任务，包括阅读理解、生成式摘要、文本蕴含以及学习与任务无关的句子表示 [4, 27, 28, 22]。

> End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

端到端记忆网络基于循环注意力机制，而非序列对齐的循环，并已被证明在简单语言问答和语言建模任务上表现良好 [34]。

> To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

然而，据我们所知，*Transformer* 是第一个完全依靠自注意力来计算其输入和输出表示、而不使用序列对齐的 RNN 或卷积的转换模型。

> In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

在接下来的章节中，我们将描述 *Transformer*，阐明自注意力的动机，并讨论它相对于 [17, 18] 和 [9] 等模型的优势。
