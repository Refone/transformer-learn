# 4 Why Self-Attention

> In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈ R^d, such as a hidden layer in a typical sequence transduction encoder or decoder.

在本节中，我们将自注意力层与常用于将一个可变长度的符号表示序列 (x1, ..., xn) 映射到另一个等长序列 (z1, ..., zn)（其中 xi, zi ∈ R^d）的循环层和卷积层进行多方面的比较，例如典型序列转换编码器或解码器中的隐藏层。

> Motivating our use of self-attention we consider three desiderata.

为阐明我们使用自注意力的动机，我们考虑三个目标。

> One is the total computational complexity per layer.

其一是每层的总计算复杂度。

> Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

其二是可并行化的计算量，以所需的最少顺序操作数来衡量。

> The third is the path length between long-range dependencies in the network.

第三是网络中长程依赖之间的路径长度。

> Learning long-range dependencies is a key challenge in many sequence transduction tasks.

学习长程依赖是许多序列转换任务中的关键挑战。

> One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network.

影响学习此类依赖能力的一个关键因素是前向和后向信号在网络中必须穿越的路径长度。

> The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12].

输入和输出序列中任意位置组合之间的这些路径越短，就越容易学习长程依赖 [12]。

> Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

因此，我们还比较了由不同层类型组成的网络中任意两个输入和输出位置之间的最大路径长度。

**Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.**

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---|---|---|
| Self-Attention | O(n² · d) | O(1) | O(1) |
| Recurrent | O(n · d²) | O(n) | O(n) |
| Convolutional | O(k · n · d²) | O(1) | O(logₖ(n)) |
| Self-Attention (restricted) | O(r · n · d) | O(1) | O(n/r) |

> As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations.

如表 1 所述，自注意力层以常数数量的顺序执行操作连接所有位置，而循环层需要 O(n) 次顺序操作。

> In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations.

就计算复杂度而言，当序列长度 n 小于表示维度 d 时，自注意力层比循环层更快，而机器翻译中最先进模型使用的句子表示（如 word-piece [38] 和字节对 [31] 表示）通常正是这种情况。

> To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position.

为了提高涉及超长序列任务的计算性能，可以将自注意力限制为只考虑以相应输出位置为中心的输入序列中大小为 r 的邻域。

> This would increase the maximum path length to O(n/r).

这将使最大路径长度增加到 O(n/r)。

> We plan to investigate this approach further in future work.

我们计划在未来工作中进一步研究这一方法。

> A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions.

单个卷积核宽度 k < n 的卷积层无法连接所有输入和输出位置对。

> Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(log_k(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network.

要做到这一点，在连续卷积核的情况下需要堆叠 O(n/k) 个卷积层，在空洞卷积 [18] 的情况下需要 O(log_k(n)) 个卷积层，这会增加网络中任意两个位置之间最长路径的长度。

> Convolutional layers are generally more expensive than recurrent layers, by a factor of k.

卷积层通常比循环层更昂贵，成本高出 k 倍。

> Separable convolutions [6], however, decrease the complexity considerably, to O(k · n · d + n · d²).

然而，可分离卷积 [6] 能显著降低复杂度，降至 O(k · n · d + n · d²)。

> Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

然而，即使 k = n，可分离卷积的复杂度也等于一个自注意力层和一个逐位置前馈层的组合，这正是我们模型中采用的方法。

> As side benefit, self-attention could yield more interpretable models.

作为附带的好处，自注意力可以产生更具可解释性的模型。

> We inspect attention distributions from our models and present and discuss examples in the appendix.

我们检查了模型的注意力分布，并在附录中展示和讨论了示例。

> Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

不仅各个注意力头明显学会了执行不同的任务，许多头似乎还表现出与句子的句法和语义结构相关的行为。
