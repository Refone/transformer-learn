## 3.4 Embeddings and Softmax

<p align="center">
    <img src="../images/03_02_transformer_architecture.png" width="50%">
</p>

> Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel.

与其他序列转换模型类似，我们使用学习到的 **嵌入层** 将输入词元和输出词元转换为维度为 $d_{model}$ 的向量。

> We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

我们还使用通常的可学习线性变换和 softmax 函数，将解码器输出转换为预测的下一个词元的概率。

> In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].

在我们的模型中，我们在两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵，类似于 [30]。

- 虽然是翻译任务，但是 Input 和 Output 嵌入层、Softmax 前的线性转译层，三层共用一套权重，也就是说这个词向量表里面又有英语，又有目标语言（德语、法语）

> In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.

在嵌入层中，我们将这些权重乘以 $\sqrt{d_{model}}$。

