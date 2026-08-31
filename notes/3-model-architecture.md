# 3 Model Architecture

> Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].

大多数有竞争力的神经序列转换模型都具有 **编码器-解码器** 结构 [5, 2, 35]。

> Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn).

在这里，编码器将符号表示构成的输入序列 (x1, ..., xn) 映射为一个连续表示序列 z = (z1, ..., zn)。

> Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time.

给定 z，解码器随后一次一个元素地生成符号输出序列 (y1, ..., ym)。

> At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

在每个步骤中，模型都是 **自回归** 的 [10]，在生成下一个符号时将先前生成的符号作为额外输入。

> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

*Transformer* 遵循这一整体架构，对编码器和解码器都使用堆叠的自注意力层和逐位置的全连接层，分别如图 1 的左半部分和右半部分所示。

## 3.1 Encoder and Decoder Stacks

> Encoder: The encoder is composed of a stack of N = 6 identical layers.

**编码器**：编码器由 N = 6 个相同层的堆叠组成。

> Each layer has two sub-layers.

每个层有两个子层。

> The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

第一个是 **多头自注意力** 机制，第二个是简单的、逐位置的全连接 **前馈网络**。

> We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1].

我们在两个子层周围各采用一个 **残差连接** [11]，随后进行 **层归一化** [1]。

> That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.

也就是说，每个子层的输出是 `LayerNorm(x + Sublayer(x))`，其中 `Sublayer(x)` 是子层自身实现的函数。

> To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

为了便于这些残差连接，模型中的所有子层以及嵌入层都产生维度为 d_model = 512 的输出。

> Decoder: The decoder is also composed of a stack of N = 6 identical layers.

**解码器**：解码器同样由 N = 6 个相同层的堆叠组成。

> In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆叠的输出执行多头注意力。

> Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.

与编码器类似，我们在每个子层周围采用残差连接，随后进行层归一化。

> We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.

我们还修改了解码器堆叠中的自注意力子层，以防止位置关注到后续位置。

> This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

这种掩码，加上输出嵌入偏移一个位置的事实，确保了位置 i 的预测只能依赖于位置小于 i 的已知输出。

## 3.2 Attention

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

注意力函数可以描述为将一个 **查询** 和一组 **键值对** 映射到一个输出，其中查询、键、值和输出都是向量。

> The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

输出被计算为值的加权和，其中分配给每个值的权重由查询与相应键的兼容性函数计算得出。

### 3.2.1 Scaled Dot-Product Attention

> We call our particular attention "Scaled Dot-Product Attention" (Figure 2).

我们将这种特定的注意力称为「**缩放点积注意力**」（图 2）。

> The input consists of queries and keys of dimension dk, and values of dimension dv.

输入由维度为 d_k 的查询和键、以及维度为 d_v 的值组成。

> We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values.

我们计算查询与所有键的点积，将每个点积除以 √d_k，并应用 softmax 函数以获得值的权重。

> In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q.

在实践中，我们同时对打包成矩阵 Q 的一组查询计算注意力函数。

> The keys and values are also packed together into matrices K and V.

键和值也被打包成矩阵 K 和 V。

> We compute the matrix of outputs as:

我们将输出矩阵计算为：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \qquad (1)
$$

> The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention.

最常用的两种注意力函数是 **加性注意力** [2] 和 **点积（乘性）注意力**。

> Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√dk.

点积注意力与我们的算法相同，只是少了 1/√d_k 的缩放因子。

> Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.

加性注意力使用具有单个隐藏层的前馈网络来计算兼容性函数。

> While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

虽然两者在理论复杂度上相似，但点积注意力在实践中更快、更节省空间，因为它可以用高度优化的矩阵乘法代码来实现。

> While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3].

虽然在 d_k 值较小时两种机制表现相似，但在 d_k 值较大时，加性注意力优于未缩放的点积注意力 [3]。

> We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

我们推测，对于较大的 d_k 值，点积的幅度会变得很大，将 softmax 函数推入梯度极小的区域。

> To counteract this effect, we scale the dot products by 1/√dk.

为了抵消这一影响，我们将点积缩放 1/√d_k。

> To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1.

> Then their dot product, q · k = Σ q_i k_i, has mean 0 and variance dk.

*（脚注 4：为说明点积为何会变大，假设 q 和 k 的分量是均值为 0、方差为 1 的独立随机变量。那么它们的点积 q · k = Σ q_i k_i 的均值为 0、方差为 d_k。）*

### 3.2.2 Multi-Head Attention

> Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively.

我们发现，与其使用 d_model 维的键、值和查询执行单个注意力函数，不如将查询、键和值用不同的、可学习的线性投影分别投影 h 次，投影到 d_k、d_k 和 d_v 维度。

> On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values.

然后，我们在查询、键和值的每个投影版本上并行执行注意力函数，产生 d_v 维的输出值。

> These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

这些输出值被拼接并再次投影，得到最终的值，如图 2 所示。

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

多头注意力使模型能够同时关注不同位置、不同表示子空间中的信息。

> With a single attention head, averaging inhibits this.

而使用单个注意力头时，平均会抑制这种能力。

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)W^O
$$

$$
\text{where } \mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

> Where the projections are parameter matrices W_i^Q ∈ R^{d_model×d_k}, W_i^K ∈ R^{d_model×d_k}, W_i^V ∈ R^{d_model×d_v} and W^O ∈ R^{h d_v×d_model}.

其中投影是参数矩阵 W_i^Q ∈ R^{d_model×d_k}、W_i^K ∈ R^{d_model×d_k}、W_i^V ∈ R^{d_model×d_v} 和 W^O ∈ R^{h d_v×d_model}。

> In this work we employ h = 8 parallel attention layers, or heads.

在这项工作中，我们采用 h = 8 个并行的注意力层，即「头」。

> For each of these we use dk = dv = dmodel/h = 64.

对于每个头，我们使用 d_k = d_v = d_model/h = 64。

> Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

由于每个头的维度降低，总计算成本与具有完整维度的单头注意力相似。

### 3.2.3 Applications of Attention in our Model

> The Transformer uses multi-head attention in three different ways:

*Transformer* 以三种不同的方式使用多头注意力：

> In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.

在「编码器-解码器注意力」层中，查询来自上一个解码器层，而记忆的键和值来自编码器的输出。

> This allows every position in the decoder to attend over all positions in the input sequence.

这使得解码器中的每个位置都能关注输入序列中的所有位置。

> This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].

这模仿了序列到序列模型中典型的编码器-解码器注意力机制，如 [38, 2, 9]。

> The encoder contains self-attention layers.

编码器包含自注意力层。

> In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.

在自注意力层中，所有的键、值和查询都来自同一个地方，在本例中是编码器中上一层的输出。

> Each position in the encoder can attend to all positions in the previous layer of the encoder.

编码器中的每个位置都可以关注编码器上一层中的所有位置。

> Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.

类似地，解码器中的自注意力层允许解码器中的每个位置关注解码器中直至并包括该位置的所有位置。

> We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.

我们需要阻止解码器中的向左信息流，以保持自回归特性。

> We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.

我们通过在缩放点积注意力内部将 softmax 输入中所有对应非法连接的值掩码掉（设置为 −∞）来实现这一点。

> See Figure 2.

参见图 2。

## 3.3 Position-wise Feed-Forward Networks

> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.

除了注意力子层之外，我们编码器和解码器中的每一层还包含一个全连接的前馈网络，该网络被单独且相同地应用于每个位置。

> This consists of two linear transformations with a ReLU activation in between.

它由两个线性变换以及两者之间的一个 ReLU 激活函数组成。

$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \qquad (2)
$$

> While the linear transformations are the same across different positions, they use different parameters from layer to layer.

虽然线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。

> Another way of describing this is as two convolutions with kernel size 1.

另一种描述方式是将其视为两个卷积核大小为 1 的卷积。

> The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048.

输入和输出的维度为 d_model = 512，内层的维度为 d_ff = 2048。

## 3.4 Embeddings and Softmax

> Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel.

与其他序列转换模型类似，我们使用学习到的 **嵌入** 将输入词元和输出词元转换为维度为 d_model 的向量。

> We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

我们还使用通常的可学习线性变换和 softmax 函数，将解码器输出转换为预测的下一个词元的概率。

> In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].

在我们的模型中，我们在两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵，类似于 [30]。

> In the embedding layers, we multiply those weights by √dmodel.

在嵌入层中，我们将这些权重乘以 √d_model。

## 3.5 Positional Encoding

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.

由于我们的模型不包含循环和卷积，为了让模型能够利用序列的顺序，我们必须向序列中词元的相对或绝对位置注入一些信息。

> To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

为此，我们在编码器和解码器堆叠的底部向输入嵌入添加「**位置编码**」。

> The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed.

位置编码与嵌入具有相同的维度 d_model，因此两者可以相加。

> There are many choices of positional encodings, learned and fixed [9].

位置编码有多种选择，包括可学习的和固定的 [9]。

> In this work, we use sine and cosine functions of different frequencies:

在这项工作中，我们使用不同频率的正弦和余弦函数：

$$
PE_{(pos, 2i)} = \sin\left(pos/10000^{2i/d_{\text{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(pos/10000^{2i/d_{\text{model}}}\right)
$$

> where pos is the position and i is the dimension.

其中 pos 是位置，i 是维度。

> That is, each dimension of the positional encoding corresponds to a sinusoid.

也就是说，位置编码的每个维度对应一个正弦波。

> The wavelengths form a geometric progression from 2π to 10000 · 2π.

波长形成一个从 2π 到 10000 · 2π 的几何级数。

> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_pos.

我们选择这个函数，是因为我们假设它能让模型轻松学会通过相对位置进行关注，因为对于任何固定偏移 k，PE_{pos+k} 都可以表示为 PE_pos 的线性函数。

> We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)).

我们还尝试使用可学习的位置嵌入 [9] 来替代，并发现两种版本产生了几乎相同的结果（见表 3 第 (E) 行）。

> We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

我们选择正弦版本，是因为它可能使模型能够外推到比训练中遇到的更长的序列长度。
