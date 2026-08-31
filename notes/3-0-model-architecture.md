# 3 Model Architecture

> Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].

大多数有竞争力的神经序列转换模型都具有 **编码器-解码器** 结构 [5, 2, 35]。

> Here, the encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $z = (z_1, ..., z_n)$.

在这里，编码器将符号表示构成的输入序列 $(x_1, ..., x_n)$ 映射为一个连续表示序列 $z = (z_1, ..., z_n)$。

> Given $z$, the decoder then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time.

给定 $z$，解码器随后一次一个元素地生成符号输出序列 $(y_1, ..., y_m)$。

> At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

在每个步骤中，模型都是 **自回归** 的 [10]，在生成下一个符号时将先前生成的符号作为额外输入。

- ![img](../images/03_01_basic_encoder_decoder.png)
- 大概是图中这个意思，但是字母不一样。按图说，意思就是：
- 给定输入序列  $(x_1, ..., x_n)$，映射为  $h = (h_1, ..., h_n)$。
- 给定 $h$, 随后一次一个元素地生成 $z = (z_1, ..., z_n)$。

> <p align="center">
>     <img src="../images/03_01_transformer_architecture.png", width="50%">
>     <p align="center">Figure 1</p>
> </p>
> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

*Transformer* 遵循这一整体架构，对编码器和解码器都使用堆叠的自注意力层和逐位置的全连接层，分别如图 1 的左半部分和右半部分所示。



