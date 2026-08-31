# 7 Conclusion

> In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

在这项工作中，我们提出了 *Transformer*，这是第一个完全基于注意力的序列转换模型，用 **多头自注意力** 取代了编码器-解码器架构中最常用的循环层。

> For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.

对于翻译任务，*Transformer* 的训练速度显著快于基于循环层或卷积层的架构。

> On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art.

在 WMT 2014 英语到德语和 WMT 2014 英语到法语两个翻译任务上，我们都达到了新的最先进水平。

> In the former task our best model outperforms even all previously reported ensembles.

在前一个任务中，我们最好的模型甚至超越了所有先前报告的集成模型。

> We are excited about the future of attention-based models and plan to apply them to other tasks.

我们对基于注意力的模型的未来感到兴奋，并计划将其应用于其他任务。

> We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video.

我们计划将 *Transformer* 扩展到涉及文本以外输入和输出模态的问题，并研究局部的、受限的注意力机制，以高效处理图像、音频和视频等大型输入和输出。

> Making generation less sequential is another research goals of ours.

让生成过程减少顺序性也是我们的另一个研究目标。

> The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.

我们用于训练和评估模型的代码可在 https://github.com/tensorflow/tensor2tensor 获取。
