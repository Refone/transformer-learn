import math

import torch
import torch.nn as nn

from config import *
from tokenizer import BaseTokenizer


class TranslationModel(nn.Module):

    def __init__(self,
                 src_tokenizer:BaseTokenizer, tgt_tokenizer:BaseTokenizer):
        super().__init__()
        self.d_model = D_MODEL
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        # 定义两个嵌入层
        # src_ids (B, Ls)
        #   -> src_emb(B, Ls, D)
        self.src_embedding = nn.Embedding(
            num_embeddings=self.src_tokenizer.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=src_tokenizer.pad_id
        )
        # tgt_ids (B, Lt)
        #   -> tgt_emb(B, Lt, D)
        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_tokenizer.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=tgt_tokenizer.pad_id
        )

        # 编码层
        # encoder_input(B, Ls, D)
        #   -> memory (B, Ls, D)
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=NUM_HEADS,
                batch_first=True
            ),
            num_layers=NUM_ENCODER_LAYERS,
        )

        # 解码层
        # memory(B, Ls, D)
        # decoder_input(B, Lt, D)
        #   ->decoder_output(B, Lt, D)
        self.decoder = nn.TransformerDecoder(
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=NUM_HEADS,
                batch_first=True
            ),
            num_layers=NUM_DECODER_LAYERS,
        )
        # 线性层
        # decoder_output(B, Lt, D)
        #   -> logits(B, Lt, Vt)
        self.linear = nn.Linear(
            in_features=self.d_model,
            out_features=tgt_tokenizer.vocab_size,
        )
        # 位置编码逻辑（因为不存在参数，所以不设置其为层）
        pe = self.create_pe(self.d_model, MAX_SEQ_LEN)
        self.register_buffer('pe', pe)

    # 创建 PE 矩阵
    @staticmethod
    def create_pe(d_model, L):
        # pe: (L, d_model)
        pe = torch.zeros(L, d_model)

        # position: (L, 1)
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model / 2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )

        # position * div_term -> (L, d_model / 2)
        # pe: (L, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展到 batch 维度
        pe = pe.unsqueeze(0)  # (1, L, d_model)
        return pe

    def positional_encoding(self, x):
        return x + self.pe[:, :x.size(1)]

    def encode(self, src_ids, src_key_padding_mask):
        src_emb = self.src_embedding(src_ids)  # (B, Ls, D)

        encoder_input = self.positional_encoding(src_emb)  # (B, Ls, D)

        memory = self.encoder(encoder_input,
                              src_key_padding_mask=src_key_padding_mask)  # (B, Ls, D)

        return memory

    def decode(self,
               memory,
               memory_key_padding_mask,
               tgt_ids,
               is_causal,
               tgt_key_padding_mask):
        tgt_emb = self.tgt_embedding(tgt_ids)  # (B, Lt, D)

        decoder_input = self.positional_encoding(tgt_emb)  # (B, Lt, D)

        device = tgt_ids.device
        if is_causal:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.shape[1]).bool().to(device)
        else:
            tgt_mask = None

        decoder_output = self.decoder(
            tgt=decoder_input, memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
        )  # (B, Lt, D)

        logits = self.linear(decoder_output)    # (B, Lt, Vt)

        return logits

    def forward(self, src_ids, tgt_ids,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                is_causal=False):
        """
        前向传播
        :param src_ids: 源语言 Token 索引序列  (B, L_src)
        :param tgt_ids: 目标语言 Token 索引序列  (B, L_tgt)
        :param src_key_padding_mask:    源语言序列填充屏蔽   (B, L_src)
        :param tgt_key_padding_mask:    目标语言序列填充屏蔽  (B, L_tgt)
        :param memory_key_padding_mask: memory填充屏蔽      (B, L_src)
        :param is_causal:           因果掩码，为 True tgt_mask 设置上三角矩阵
        :return:

        直接 tgt_is_causal 方式创建因果掩码有版本兼容问题，故底层采用 tgt_mask
        """
        memory = self.encode(
            src_ids=src_ids,
            src_key_padding_mask=src_key_padding_mask)    # (B, Ls, D)

        logits = self.decode(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_ids=tgt_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
            is_causal=is_causal
        )   # (B, Lt, Vt)

        return logits

    @staticmethod
    def logits_to_ids(logits):
        probs = torch.softmax(logits, dim=-1)   # (B, Lt, Vt)
        output_ids = torch.argmax(probs, dim=-1)    # (B, Lt)
        return output_ids

    def predict_batch(self, src_ids):
        device = src_ids.device
        self.eval()
        with torch.no_grad():
            # 前向传播
            src_key_padding_mask = (src_ids == self.src_embedding.padding_idx)
            memory = self.encode(src_ids=src_ids, src_key_padding_mask=src_key_padding_mask)

            # 解码(自回归生成)
            # 1. 定义解码器初始输入(<sos>),形状(N,Lt=1)
            N = src_ids.shape[0]
            decoder_inputs = torch.full((N, 1), self.tgt_tokenizer.sos_id).to(device)

            # 定义标志位,记录当前数据样本是否已生成<eos>,默认 N 个 False
            is_finished = torch.full((N,), False, dtype=torch.bool).to(device)

            # 2. 循环迭代,自回归生成
            generated_ids = []
            for i in range(MAX_SEQ_LEN):
                # 2.1. 调用模型的一步解码,得到输出(N, T, Vt)
                tgt_key_padding_mask = (decoder_inputs == self.tgt_embedding.padding_idx)
                logits = self.decode(
                    memory=memory,
                    memory_key_padding_mask=src_key_padding_mask,
                    tgt_ids=decoder_inputs,
                    is_causal=True,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )

                # 2.2 取最后一个位置的特征向量,贪心解码,得到形状为 (N,) 的预测 ids
                next_token_ids = torch.argmax(logits[:, -1], dim=-1)

                # 2.3 更新解码器输入, 拼接一个 id (N, T) -> (N, T+1)
                decoder_inputs = torch.cat([decoder_inputs, next_token_ids.unsqueeze(1)], dim=-1)

                # 2.4 保存当前生成的 id
                generated_ids.append(next_token_ids.unsqueeze(1))

                # 2.5 预判是否生成结束 (有没有<eos>)
                is_finished |= (next_token_ids == self.tgt_tokenizer.eos_id)

                if is_finished.all():
                    break

            # 3. 整理最终输出, id 列表的二维列表
            # 3.1 合并每一步生成的 token id, 形状为(N, L), 再转成二维 list
            generated_list = torch.cat(generated_ids, dim=-1).tolist()

            # 3.2 删除 <eos> 之后的无效 token id
            for i, ids in enumerate(generated_list):
                # 如果找到 <eos> 就返回缩影位置, 截断处理
                if self.tgt_tokenizer.eos_id in ids:
                    eos_pos = ids.index(self.tgt_tokenizer.eos_id)
                    generated_list[i] = ids[:eos_pos]

        return generated_list

    def predict(self, text):
        # 1. 处理输入,编码为模型输入 (N=1, L)
        device = self.src_embedding.weight.device
        ids = self.src_tokenizer.encode(text)
        inputs = torch.tensor([ids]).to(device)

        # 2. 推理预测
        results = self.predict_batch(inputs)

        # 3. 处理输出,解码为英文句子
        sentence = self.tgt_tokenizer.decode(results[0])
        return sentence
