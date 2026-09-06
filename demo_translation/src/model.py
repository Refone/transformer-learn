import math

import torch
import torch.nn as nn
from config import *

class TranslationModel(nn.Module):

    def __init__(self,
                 src_vocab_size, src_padding_idx, max_src_seq_len,
                 tgt_vocab_size, tgt_padding_idx, max_tgt_seq_len):
        super().__init__()
        self.d_model = D_MODEL
        # 定义两个嵌入层
        # src_ids (B, Ls)
        #   -> src_emb(B, Ls, D)
        self.input_embedding = nn.Embedding(
            src_vocab_size,
            self.d_model,
            padding_idx=src_padding_idx
        )
        # tgt_ids (B, Lt)
        #   -> tgt_emb(B, Lt, D)
        self.output_embedding = nn.Embedding(
            tgt_vocab_size,
            self.d_model,
            padding_idx=tgt_padding_idx
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
            out_features=tgt_vocab_size
        )
        # 位置编码逻辑（因为不存在参数，所以不设置其为层）
        pe = self.create_pe(self.d_model, max(max_src_seq_len, max_tgt_seq_len))
        self.register_buffer('pe', pe)

    # 创建 PE 矩阵
    def create_pe(self, d_model, L):
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
        src_emb = self.input_embedding(src_ids)  # (B, Ls, D)

        encoder_input = self.positional_encoding(src_emb)  # (B, Ls, D)

        memory = self.encoder(encoder_input,
                              src_key_padding_mask=src_key_padding_mask)  # (B, Ls, D)

        return memory

    def decode(self,
               memory,
               memory_key_padding_mask,
               tgt_ids,
               tgt_is_causal,
               tgt_key_padding_mask):
        tgt_emb = self.output_embedding(tgt_ids)  # (B, Lt, D)

        decoder_input = self.positional_encoding(tgt_emb)  # (B, Lt, D)

        if tgt_is_causal:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.shape[1]).bool()
        else:
            tgt_mask = None

        decoder_output = self.decoder(
            tgt=decoder_input, memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
        )  # (B, Lt, D)

        return decoder_output

    def forward(self, src_ids, tgt_ids,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=False):
        """
        前向传播
        :param src_ids: 源语言 Token 索引序列  (B, L_src)
        :param tgt_ids: 目标语言 Token 索引序列  (B, L_tgt)
        :param src_key_padding_mask:    源语言序列填充屏蔽   (B, L_src)
        :param tgt_key_padding_mask:    目标语言序列填充屏蔽  (B, L_tgt)
        :param memory_key_padding_mask: memory填充屏蔽      (B, L_src)
        :param tgt_is_causal:           因果掩码，为 True tgt_mask 设置上三角矩阵
        :return:

        直接 tgt_is_causal 方式创建因果掩码有版本兼容问题，故底层采用 tgt_mask
        """
        memory = self.encode(
            src_ids=src_ids,
            src_key_padding_mask=src_key_padding_mask)    # (B, Ls, D)

        decoder_output = self.decode(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_ids=tgt_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_is_causal=tgt_is_causal
        )   # (B, Lt, D)

        logits = self.linear(decoder_output)    # (B, Lt, Vt)

        probs = torch.softmax(logits, dim=-1)   # (B, Lt, Vt)

        output_ids = torch.argmax(probs, dim=-1)    # (B, Lt)

        return output_ids

