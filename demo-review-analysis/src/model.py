"""
    定义模型
"""
from torch import nn
from transformers import AutoModel

from config import *

class ReviewAnalysisModel(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()

        # BERT 主干结构
        self.bert = AutoModel.from_pretrained( MODEL_NAME_OR_PATH )

        # 任务头（线性层分类器）
        # 二分类，所以最终 out_features 定为 1
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    # 前向传播
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 1. BERT 前向传播
        # BERT 主干的输出：BaseModelOutputWithPoolingAndCrossAttentions
        # https://hf-mirror.com/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        # [last_hidden_state, pooler_output, hidden_states, attentions, ...]
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
            )

        # 2. 线性层的前向传播
        # 输入为 CLS 位置的特征向量 (N, 768)
        # 输出 (N, 1)
        output = self.classifier(output.pooler_output)

        # 输出列表 (N, )
        return output.squeeze(-1)

if __name__ == '__main__':
    model = ReviewAnalysisModel()
    print(model)
