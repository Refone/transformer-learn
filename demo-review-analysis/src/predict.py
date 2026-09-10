"""
    包装推理啊预测逻辑，构建程序
"""
import torch

from config import *
from model import ReviewAnalysisModel
from transformers import AutoTokenizer

# 推理预测核心逻辑
def predict_batch(model, inputs, device):
    model.eval()

    with torch.no_grad():
        inputs = { k:v.to(device) for k,v in inputs.items() }

        # 前向传播，得到输出 == 正类预测得分 (N, )
        outputs = model(**inputs)

        # 得分有正有负，
        # 通过一个 sigmoid 函数转化为 [0，1] 概率，
        # 形状为 (N, )
        results = torch.sigmoid(outputs)

    # print(type(results))  # torch.tensor
    return results.tolist() # 张量转 list

# 预测流程：传入一句评论文本，返回对应的评价是好评还是差评
def predict(text, model, tokenizer, device):
    # 1. 处理输入，编码成模型输入
    inputs = tokenizer(
        text,
        return_tensors='pt'
        )

    # 2. 推理预测
    results = predict_batch(model, inputs, device)

    return results[0]

# 启动应用程序
def run_predict_app():
    # 1. 定义设备
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # 2. 创建分词器
    tokenizer = AutoTokenizer.from_pretrained( MODEL_NAME_OR_PATH )

    # 3. 创建并加载模型
    model = ReviewAnalysisModel().to(device)
    model.load_state_dict( torch.load( BEST_MODEL ) )

    # 4. 用循环实现应用程序
    print('欢迎使用评论情感分析程序! 输入 q/quit/exit 退出...')
    while True:
        # 等待用户输入
        text = input("请输入评论: ")

        if text in ['q', 'quit', 'exit']:
            print('欢迎下次再来')
            break

        if text.strip() == '':
            continue

        # 预测
        proba = predict(text, model, tokenizer, device)

        if proba > 0.5:
            print(f'好评 置信度: {proba:.6f}')
        else:
            print(f'差评 置信度: {1-proba:.6f}')

if __name__ == '__main__':
    run_predict_app()
