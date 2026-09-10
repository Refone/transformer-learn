"""
    评估脚本
"""
import torch

from tqdm import tqdm

from config import *
from model import ReviewAnalysisModel
from predict import predict_batch
from dataset import get_loader

# 评估逻辑
def evaluate(model, test_loader, device):
    model.eval()
    correct_cnt = 0
    total_cnt = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='[ Evaluate ]'):
            targets = batch.pop('labels').tolist()
            inputs = { k:v.to(device) for k,v in batch.items() }

            # 推理预测, 得到一批数据的正类概率
            results = predict_batch(model, inputs, device)

            # 一次对比预测结果和正确标签
            for result, target in zip(results, targets):
                # 得到预测标签
                pred_label = 1 if result > 0.5 else 0
                if pred_label == target:
                    correct_cnt += 1
                total_cnt += 1

    # 返回准确率
    return correct_cnt / total_cnt

# 评估主流程
def run_evaluate():
    # 1. 定义设备
    device = 'cuda' if torch.cuda.is_available else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # 2. 创建并加载模型
    model = ReviewAnalysisModel().to(device)
    model.load_state_dict( torch.load( BEST_MODEL ) )

    # 3. 获取测试集加载器
    test_loader = get_loader(train=False)

    # 4. 评估
    acc = evaluate(model, test_loader, device)

    print('评估结果: ACC - ', acc)

if __name__ == '__main__':
    run_evaluate()
