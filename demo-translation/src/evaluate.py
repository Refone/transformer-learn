'''
    评估脚本
'''
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from config import *
from data import *
from model import TranslationModel
from tokenizer import ZhTokenizer, EnTokenizer

# 评估逻辑
def evaluate(model, test_loader, tokenizer, device):
    references = []
    predictions = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='评估'):
            inputs, targets = inputs.to(device), targets.tolist()
            # 推理预测
            batch_prediction = model.predict_batch(inputs)
            # 将预测结果添加到列表
            predictions.extend(batch_prediction)
            # 将参考译文添加到列表
            references.extend( [ [target[1:target.index(tokenizer.eos_id)]] for target in targets ] )

    return corpus_bleu(references, predictions)

# 评估主流程
def run_evaluate():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建分词器
    zh_tokenizer = ZhTokenizer.create_from_vocab_file(ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.create_from_vocab_file(EN_VOCAB_FILE)

    # 3. 创建并加载模型
    model = TranslationModel(src_tokenizer=zh_tokenizer, tgt_tokenizer=en_tokenizer).to(device)
    model.load_state_dict(torch.load( BEST_MODEL))

    # 4. 获取测试集加载器
    test_loader = TranslateDataLoader(is_train=False)

    # 5. 评估
    bleu = evaluate(model, test_loader, en_tokenizer, device)

    print("评估结果：BLEU - ", bleu)

if __name__ == '__main__':
    run_evaluate()