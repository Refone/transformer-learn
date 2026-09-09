"""
    主程序入口
"""
import torch

from config import *
from tokenizer import ZhTokenizer, EnTokenizer
from model import TranslationModel

def run_predict_app():
    # 1. 定义设备
    device = 'cuda' if torch.cuda.is_available() \
        else 'mps' if torch.backends.mps.is_available() \
        else 'cpu'

    # 2. 创建分词器
    zh_tokenizer = ZhTokenizer.create_from_vocab_file( ZH_VOCAB_FILE )
    en_tokenizer = EnTokenizer.create_from_vocab_file( EN_VOCAB_FILE )

    # 3. 定义模型
    model = TranslationModel(
        src_tokenizer=zh_tokenizer,
        tgt_tokenizer=en_tokenizer
    ).to(device)
    model.load_state_dict( torch.load( BEST_MODEL ) )

    # 4. 开始进入循环
    print('欢迎使用中译英翻译程序，输入 exit 退出...')
    print('建议句末加上标点符号！')
    while True:
        text = input("中文: ")

        if text == 'exit':
            print('欢迎下次再来使用')
            break

        if text.strip() == '':
            continue

        # 预测
        sentences = model.predict( text )

        print('英文: ', sentences)

if __name__ == '__main__':
    run_predict_app()