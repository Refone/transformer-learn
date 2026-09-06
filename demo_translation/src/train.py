import json
import pandas as pd
from model import *
from config import *
from data import *
import torch
from torch.utils.data import Dataset, DataLoader

def train_chinese_to_english(device):
    # preprocess()
    dataset = TranslateDataset(TEST_IDS_JSONL_FILE)
    enTokenizer = EnTokenizer.create_from_vocab_file(EN_VOCAB_FILE)
    zhTokenizer = ZhTokenizer.create_from_vocab_file(ZH_VOCAB_FILE)

    model = TranslationModel(
        src_vocab_size = zhTokenizer.vocab_size,
        src_padding_idx=zhTokenizer.pad_id,
        max_src_seq_len=dataset.max_zh_len,
        tgt_vocab_size = enTokenizer.vocab_size,
        tgt_padding_idx = enTokenizer.pad_id,
        max_tgt_seq_len=dataset.max_en_len,
    ).to(device)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    # for epochs in range(EPOCHS):


if __name__ == '__main__':
    # 数据整理
    # preprocess()
    dev = 'cuda' if torch.cuda.is_available() else \
            'mps' if torch.backends.mps.is_available() else \
            'cpu'

    train_chinese_to_english(dev)