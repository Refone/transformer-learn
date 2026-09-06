import json
from hmac import trans_5C

import torch
from sympy.tensor import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from demo_translation.src.tokenizer import EnTokenizer, ZhTokenizer

def preprocess():
    # 清空中间文件
    for file_path in PROCESSED_DATA_DIR.iterdir():
        if file_path.is_file():
            file_path.unlink()

    # 1. 读取文件
    df = pd.read_csv(
        RAW_DATA_FILE,
        header=None,
        usecols=[0, 1],
        names=['en', 'zh'],
        encoding='utf8',
        sep='\t')
    # 去掉最后一列
    df.dropna(inplace=True)

    # 2. 数据集划分
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    # 这里为了方便，将训练集和测试集写入文件，方便追溯。
    train_set.to_csv(TRAIN_RAW_FILE)
    test_set.to_csv(TEST_RAW_FILE)

    # 3. 创建词表，并保存文件
    EnTokenizer.build_vocab_file(train_set['en'], EN_VOCAB_FILE)
    ZhTokenizer.build_vocab_file(train_set['zh'], ZH_VOCAB_FILE)

    # 4. 创建分词器
    en_tokenizer = EnTokenizer.create_from_vocab_file(EN_VOCAB_FILE)
    zh_tokenizer = ZhTokenizer.create_from_vocab_file(ZH_VOCAB_FILE)

    # 5. 编码转化数据
    # 转化为形如：
    # {"en":[23, 345, .., 2342], "zh":[235, 1234, ..., 2134]}
    en_encoder = lambda text: en_tokenizer.encode(text)
    zh_encoder = lambda text: zh_tokenizer.encode(text)
    train_set['en'] = train_set['en'].apply( en_encoder )
    train_set['zh'] = train_set['zh'].apply( zh_encoder )
    test_set['zh'] = test_set['zh'].apply( zh_encoder )
    test_set['en'] = test_set['en'].apply( en_encoder )

    # 6. 保存数据集
    train_set.to_json(TRAIN_IDS_JSONL_FILE, orient='records', lines=True)
    test_set.to_json(TEST_IDS_JSONL_FILE, orient='records', lines=True)

    print(f'数据预处理结束, 训练集{len(train_set)}, 测试集{len(test_set)}')

class TranslateDataset(Dataset):
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = pd.read_json(TRAIN_IDS_JSONL_FILE, lines=True).to_dict(orient='records')

        self.max_zh_len = max((len(item['zh']) for item in self.data), default=0)
        self.max_en_len = max((len(item['en']) for item in self.data), default=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        获取第 index 条数据
        :param index:
        :return: zh, en
        """
        return self.data[index]['zh'], self.data[index]['en']

def get_loader(is_train):
    path = PROCESSED_DATA_DIR / (TRAIN_RAW_FILE if is_train else TEST_RAW_FILE)
    # 获取数据集
    dataset = TranslateDataset(path)

    def collate_fn(batch):
        input_list = [ torch.tensor(item[0]) for item in batch ]
        target_list = [ torch.tensor(item[1]) for item in batch ]

        # 各自填充
        input_batch = pad_sequence(input_list, batch_first=True, padding_value=0)
        target_batch = pad_sequence(target_list, batch_first=True, padding_value=0)

        return input_batch, target_batch

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=is_train, collate_fn=collate_fn)
    return loader