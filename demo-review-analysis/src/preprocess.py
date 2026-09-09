"""
    数据预处理
"""
from transformers import AutoTokenizer
from datasets import load_dataset, ClassLabel

from config import *

def preprocess():
    print("数据预处理开始...")
    # 1. 读取文件
    dataset = load_dataset('csv', data_files=str(RAW_DATA_FILE))
    dataset = dataset['train']

    # 2. 去掉 cat 列（不需要），过滤数据
    dataset = dataset.remove_columns('cat')
    dataset = dataset.filter( lambda x: x['review'] is not None and x['label'] in [0, 1])

    # 3. 划分数据集
    dataset = dataset.cast_column('label', ClassLabel(names=['pos','neg']))
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')

    # 4. 创建分词器
    tokenizer = AutoTokenizer.from_pretrained( MODEL_NAME_OR_PATH )

    # 5. 编码数据
    def encode_fn(batch):
        inputs = tokenizer(
            batch['review'],
            # padding='max_length',
            max_length=MAX_SEQ_LEN,
            truncation=True
        )
        # HuggingFace Trainer
        # 强制要求 数据集中有一个字段名叫 labels（小写复数）
        # 作为训练的目标标签。
        inputs['labels'] = batch['label']
        return inputs

    dataset_dict = dataset_dict.map(encode_fn, batched=True,
                                    remove_columns=['review', 'label'])
    print(dataset_dict)

    # 6. 保存数据集
    dataset_dict.save_to_disk( ARROW_DIR )

    print("数据预处理结束...")

if __name__ == '__main__':
    preprocess()
