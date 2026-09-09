"""
    构建数据集
"""

from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import *

# 获取数据加载器
def get_loader(train=True):
    path = ARROW_DIR / ('train' if train else 'test')
    dataset = load_from_disk(path)
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    padding_fn = DataCollatorWithPadding(tokenizer, padding='longest')

    # [当前方案] 动态 PADDING：预处理不 padding，装填时 padding，每批长度不一样，shape 不固定。
    #           静态 PADDING：在预处理阶段，给所有数据 padding 到统一长度，显存消耗大，计算慢。
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train, collate_fn=padding_fn)

    return loader

if __name__ == '__main__':
    print('-'*5, 'UNIT-TEST', '-'*5)

    train_loader = get_loader(train=True)
    test_loader = get_loader(train=False)

    i = 3
    for batch in train_loader:
        for k,v in batch.items():
            print(k, ' -> ', v.shape)
        if (i:= i-1) < 0:
            break

    i = 3
    for batch in test_loader:
        for k,v in batch.items():
            print(k, ' -> ', v.shape)
        if (i:= i-1) < 0:
                break
