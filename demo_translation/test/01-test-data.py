from torch.utils.data import DataLoader

from config import TEST_IDS_JSONL_FILE
from data import *

preprocess()

dataset = TranslateDataset(TEST_IDS_JSONL_FILE)
print(dataset[0])

loader = get_loader(is_train=False)
for batch in loader:
    print(batch[0])
    print(batch[1])
    break