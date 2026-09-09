"""
    配置文件
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = ROOT_DIR / 'model'
LOG_DIR = ROOT_DIR / 'log'

RAW_DATA_FILE = RAW_DATA_DIR / 'cmn.txt'
TRAIN_RAW_FILE = PROCESSED_DATA_DIR / 'train-raw.csv'
TEST_RAW_FILE = PROCESSED_DATA_DIR / 'test-raw.csv'

ZH_VOCAB_FILE = PROCESSED_DATA_DIR / 'zh_vocab.txt'  # 词表文件
EN_VOCAB_FILE = PROCESSED_DATA_DIR / 'en_vocab.txt'

TRAIN_IDS_JSONL_FILE = PROCESSED_DATA_DIR / 'train_ids.jsonl'
TEST_IDS_JSONL_FILE = PROCESSED_DATA_DIR / 'test_ids.jsonl'

BEST_MODEL = MODEL_DIR / 'best_model.pt'

# 定义特殊token
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

# 训练超参数
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50
MAX_SEQ_LEN = 128

# 模型结构超参数
D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2