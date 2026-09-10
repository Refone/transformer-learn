"""
    配置文件
"""
from pathlib import Path

# 目录和文件
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = ROOT_DIR / 'model'
LOG_DIR = ROOT_DIR / 'log'
ARROW_DIR = PROCESSED_DATA_DIR / 'arrow'

RAW_DATA_FILE = RAW_DATA_DIR / 'online_shopping_10_cats.csv'

BEST_MODEL = MODEL_DIR / 'best_model.pt'

MODEL_NAME_OR_PATH = MODEL_DIR / 'bert-base-chinese'

# 训练超参数
LEARNING_RATE = 1e-5
BATCH_SIZE = 24
EPOCHS = 20
MAX_SEQ_LEN = 512

if __name__ == '__main__':
    # --- UNIT-TEST ---
    print(f'{ROOT_DIR=}')
    print(f'{DATA_DIR=}')
    print(f'{RAW_DATA_DIR=}')
    print(f'{PROCESSED_DATA_DIR=}')
    print(f'{MODEL_DIR=}')
    print(f'{LOG_DIR=}')
    print(f'{RAW_DATA_FILE=}')
    print(f'{BEST_MODEL=}')
    print(f'{MODEL_NAME_OR_PATH=}')
