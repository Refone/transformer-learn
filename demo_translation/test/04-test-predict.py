from data import *
from model import *

device = 'cpu'  # mps 有 padding mask 的兼容性问题

en_tokenizer = EnTokenizer.create_from_vocab_file(EN_VOCAB_FILE)
zh_tokenizer = ZhTokenizer.create_from_vocab_file(ZH_VOCAB_FILE)

model = TranslationModel(
    src_tokenizer=zh_tokenizer,
    tgt_tokenizer=en_tokenizer).to(device)
model.load_state_dict( torch.load( BEST_MODEL, map_location=device ) )

text = '你好'
sentences = model.predict(text)

print(sentences)