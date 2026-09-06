from data import preprocess
from model import *
from tokenizer import EnTokenizer, ZhTokenizer

N = 64
L_src = 20
L_tgt = 26

preprocess()

en_tokenizer = EnTokenizer.create_from_vocab_file(EN_VOCAB_FILE)
zh_tokenizer = ZhTokenizer.create_from_vocab_file(ZH_VOCAB_FILE)

model = TranslationModel(
    src_tokenizer=zh_tokenizer,
    tgt_tokenizer=en_tokenizer,
)

print(zh_tokenizer.vocab_size)
print(en_tokenizer.vocab_size)

src_ids = torch.randint(0, zh_tokenizer.vocab_size, (64, L_src))
tgt_ids = torch.randint(0, en_tokenizer.vocab_size, (64, L_tgt))
src_key_padding_mask = torch.zeros((64, L_src)).bool()
tgt_key_padding_mask = torch.zeros((64, L_tgt)).bool()

output_ids = model(
    src_ids=src_ids,
    tgt_ids=tgt_ids,
    src_key_padding_mask=src_key_padding_mask,
    tgt_key_padding_mask=tgt_key_padding_mask,
    memory_key_padding_mask=src_key_padding_mask,
    is_causal=True
)
print(output_ids.shape)