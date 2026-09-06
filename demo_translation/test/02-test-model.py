from model import *

V_src = 10000
V_tgt = 12000
N = 64
L_src = 20
L_tgt = 26

model = TranslationModel(
    V_src, 1, L_src,
    V_tgt, 1, L_tgt,
)

src_ids = torch.randint(0, V_src, (64, L_src))
tgt_ids = torch.randint(0, V_tgt, (64, L_tgt))
src_key_padding_mask = torch.ones((64, L_src)).bool()
tgt_key_padding_mask = torch.ones((64, L_tgt)).bool()

output_ids = model(
    src_ids=src_ids,
    tgt_ids=tgt_ids,
    src_key_padding_mask=src_key_padding_mask,
    tgt_key_padding_mask=tgt_key_padding_mask,
    memory_key_padding_mask=src_key_padding_mask,
    tgt_is_causal=True
)
print(output_ids.shape)