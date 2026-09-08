from config import *
from tqdm import tqdm
from model import *
from data import *
import torch

def train_chinese_to_english():
    device = 'cuda' if torch.cuda.is_available() else \
            'mps' if torch.backends.mps.is_available() else \
            'cpu'

    preprocess()
    en_tokenizer = EnTokenizer.create_from_vocab_file(EN_VOCAB_FILE)
    zh_tokenizer = ZhTokenizer.create_from_vocab_file(ZH_VOCAB_FILE)

    data_loader = TranslateDataLoader(is_train=True)

    model = TranslationModel(
        src_tokenizer=zh_tokenizer,
        tgt_tokenizer=en_tokenizer,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_loss = float('inf')
    for epochs in range(EPOCHS):
        print('='*10, f' Epoch [{epochs + 1}/{EPOCHS}] ', '='*10)
        # 训练一个轮次, 返回平均损失
        train_loss = train_one_epoch(decoder_tokenizer=en_tokenizer, model=model,
                                     train_loader=data_loader, test_loader=data_loader,
                                     optimizer=optimizer, loss_fn=loss_fn, device=device)
        print(f'训练误差: {train_loss}')

        if train_loss < min_loss:
            min_loss = train_loss
            torch.save( model.state_dict(), BEST_MODEL)
            print('最佳模型保存成功')

def train_one_epoch(model, decoder_tokenizer,
                    train_loader, test_loader,
                    optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    src_pad_id = model.src_embedding.padding_idx
    tgt_pad_id = model.tgt_embedding.padding_idx
    for inputs, targets in tqdm(train_loader, desc='训练'):
        inputs, targets = inputs.to(device), targets.to(device)

        # 0. 准备工作,截取 decoder_inputs 和 decoder_outputs
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]
        # 定义掩码
        src_key_padding_mask = (inputs == src_pad_id).to(device)
        tgt_key_padding_mask = (decoder_inputs == tgt_pad_id).to(device)

        # 1. 前向传播
        decoder_outputs = model(
            src_ids=inputs, tgt_ids=decoder_inputs,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            is_causal=True
        )   # (B, Lt+1, Vt)

        # 2. 计算损失
        # decoder_outputs: (B, T, Vt)
        # decoder_targets: (B, T)
        loss = loss_fn(decoder_outputs.mT, decoder_targets)

        # 3. 反向传播
        loss.backward()

        # 4. 更新梯度
        optimizer.step()

        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加损失
        total_loss += loss.item()

    return total_loss / len(train_loader)