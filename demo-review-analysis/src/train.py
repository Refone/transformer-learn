"""
    训练脚本
"""
import torch
import time

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import *
from dataset import get_loader
from model import ReviewAnalysisModel

# 训练流程
def train():
    # 1. 定义设备
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device)

    # 2. 获取训练集加载器
    dataloader = get_loader(train=True)

    # 3. 定义模型
    model = ReviewAnalysisModel().to(device)

    # 4. 损失函数和优化器
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 日志写入器
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 6. 开始训练
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        print('='*10, f'Epoch {epoch+1}', '='*10)
        # 训练一个轮次，返回平均损失
        this_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'Loss: {this_loss:.4f}')

        # 记录日志
        writer.add_scalar('loss', this_loss, epoch+1)

        # 判断保存模型
        if this_loss < min_loss:
            min_loss = this_loss
            torch.save( model.state_dict(), BEST_MODEL )
            print(f'Best Model save to {BEST_MODEL}!')

    writer.close()

# 训练一个轮次
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0
    for batch in tqdm(dataloader, desc='[ Training ]'):
        targets = batch.pop('labels').to(device).float()
        inputs = { k:v.to(device) for k,v in batch.items() }

        # 1. 前向传播, 得到 (N, )
        outputs = model(**inputs)

        # 2. 计算损失
        loss = loss_fn(outputs, targets)

        # 3. 反向传播计算梯度
        loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加损失
        total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == '__main__':
    train()
