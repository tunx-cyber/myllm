import os
import sys
__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.MyLlama import Transformer, LLMConfig
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from utils.llm_dataset import PretrainDataset

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def get_model_tokenizer(device):
    config = LLMConfig()
    model = Transformer(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("../llm_tokenizer")
    return model, tokenizer

def save_model(model, epoch,output_dir="../checkpoints"):
    if dist.get_rank() == 0:
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        torch.save(state_dict, save_path)
        model.train()

def set_optimizer(model, args):
    return optim.AdamW(model.parameters(), lr=args.learning_rate)

def get_dataloader(args, tokenizer):
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.get_world_size()>1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    return train_loader

def train_one_epoch(model:Transformer, optimizer, data_loader, device, ctx, scaler, epoch,args):
    start_time = time.time()
    iter_per_epoch = len(data_loader)
    model.train()
    for step, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with ctx:
            loss = model(input_ids, attention_mask, labels)["loss"]
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            if dist.get_rank() == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{iter_per_epoch}] "
                      f"Loss: {loss.item()*args.accumulation_steps:.4f} "
                      f"Time: {time.time() - start_time:.2f}s")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    ctx = torch.amp.autocast('cuda')

    init_distributed_mode()

    model,tokenizer = get_model_tokenizer(device=f"cuda:{dist.get_rank()}")
    if dist.get_rank() == 0:
        print(f'LLM可训练总参数量:{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    ddp_model = DistributedDataParallel(model, device_ids=[dist.get_rank()])

    dataloader = get_dataloader(args, tokenizer)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            ddp_model,
            set_optimizer(ddp_model, args),
            dataloader,
            device=f"cuda:{dist.get_rank()}",
            ctx=ctx,
            scaler=torch.amp.GradScaler('cuda'),
            epoch=epoch,
            args=args
        )
        save_model(ddp_model, epoch)
    dist.destroy_process_group()