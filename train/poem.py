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
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from llm_tokenizer.utils import ChatMLSFT
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
    model.load_state_dict(torch.load("../checkpoints/model_512_2.pth"))
    return model, tokenizer

def save_model(model, epoch,output_dir="../checkpoints"):
    if dist.get_rank() == 0:
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"model_poem_{epoch}.pth")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        torch.save(state_dict, save_path)
        model.train()

def set_optimizer(model, args):
    return optim.AdamW(model.parameters(), lr=args.learning_rate)

class PoemDataset(Dataset):
    def __init__(self,data,tokenizer,max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        print(f"total samples are {len(self.data)}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        conversation = [{"role":"user","content":f"{example["input"]}"},
                        {"role":"assistant","content":f"{example["output"]}"}]
        prompt = ChatMLSFT(conversation)
        encoded  = self.tokenizer(prompt,
                                    max_length=self.max_len,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
        input_ids = encoded["input_ids"].flatten()
        attention_mask = encoded["attention_mask"].flatten()
        label = input_ids.clone()

        loss_mask = [0] * len(label)
        i = 0
        while i < len(label):
            if label[i:i+len(self.bos_id)] == self.bos_id:# find the assistant part
                start = i+len(self.bos_id)
                end = start
                while end < len(label):
                    if label[end:end+len(self.eos_id)] == self.eos_id:# find the end of assistant part
                        break
                    end += 1
                for j in range(start+1, min(end + len(self.eos_id) + 1, self.max_length)): # make assistant part be seen by model
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(label) else len(label)
            else:
                i += 1
        label[loss_mask == 0] = -100# 屏蔽gpt不需要回答的部分。

        input_ids = input_ids[:-1].detach().clone()
        label = label[1:].detach().clone()
        attention_mask = attention_mask[1:].detach().clone().type_as(label)

        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":label
        }

def get_dataloader(args, tokenizer):
    from datasets import load_dataset
    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset = load_dataset("Million/Chinese-Poems")
    dataset = dataset["train"]
    dataset.shuffle(42)
    train_ds = PoemDataset(dataset, tokenizer, max_len=args.max_seq_len)
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument('--max_seq_len', default=256, type=int)
    # parser.add_argument("--data_path", type=str, default="../dataset/sft_2048.jsonl")
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