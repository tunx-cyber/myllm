from torch.utils.data import Dataset
import json
from llm_tokenizer.utils import ChatMLSFT
'''
TODO: 优化不要一次性全部加载,使用MMAP技术,支持分布式数据集
'''
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ids = encoding.input_ids.squeeze()
        attention_mask = (ids != self.tokenizer.pad_token_id)

        input_ids = ids[:-1].detach().clone()
        label = ids[1:].detach().clone()
        attention_mask = attention_mask[1:].detach().clone().type_as(label)
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":label
        }
    
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []
        with open(data_path,'r', encoding="utf-8") as f:
            for line_no, line in enumerate(f,start=1):
                data = json.loads(line.strip())
                self.conversations.append(data)
        self.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        print(f"total samples are {len(self.conversations)}")

    def _creat_prompt(self, sample):
        conversations = sample["conversations"]
        return ChatMLSFT(conversations,False)
    
    def __getitem__(self, index):
        sample = self.conversations[index]
        prompt = self._creat_prompt(sample)
        encoded  = self.tokenizer(prompt,
                                    max_length=self.max_length,
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

    def __len__(self):
        return len(self.conversations)
    

class VLLMDataset(Dataset):
    def __init__():
        pass