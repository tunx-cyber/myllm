import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

# 简单的文本分割函数
def split_text(text: str, chunk_size: int = 50, overlap: int = 20) -> List[str]:
    """将文本分割成重叠的块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 读取文档（这里以TXT为例，PDF需要额外处理）
def load_documents(folder_path: str) -> List[str]:
    """从文件夹加载所有文本文件"""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = split_text(text)
                documents.extend(chunks)
    # print(f"加载了 {len(documents)} 个文本块")
    return documents



