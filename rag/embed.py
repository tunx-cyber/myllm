from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import numpy as np
'''
- 'BAAI/bge-small-en' (英文小模型)
- 'BAAI/bge-small-zh' (中文小模型)
- 'BAAI/bge-base-en' (英文基础模型)
'''
class TextEmbedder:
    def __init__(self, model_name='BAAI/bge-small-zh'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为嵌入向量"""
        embeddings = []
        
        for text in texts:
            # 标记化
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 使用平均池化获取句子嵌入
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)
        
        return np.array(embeddings)
