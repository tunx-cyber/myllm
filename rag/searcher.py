import numpy as np
from typing import List, Tuple

class VectorStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """使用余弦相似度搜索最相关的文档"""
        # 归一化向量以便计算余弦相似度
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarities = np.dot(doc_norms, query_norm)
        
        # 获取最相似的k个文档
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        results = [(self.documents[i], similarities[i]) for i in top_k_indices]
        
        return results
