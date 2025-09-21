from .data_preprocess import load_documents
from .embed import TextEmbedder
from .searcher import VectorStore
from utils.llm_dataset import ChatMLSFT
from llm_tokenizer.utils import SimpleStreamDecoder
def generate_answer(question, context, model, tokenizer,device):
    prompt = f"根据以下信息回答问题:\n{context}\n\n问题: {question}\n"
    # print(f"prompt:{prompt}")
    conversation = [{
        "role":"user",
        "content":prompt
    }]
    prompt = ChatMLSFT(conversation,True)
    enc = tokenizer(prompt,
                    truncation=True,
                    return_tensors='pt')
    decoder = SimpleStreamDecoder(tokenizer)
    for next_token in model.generate_stream(input_ids=enc["input_ids"].to(device),
                   attention_mask=enc["attention_mask"].to(device),
                   max_length=100,
                   eos_token_id=tokenizer.encode("<|im_end|>")[0]):
        new_text = decoder.decode(next_token.squeeze(0).item())
        if new_text == "<|im_end|>":
            break
        if new_text:
            print(new_text,end='')
    print()
class RAGSys:
    def __init__(self, document_folder, model, tokenizer, device):
        # 加载和处理文档
        self.documents = load_documents(document_folder)
        
        # 初始化嵌入器
        self.embedder = TextEmbedder()
        
        # 为文档生成嵌入向量
        self.document_embeddings = self.embedder.embed(self.documents)
        
        # 创建向量存储
        self.vector_store = VectorStore(self.documents, self.document_embeddings)
        
        # 设置生成模型和分词器
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def ask(self, question: str, k: int = 1) -> str:
        # 将问题转换为嵌入向量
        question_embedding = self.embedder.embed([question])[0]
        
        # 检索相关文档
        relevant_docs = self.vector_store.search(question_embedding, k=k)
        
        # 合并检索到的上下文
        context = "\n".join([doc for doc, score in relevant_docs])
        # print(f"context search result:{context}")
        # print("generated answer:")
        # 生成答案
        generate_answer(question, context, self.model, self.tokenizer, self.device)

