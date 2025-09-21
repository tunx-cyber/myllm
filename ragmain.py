from rag.rag import RAGSys
from model.MyLlama import Transformer, LLMConfig
from transformers import AutoTokenizer
import torch
def get_model_tokenizer(device):
    config = LLMConfig()
    model = Transformer(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./llm_tokenizer")
    return model, tokenizer
device = "cuda:0"
model, tokenizer = get_model_tokenizer(device)
model.load_state_dict(torch.load("checkpoints/model_sft_2.pth"))
# 初始化RAG系统
rag_system = RAGSys('mydata', model, tokenizer,device)

# 使用系统
question = "cat 命令的作用是什么？"
answer = rag_system.ask(question)