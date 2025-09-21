from model.MyLlama import Transformer, LLMConfig
from transformers import AutoTokenizer
import torch
from llm_tokenizer.utils import ChatMLSFT
from llm_tokenizer.utils import SimpleStreamDecoder

def get_model_tokenizer(device):
    config = LLMConfig()
    model = Transformer(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./llm_tokenizer")
    return model, tokenizer
device = "cuda:0"
model, tokenizer = get_model_tokenizer(device)
model.load_state_dict(torch.load("checkpoints/model_sft_2.pth"))

while True:
    print("input your instruction")
    ins = input()
    conversation = [{
        "role":"user",
        "content":ins
    }]
    prompt = ChatMLSFT(conversation,True)
    enc = tokenizer(prompt,
                    truncation=True,
                    return_tensors='pt')
    # output = model.generate(input_ids=enc["input_ids"].to(device),
    #                attention_mask=enc["attention_mask"].to(device),
    #                max_length=100,
    #                eos_token_id=tokenizer.encode("<|im_end|>")[0])
    # print(tokenizer.decode(output.squeeze(0)))
    # print("==============")
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