import streamlit as st
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
model.load_state_dict(torch.load("checkpoints/model_512_2.pth"))

# 设置页面配置
st.set_page_config(page_title="Hello LLM!",page_icon=":man-surfing:")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是一个专注写古诗的机器人，请输入诗歌风格和模仿的诗人进行创作！例如：请你模仿彭绩的风格，创作一首古诗词，表达对亲友的欣赏和期待。请注重语言的质朴和情感的真挚。"}]

# 定义响应生成器（模拟打字机效果）
def generate_response(messages):
    # 这里是模拟响应，实际应用中可替换为真实的AI模型API调用
    # 实际上要调用最近的几个聊天记录
    prompt = ChatMLSFT(messages,True)
    decoder = SimpleStreamDecoder(tokenizer)
    enc = tokenizer(prompt,
                    truncation=True,
                    return_tensors='pt')
    full_response = ""
    for next_token in model.generate_stream(input_ids=enc["input_ids"].to(device),
                   attention_mask=enc["attention_mask"].to(device),
                   max_length=1024,
                   eos_token_id=tokenizer.encode("<|im_end|>")[0]):
        new_text = decoder.decode(next_token.squeeze(0).item())
        if new_text == "<|im_end|>":
            break
        if new_text:
            full_response += new_text
            yield new_text
            
    return full_response
    

# # 页面标题
# st.title("🤖 多轮聊天应用演示")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("Say something to LLM"):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成助手响应
    with st.chat_message("assistant"):
        # response = st.write_stream(response_generator(prompt))
        message_placeholder = st.empty()
        full_response = ""
        
        # 调用生成器获取流式响应
        for chunk in generate_response(st.session_state.messages):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # 添加助手响应到历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 创建更丰富的界面布局
st.sidebar.title("聊天设置")

# 添加聊天控制选项
clear_chat = st.sidebar.button("清空聊天记录")
if clear_chat:
    st.session_state.messages = []
    st.rerun()

# # 添加样式美化
# st.markdown("""
# <style>
#     .stChatMessage {
#             padding: 1rem;
#             border-radius: 0.5rem;
#             margin-bottom: 1rem;
#         }
#     .stChatInput {
#         position: fixed;
#         bottom: 20px;
#     }
# </style>
# """, unsafe_allow_html=True)
