import datetime
from llm_tokenizer.utils import ChatMLSFT
from model.MyLlama import Transformer, LLMConfig
from transformers import AutoTokenizer
import torch
def get_model_tokenizer(device):
    config = LLMConfig()
    model = Transformer(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./llm_tokenizer")
    model.load_state_dict(torch.load("checkpoints/model_sft_2.pth"))
    return model, tokenizer
# 定义一个获取当前时间的工具函数
def get_current_time():
    """
    获取当前的日期和时间。
    返回格式：YYYY-MM-DD HH:MM:SS
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# 你可以定义更多工具，比如查询天气、计算器等
# def get_weather(city): ...
# def calculator(expression): ...

# 创建一个工具列表，供LLM选择
tools = [
    {
        "name": "get_current_time",
        "func": get_current_time,
        "description": "获取当前的日期和时间。当用户询问现在几点、今天日期、现在是什么时候时使用。"
    }
    # ... 可以加入其他工具
]

class Agent:
    def __init__(self, model, tokenizer, actions):
        self.model = model
        self.tokenizer = tokenizer
        self.actions = actions
    
    def run(self, query):
        sys_prompt = f'''You are a help AI assitant and you can use the following tools
{str(self.actions)}
Please strictly follow the following steps when replying to users:
1. Understand the user\'s request.
2. If the user\'s request requires calling a tool, generate a JSON object strictly containing the following fields:
  - \'tool_name\': The name of the tool to be called, which must be one of the following: [{', '.join([t['name'] for t in tools])}]
3. If the request can be answered without a tool, or if the final answer has already been obtained, reply to the user directly.'''
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ]
        prompt = ChatMLSFT(messages,True)
        enc = self.tokenizer(prompt,
                        truncation=True,
                        return_tensors='pt')
        device = "cuda:0"
        model_response = self.model.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_length=100,
            eos_token_id=self.tokenizer.encode("<|im_end|>")[0]
        )
        model_response = self.tokenizer.decode(model_response)
        print(model_response)
        # 尝试解析模型的响应，看它是不是一个JSON（表示它想调用工具）
        try:
            import json
            action = json.loads(model_response)
            # 如果解析成功，说明模型想要调用工具
            tool_name = action.get("tool_name")
            # thought = action.get("thought")

            # print(f"AI思考: {thought}")
            print(f"AI决定调用工具: {tool_name}")

            # 找到对应的工具函数并执行
            for tool in tools:
                if tool['name'] == tool_name:
                    tool_result = tool['func']()  # 执行工具函数，例如get_current_time()
                    print(f"工具执行结果: {tool_result}")

                    # 将工具结果追加到对话历史中，再次发送给模型，让它来总结回复
                    messages.append({"role": "assistant", "content": model_response}) # 记录模型刚才的JSON决策
                    messages.append({"role": "user", "content": f"Tool Result: {tool_result}. Please now answer the user's original question based on this result."})

                    # 第二次调用模型，让它基于工具结果生成最终回答
                    print("第二次调用")
                    # final_answer = second_response.choices[0].message.content
                    return "tool"

            # 如果没找到对应的工具
            return f"Error: Tool '{tool_name}' not found."

        except json.JSONDecodeError:
            # 如果模型的响应不是JSON，说明它想直接回答
            print("AI决定直接回答。")
            return model_response

# 测试Agent
user_input = "现在几点了？"
model, tokenizer = get_model_tokenizer("cuda:0")
agent = Agent(model,tokenizer,tools)
agent.run(user_input)
