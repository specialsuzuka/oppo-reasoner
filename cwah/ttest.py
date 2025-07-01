import os
from openai import OpenAI
# Initialize OpenAI client with environment variables

from typing import List
# 初始化客户端（自动从环境变量 OPENAI_API_KEY 读取密钥）
client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
    base_url=os.getenv("API_BASE")
)
from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionMessageParam # 如需代理：OpenAI(base_url="http://your-proxy.com/v1")

# 定义对话消息（类型提示增强安全性）
messages: List[ChatCompletionMessageParam] = [
    {"role": "system", "content": "你是一个专业翻译官"},
    {"role": "user", "content": "将以下英文翻译成中文: 'Hello, how are you?'"}
]

try:
    # 同步调用 GPT-4-turbo
    response = client.chat.completions.create(
        model="deepseek-r1",  # 或 "gpt-4"
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        stream=False  # 设为 True 可启用流式响应
    )
    
    # 打印完整响应
    print("翻译结果:", response.choices[0].message.content)
    print("消耗 Token 数:", response.usage.total_tokens)

except APIError as e:
    print(f"API 错误: {e.status_code} - {e.message}")
except Exception as e:
    print(f"其他错误: {type(e).__name__} - {e}")
print(os.environ.get("OPENAI_KEY"))
print(os.getenv("OPENAI_KEY"))
base_url=os.getenv("API_BASE")
print(base_url)