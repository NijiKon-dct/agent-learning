"""模型调用封装（OpenAI 兼容 API：DeepSeek / 通义 / 智谱 / OpenAI 都能用）

运行前先设置环境变量（Windows PowerShell）：
    $env:LLM_API_KEY="sk-xxx"
    $env:LLM_BASE_URL="https://api.deepseek.com"   # 不设则用默认
    $env:LLM_MODEL="deepseek-chat"                 # 不设则用默认
"""
import os
import sys

from openai import OpenAI

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
# 其他选择：
#   通义  https://dashscope.aliyuncs.com/compatible-mode/v1  (qwen-plus)
#   智谱  https://open.bigmodel.cn/api/paas/v4              (glm-4-flash)
#   OpenAI https://api.openai.com/v1                        (gpt-4o-mini)
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

if not API_KEY:
    sys.exit("请先设置环境变量 LLM_API_KEY，设置方法见本文件顶部注释。")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(messages: list[dict]) -> str:
    """普通（非流式）调用，返回完整文本。

    为什么不用流式？ReAct 循环必须拿到完整输出才能解析出 Action。
    为什么用 stop 参数？模型经常自己把 Observation 编出来，
    用停止符强制它在写下 Observation 之前停住，把执行权交还给循环。
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,        # Agent 要的是稳定，不是创意
        stop=["Observation:"],
    )
    return resp.choices[0].message.content
