import os
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

# ===================== 初始化 =====================
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODEL = os.getenv("MODEL_NAME")

# ===================== 1. 工具实现 =====================
def calculator(expression: str) -> str:
    """数学计算器"""
    try:
        expression = expression.strip().strip('"').strip("'")
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

def web_search(query: str) -> str:
    """联网搜索"""
    try:
        query = query.strip().strip('"').strip("'")
        response = tavily.search(query=query, max_results=3)
        results = []
        for item in response["results"]:
            results.append(f"标题：{item['title']}\n摘要：{item['content']}\n来源：{item['url']}")
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败：{str(e)}"

# 工具映射表：工具名 → 实际函数
TOOL_MAP = {
    "calculator": calculator,
    "web_search": web_search
}

# ===================== 2. 工具定义（传给大模型的JSON Schema）=====================
# 这是给大模型"看"的工具说明书，严格遵循OpenAI格式
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行加减乘除等数学运算，输入是数学表达式字符串",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 123 + 456 * 789"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索实时信息、新闻、未知知识",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ===================== 3. 系统提示词 =====================
SYSTEM_PROMPT = """
你是一个智能助手，你可以使用提供的工具来解决用户的问题。
- 数学计算请使用 calculator 工具
- 实时信息、未知知识请使用 web_search 工具
- 如果你已经有足够的信息，请直接回答用户问题
"""

# ===================== 4. 滑动窗口记忆 =====================
def trim_messages(messages: list, max_keep: int = 6) -> list:
    if len(messages) <= max_keep + 1:
        return messages
    system_msg = messages[0]
    recent_msgs = messages[-max_keep:]
    return [system_msg] + recent_msgs

# ===================== 5. Agent核心循环 =====================
def run_agent(user_query: str, max_steps: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for step in range(max_steps):
        print(f"\n===== 第 {step+1} 步 =====")
        messages = trim_messages(messages, max_keep=6)

        # 调用大模型，传入工具定义
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEFINITION,  # 关键：传入工具定义
            tool_choice="auto",       # 让大模型自动决定是否调用工具
            temperature=0
        )
        message = response.choices[0].message
        print("LLM回复：")
        print(message.content if message.content else "（无文本内容，准备调用工具）")

        # 情况1：没有工具调用，直接返回最终答案
        if not message.tool_calls:
            print("\n===== Agent执行完成 =====")
            return message.content

        # 情况2：有工具调用，执行所有工具
        print(f"\n▶ 本次调用 {len(message.tool_calls)} 个工具")
        
        # 先把大模型的回复加入记忆
        messages.append(message)

        # 遍历执行每个工具调用
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            # 解析参数（大模型返回的是JSON字符串）
            import json
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"  - 工具：{tool_name}，参数：{tool_args}")

            # 查找并执行工具
            if tool_name not in TOOL_MAP:
                result = f"错误：不存在工具 {tool_name}"
            else:
                try:
                    result = TOOL_MAP[tool_name](**tool_args)
                except Exception as e:
                    result = f"工具执行错误：{str(e)}"

            # 把工具结果加入记忆，格式必须严格遵守
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": result
            })
            
            show_result = result[:200] + "..." if len(result) > 200 else result
            print(f"    返回：{show_result}")

    return "已达到最大执行步数，Agent未能完成任务。"

# ===================== 6. 测试 =====================
if __name__ == "__main__":
    # 测试1：数学题
    # question = "1234 * 5678 + 9876 等于多少？"
    
    # 测试2：搜索题
    question = "2024年巴黎奥运会中国金牌数是多少？用这个数乘以2等于多少？"
    
    print(f"用户问题：{question}")
    answer = run_agent(question)
    print(f"\n最终答案：{answer}")
