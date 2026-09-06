import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

# ===================== 初始化环境 =====================
load_dotenv()
# 大模型客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
# 搜索客户端
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODEL = os.getenv("MODEL_NAME")

# ===================== 1. 工具库 =====================
def calculator(expression: str) -> str:
    """数学计算器：输入数学表达式字符串，返回计算结果"""
    try:
        # 参数清洗：去除首尾引号、空格
        expression = expression.strip().strip('"').strip("'")
        # 教学演示用eval，生产环境禁止直接使用
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

def web_search(query: str) -> str:
    """联网搜索：输入搜索关键词，返回结构化的网页摘要信息"""
    try:
        query = query.strip().strip('"').strip("'")
        # 调用Tavily搜索，只返回最相关的3条结果
        response = tavily.search(query=query, max_results=3)
        # 格式化结果，方便大模型读取
        results = []
        for item in response["results"]:
            results.append(
                f"【标题】{item['title']}\n"
                f"【摘要】{item['content']}\n"
                f"【来源】{item['url']}"
            )
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败：{str(e)}"

# 工具注册表：Agent可调用的所有工具
TOOLS = [
    {
        "name": "calculator",
        "description": "数学计算器，用于执行加减乘除等数学运算，输入是数学表达式字符串",
        "function": calculator
    },
    {
        "name": "web_search",
        "description": "联网搜索工具，用于获取实时信息、新闻、未知知识等，输入是搜索关键词",
        "function": web_search
    }
]

# ===================== 2. 系统提示词（Agent行为准则） =====================
SYSTEM_PROMPT = """
你是一个智能Agent，你可以使用工具来解决问题。

你必须严格按照以下格式输出：
1. 如果你需要调用工具，请输出：
Thought: 你的思考过程，说明为什么要调用这个工具
Action: 工具名称
Action Input: 工具参数，直接写内容，不要加引号

2. 如果你已经得到最终答案，请输出：
Thought: 我已经得到最终答案
Final Answer: 你的最终答案

注意规则：
- 每次只能调用一个工具
- 数学计算必须使用calculator工具，禁止自己口算
- 实时信息、未知知识必须使用web_search工具
- 可用工具列表：
{tools_desc}
""".format(
    tools_desc="\n".join([f"- {tool['name']}: {tool['description']}" for tool in TOOLS])
)

# ===================== 3. 滑动窗口记忆管理 =====================
def trim_messages(messages: list, max_keep: int = 6) -> list:
    """
    裁剪对话历史，防止上下文溢出token限制
    - 永远保留第一条system提示词
    - 只保留最近max_keep条对话消息
    - max_keep建议设为偶数，对应完整的几轮交互
    """
    # 总消息数没超过限制，直接返回
    if len(messages) <= max_keep + 1:
        return messages
    
    system_msg = messages[0]  # 保留系统规则
    recent_msgs = messages[-max_keep:]  # 保留最近的对话
    return [system_msg] + recent_msgs

# ===================== 4. Agent核心执行循环 =====================
def run_agent(user_query: str, max_steps: int = 8) -> str:
    # 初始化短期记忆（对话历史）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for step in range(max_steps):
        print(f"\n===== 第 {step+1} 步 =====")
        
        # 【关键】每次调用大模型前裁剪记忆
        messages = trim_messages(messages, max_keep=6)
        
        # 调用大模型进行思考
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0
        )
        llm_output = response.choices[0].message.content
        print("LLM思考：")
        print(llm_output)

        # 判断是否得出最终答案
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            print("\n===== Agent执行完成 =====")
            return final_answer

        # 解析工具调用
        action_match = re.search(r"Action: (.*?)(?=\n|$)", llm_output)
        input_match = re.search(r"Action Input: (.*?)(?=\n|$)", llm_output)

        # 格式错误处理：把错误反馈给大模型，让它自己修正
        if not action_match:
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": "输出格式错误，请严格按照要求的格式输出。"})
            continue

        action_name = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""
        # 参数清洗
        action_input = action_input.strip().strip('"').strip("'")

        # 查找对应的工具函数
        tool_func = None
        for tool in TOOLS:
            if tool["name"] == action_name:
                tool_func = tool["function"]
                break

        # 工具不存在处理
        if not tool_func:
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"错误：不存在工具 {action_name}，请重新选择。"})
            continue

        # 执行工具调用
        print(f"\n▶ 调用工具：{action_name}")
        print(f"▶ 参数：{action_input}")
        observation = tool_func(action_input)
        # 结果太长就截断显示，避免刷屏
        show_obs = observation[:300] + "..." if len(observation) > 300 else observation
        print(f"▶ 工具返回：{show_obs}")

        # 将工具结果写入记忆
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "已达到最大执行步数，Agent未能完成任务。"

# ===================== 5. 测试运行 =====================
if __name__ == "__main__":
    # 测试1：数学计算（应该自动选calculator）
    # question = "999 × 999 + 1234 等于多少？"
    
    # 测试2：实时信息（应该自动选web_search）
    question = "2024年巴黎奥运会中国代表团获得了多少枚金牌？"
    
    print(f"用户问题：{question}")
    answer = run_agent(question)
    print(f"\n最终答案：{answer}")
