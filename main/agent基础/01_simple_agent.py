import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL = os.getenv("MODEL_NAME")

# ===================== 1. 定义工具 =====================
def calculator(expression: str) -> str:
    """计算器工具，输入数学表达式，返回计算结果"""
    try:
        # 清洗参数：去除首尾引号、空格
        expression = expression.strip().strip('"').strip("'")
        # 教学示例用eval，生产环境禁止这么写，有安全风险
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

# 工具清单
TOOLS = [
    {
        "name": "calculator",
        "description": "数学计算器，用于执行加减乘除等数学运算，输入是数学表达式字符串",
        "function": calculator
    }
]

# ===================== 2. 系统提示词 =====================
SYSTEM_PROMPT = """
你是一个智能Agent，你可以使用工具来解决问题。

你必须严格按照以下格式输出：
1. 如果你需要调用工具，请输出：
Thought: 你思考的内容，为什么要调用这个工具
Action: 工具名称
Action Input: 工具参数，直接写表达式，不要加引号

2. 如果你已经得到最终答案，请输出：
Thought: 我已经得到最终答案
Final Answer: 你的最终答案

注意：
- 每次只能调用一个工具
- 调用工具后，你会得到工具返回的结果，再继续思考
- 不要编造不存在的工具
- 可用工具列表：
{tools_desc}
""".format(
    tools_desc="\n".join([f"- {tool['name']}: {tool['description']}" for tool in TOOLS])
)

# ===================== 3. Agent核心循环 =====================
def run_agent(user_query: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for step in range(max_steps):
        print(f"\n===== 第 {step+1} 步 =====")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0
        )
        llm_output = response.choices[0].message.content
        print("LLM输出：")
        print(llm_output)

        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            print("\n===== Agent执行完成 =====")
            return final_answer

        # 解析Action和Action Input
        action_match = re.search(r"Action: (.*?)(?=\n|$)", llm_output)
        input_match = re.search(r"Action Input: (.*?)(?=\n|$)", llm_output)

        if not action_match:
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": "输出格式错误，请严格按照要求的格式输出。"})
            continue

        action_name = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""
        
        # 关键修复：清洗参数，去除首尾引号
        action_input = action_input.strip().strip('"').strip("'")

        # 查找工具
        tool_func = None
        for tool in TOOLS:
            if tool["name"] == action_name:
                tool_func = tool["function"]
                break

        if not tool_func:
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"错误：不存在工具 {action_name}，请重新选择。"})
            continue

        # 调用工具
        print(f"\n调用工具：{action_name}，参数：{action_input}")
        observation = tool_func(action_input)
        print(f"工具返回结果：{observation}")

        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "已达到最大步数，Agent未能完成任务。"

# ===================== 4. 运行测试 =====================
if __name__ == "__main__":
    question = "1234 * 5678 + 9876 等于多少？"
    print(f"用户问题：{question}")
    answer = run_agent(question)
    print(f"\n最终答案：{answer}")
