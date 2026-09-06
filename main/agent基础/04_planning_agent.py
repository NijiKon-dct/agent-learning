import os
import json
os.environ["NO_PROXY"] = "*"

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

# ===================== 工具库 =====================
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

TOOL_MAP = {
    "calculator": calculator,
    "web_search": web_search
}

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
                        "description": "要计算的数学表达式"
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
            "description": "联网搜索实时信息、新闻、技术知识",
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

# ===================== 1. 规划器：拆解任务 =====================
def make_plan(task: str) -> list:
    """
    接收用户的大任务，拆解成有序的子任务列表
    """
    system_prompt = """
你是一个专业的任务规划师。请将用户的复杂任务拆解成3-5个有序的子任务。
要求：
1. 子任务要按执行顺序排列
2. 每个子任务要具体、可执行
3. 只输出子任务列表，每行一个，格式为：
1. 子任务1
2. 子任务2
...
不要输出多余的解释。
"""
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请拆解这个任务：{task}"}
        ],
        temperature=0
    )
    
    plan_text = response.choices[0].message.content.strip()
    # 解析成列表
    tasks = []
    for line in plan_text.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # 去掉序号
            task_content = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
            tasks.append(task_content)
    
    return tasks

# ===================== 2. 执行器：执行单个子任务 =====================
def execute_sub_task(sub_task: str, context: str = "") -> str:
    """
    执行单个子任务，返回执行结果
    context: 之前子任务的执行结果，作为上下文参考
    """
    messages = [
        {"role": "system", "content": "你是一个任务执行者，使用提供的工具完成子任务。如果有之前的上下文信息，请结合上下文回答。"},
        {"role": "user", "content": f"已完成的上下文：\n{context}\n\n当前需要完成的子任务：{sub_task}"}
    ]
    
    max_steps = 5
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEFINITION,
            tool_choice="auto",
            temperature=0
        )
        message = response.choices[0].message
        
        if not message.tool_calls:
            return message.content
        
        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            if tool_name not in TOOL_MAP:
                result = f"错误：不存在工具 {tool_name}"
            else:
                try:
                    result = TOOL_MAP[tool_name](**tool_args)
                except Exception as e:
                    result = f"工具执行错误：{str(e)}"
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": result
            })
    
    return "子任务执行超时"

# ===================== 3. 汇总器：整合所有结果 =====================
def summarize_result(task: str, all_results: list) -> str:
    """
    整合所有子任务的结果，输出最终报告
    """
    system_prompt = """
你是一个报告撰写者。请根据所有子任务的执行结果，整合出一份完整、清晰的最终答案。
要求结构清晰、重点突出，不要简单罗列，要形成有逻辑的完整回答。
"""
    
    context = "\n\n".join([f"子任务{i+1}结果：\n{res}" for i, res in enumerate(all_results)])
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"原始任务：{task}\n\n{context}"}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

# ===================== 4. 主Agent入口 =====================
def run_planning_agent(task: str) -> str:
    print(f"📌 收到任务：{task}")
    
    # 第一步：制定计划
    print("\n🗓️  正在制定执行计划...")
    plan = make_plan(task)
    for i, t in enumerate(plan):
        print(f"  {i+1}. {t}")
    
    # 第二步：逐个执行子任务
    all_results = []
    context = ""
    for i, sub_task in enumerate(plan):
        print(f"\n⚙️  正在执行第 {i+1} 个子任务：{sub_task}")
        result = execute_sub_task(sub_task, context)
        all_results.append(result)
        context += f"\n子任务{i+1}：{sub_task}\n结果：{result}\n"
        print(f"  ✅ 完成，结果摘要：{result[:100]}..." if len(result)>100 else f"  ✅ 完成：{result}")
    
    # 第三步：汇总输出
    print("\n📝 正在整合最终结果...")
    final_report = summarize_result(task, all_results)
    
    print("\n===== 任务完成 =====")
    return final_report

# ===================== 测试运行 =====================
if __name__ == "__main__":
    task = "调研一下目前主流的3个Agent开发框架，对比它们的优缺点和适用场景，输出一份对比总结"
    final = run_planning_agent(task)
    print("\n" + final)
