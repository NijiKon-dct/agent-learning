import os
import json
os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from tavily import TavilyClient

# ===================== 初始化 =====================
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 大模型
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0
)

# ===================== 1. 定义工具 =====================
def calculator(expression: str) -> str:
    """数学计算器，执行加减乘除运算"""
    try:
        expression = expression.strip().strip('"').strip("'")
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

def web_search(query: str) -> str:
    """联网搜索实时信息"""
    try:
        query = query.strip().strip('"').strip("'")
        response = tavily.search(query=query, max_results=3)
        results = []
        for item in response["results"]:
            results.append(f"标题：{item['title']}\n摘要：{item['content']}\n来源：{item['url']}")
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败：{str(e)}"

# 工具映射
tools = {
    "calculator": calculator,
    "web_search": web_search
}

# 把工具绑定给大模型（自动生成工具定义）
llm_with_tools = llm.bind_tools(list(tools.values()))

# ===================== 2. 定义状态 State =====================
class AgentState(TypedDict):
    """Agent的共享状态，目前只存消息列表"""
    messages: Annotated[list, add_messages]

# ===================== 3. 定义节点 =====================
def agent_node(state: AgentState) -> AgentState:
    """
    大模型决策节点：接收当前状态，调用大模型，决定是否调用工具
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # 返回更新后的状态：把大模型的回复加入消息列表
    return {"messages": [response]}

def tool_node(state: AgentState) -> AgentState:
    """
    工具执行节点：执行所有工具调用，返回结果
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        print(f"▶ 执行工具：{tool_name}，参数：{tool_args}")
        
        if tool_name in tools:
            try:
                result = tools[tool_name](**tool_args)
            except Exception as e:
                result = f"执行错误：{str(e)}"
        else:
            result = f"错误：不存在工具 {tool_name}"
        
        tool_results.append(
            ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name)
        )
    
    # 返回更新后的状态：把工具结果加入消息列表
    return {"messages": tool_results}

# ===================== 4. 定义条件边：判断是否继续调用工具 =====================
def should_continue(state: AgentState) -> str:
    """
    路由函数：根据最后一条消息判断下一步
    - 有工具调用 → 去工具节点
    - 没有工具调用 → 结束
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ===================== 5. 构建图 =====================
# 1. 创建状态图
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 3. 设置入口点
workflow.set_entry_point("agent")

# 4. 添加边
# 条件边：agent节点之后，根据条件决定去tools还是结束
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
# 普通边：tools执行完，回到agent继续思考
workflow.add_edge("tools", "agent")

# 5. 编译图
app = workflow.compile()

# ===================== 6. 运行测试 =====================
if __name__ == "__main__":
    system_prompt = SystemMessage(content="""
你是智能助手，可以使用工具解决问题。
- 数学计算用 calculator
- 实时信息用 web_search
- 有足够信息就直接回答
""")
    
    user_query = "2024年巴黎奥运会中国金牌数是多少？乘以3等于多少？"
    
    print(f"用户问题：{user_query}\n")
    
    # 运行图，传入初始状态
    result = app.invoke({
        "messages": [
            system_prompt,
            HumanMessage(content=user_query)
        ]
    })
    
    # 输出最终答案
    final_answer = result["messages"][-1].content
    print(f"\n最终答案：\n{final_answer}")
