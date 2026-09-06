import os
os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # 核心：检查点存储器
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from tavily import TavilyClient

# ===================== 初始化环境 =====================
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0,
    timeout=120,
    max_retries=2
)

# ===================== 工具定义 =====================
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
            results.append(f"【{item['title']}】{item['content']}")
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败：{str(e)}"

tools = {"calculator": calculator, "web_search": web_search}
llm_with_tools = llm.bind_tools(list(tools.values()))

# ===================== 状态定义 =====================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ===================== 节点定义 =====================
def agent_node(state: AgentState):
    print("正在调用大模型，请稍候（首次联网查询可能需要 10~30 秒）...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        print(f"执行工具：{tc['name']}")
        result = tools[tc["name"]](**tc["args"])
        results.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": results}

def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# ===================== 【核心】构建带持久化的图 =====================
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# --- 持久化关键1：创建检查点存储器 ---
memory = MemorySaver()

# --- 持久化关键2：编译图时绑定 checkpointer ---
app = workflow.compile(checkpointer=memory)

# ===================== 运行演示 =====================
if __name__ == "__main__":
    # --- 持久化关键3：每个会话一个唯一 thread_id ---
    # 相当于给这个用户的对话开一个独立存档
    config = {"configurable": {"thread_id": "user_001"}}

    system_prompt = SystemMessage(content="""
    你是智能助手，可以使用搜索和计算器工具。
    请记住对话历史，用户后续提问会基于上下文。
    """)

    # ========== 第一轮对话 ==========
    print("=" * 50)
    print("📌 第一轮对话（新建会话）")
    print("=" * 50)
    result = app.invoke(
        {"messages": [system_prompt, HumanMessage(content="2024年巴黎奥运会中国金牌数是多少？")]},
        config=config  # 传入会话配置
    )
    print("回答：", result["messages"][-1].content)

    # ========== 第二轮对话（同一个 thread_id，自动继承上下文） ==========
    print("\n" + "=" * 50)
    print("📌 第二轮对话（同一会话，自动带上下文）")
    print("=" * 50)
    # 注意：这里只传新的用户消息，不用再传 system 和历史！
    result = app.invoke(
        {"messages": [HumanMessage(content="这个数乘以 7 等于多少？")]},
        config=config  # 同一个 thread_id，自动加载历史状态
    )
    print("回答：", result["messages"][-1].content)

    # ========== 查看当前状态快照 ==========
    print("\n" + "=" * 50)
    print("🔍 当前会话状态快照")
    print("=" * 50)
    state = app.get_state(config)
    print(f"当前状态消息数：{len(state.values['messages'])}")
    print(f"下一个待执行节点：{state.next}")
    print(f"已执行节点数：{len(state.tasks)}")

    # ========== 第三个会话（不同 thread_id，状态完全隔离） ==========
    print("\n" + "=" * 50)
    print("📌 新用户会话（状态完全隔离）")
    print("=" * 50)
    config2 = {"configurable": {"thread_id": "user_002"}}  # 新的 ID
    result = app.invoke(
        {"messages": [system_prompt, HumanMessage(content="我们刚才聊了什么？")]},
        config=config2
    )
    print("回答：", result["messages"][-1].content)

