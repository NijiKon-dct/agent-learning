import os
os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from tavily import TavilyClient

# ===================== 初始化 =====================
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.7
)

# ===================== 工具 =====================
def web_search(query: str) -> str:
    """联网搜索资料"""
    try:
        query = query.strip().strip('"').strip("'")
        response = tavily.search(query=query, max_results=3)
        results = []
        for item in response["results"]:
            results.append(f"【{item['title']}】\n{item['content']}\n来源：{item['url']}")
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败：{str(e)}"

tools = {"web_search": web_search}
llm_with_tools = llm.bind_tools(list(tools.values()))

# ===================== 1. 定义状态 =====================
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str                # 写作主题
    draft: str                # 当前稿件
    review_passed: bool       # 审稿是否通过
    rewrite_count: int        # 重写次数

# ===================== 2. 定义三个角色 Agent =====================

## 角色1：研究员
def researcher_agent(state: TeamState) -> TeamState:
    """研究员：负责搜索主题相关资料，整理素材"""
    system_prompt = SystemMessage(content="""
你是专业的行业研究员。你的任务是针对用户指定的主题，通过搜索工具收集全面、准确的资料。
要求：
1. 只收集客观事实和数据，不加入主观观点
2. 标注信息来源
3. 整理成清晰的要点形式
4. 资料要覆盖主题的核心方面
""")
    
    messages = [
        system_prompt,
        HumanMessage(content=f"请围绕以下主题收集资料：{state['topic']}")
    ]
    
    # 工具调用循环
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        if not response.tool_calls:
            break
        messages.append(response)
        for tc in response.tool_calls:
            tc_name = tc["name"]
            tc_args = tc["args"]
            result = tools[tc_name](**tc_args)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc_name))
    
    print("✅ 研究员完成资料收集")
    return {
        "messages": [response],
        "draft": response.content
    }

## 角色2：撰稿人
def writer_agent(state: TeamState) -> TeamState:
    """撰稿人：根据研究员的素材撰写正式文章"""
    system_prompt = SystemMessage(content="""
你是专业的科技撰稿人。请根据提供的参考资料，撰写一篇结构清晰、语言流畅的完整文章。
要求：
1. 包含引言、主体分点、总结三部分
2. 逻辑连贯，重点突出
3. 字数控制在800字左右
4. 不要直接堆砌资料，要用自己的语言组织
""")
    
    messages = [
        system_prompt,
        HumanMessage(content=f"""
写作主题：{state['topic']}
参考资料：
{state['draft']}

请基于以上资料撰写正式文章。
""")
    ]
    
    response = llm.invoke(messages)
    
    print("✍️ 撰稿人完成文章撰写")
    return {
        "messages": [response],
        "draft": response.content,
        "rewrite_count": state.get("rewrite_count", 0) + 1
    }

## 角色3：审稿人
def reviewer_agent(state: TeamState) -> TeamState:
    """审稿人：检查文章质量，给出修改意见或决定通过"""
    system_prompt = SystemMessage(content="""
你是严格的审稿编辑。请审核这篇文章的质量，从准确性、逻辑性、结构完整性三个方面评判。
如果文章质量合格，请输出：【通过】
如果不合格，请输出：【重写】并逐条给出具体的修改意见。
注意：最多允许重写2次，第2次必须通过。
""")
    
    messages = [
        system_prompt,
        HumanMessage(content=f"""
文章主题：{state['topic']}
文章内容：
{state['draft']}

当前是第 {state.get('rewrite_count', 1)} 次撰写。请审核。
""")
    ]
    
    response = llm.invoke(messages)
    passed = "【通过】" in response.content
    
    print(f"🔍 审稿完成：{'通过' if passed else '打回重写'}")
    return {
        "messages": [response],
        "review_passed": passed
    }

# ===================== 3. 路由判断 =====================
def review_decision(state: TeamState) -> str:
    """审稿结果路由：通过就结束，不通过就返回撰稿人重写"""
    if state["review_passed"] or state.get("rewrite_count", 0) >= 2:
        return END
    return "writer"

# ===================== 4. 构建工作流图 =====================
workflow = StateGraph(TeamState)

# 添加节点
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

# 设置入口
workflow.set_entry_point("researcher")

# 定义边：研究员 → 撰稿人 → 审稿人 → （判断）
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    review_decision,
    {
        "writer": "writer",
        END: END
    }
)

# 编译
app = workflow.compile()

# ===================== 5. 运行测试 =====================
if __name__ == "__main__":
    topic = "AI Agent 开发框架 LangGraph 的核心优势和应用场景"
    
    print(f"📌 团队任务：撰写关于「{topic}」的文章\n")
    
    result = app.invoke({
        "topic": topic,
        "rewrite_count": 0,
        "review_passed": False
    })
    
    print("\n" + "="*50)
    print("📄 最终定稿：")
    print("="*50)
    print(result["draft"])
