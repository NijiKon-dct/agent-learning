import os
os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# ===================== 初始化 =====================
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.2,
    timeout=60,
    max_retries=1
)

# ===================== 状态定义 =====================
class PublishState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    draft: str
    review_opinion: str
    status: str
    rewrite_count: int  # 新增：重写次数，限制最多2次

# ===================== 节点定义 =====================
def writer_node(state: PublishState):
    """撰稿节点：根据主题或修改意见撰写/修改文章"""
    system_prompt = SystemMessage(content="""
    你是专业撰稿人。请围绕主题写一篇简洁的正式短文。
    要求：结构清晰，重点突出，严格控制在200字以内。
    如有修改意见，请严格按照意见修改。
    """)
    
    if state.get("review_opinion"):
        prompt = f"""
        主题：{state['topic']}
        修改意见：{state['review_opinion']}
        当前稿：{state['draft']}
        请按意见修改，200字内。
        """
    else:
        prompt = f"围绕主题写短文：{state['topic']}，200字内。"
    
    print("\n✍️  正在撰写：", end="", flush=True)
    full_content = ""
    for chunk in llm.stream([system_prompt, HumanMessage(content=prompt)]):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_content += content
    print("\n")
    
    return {
        "draft": full_content,
        "status": "reviewing",
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "messages": [AIMessage(content=full_content)]
    }

def publish_node(state: PublishState):
    """发布节点：审核通过后正式发布"""
    print("\n📢 系统：文章已正式发布！")
    return {"status": "published"}

def review_gate(state: PublishState):
    """审核网关：判断是等待审核、发布还是重写"""
    opinion = state.get("review_opinion", "")
    rewrite_count = state.get("rewrite_count", 0)
    
    if not opinion:
        # 没有审核意见 → 暂停等待人工输入
        return "wait"
    elif "通过" in opinion or "同意" in opinion or "ok" in opinion.lower():
        # 审核通过 → 发布
        return "publish"
    elif rewrite_count >= 2:
        # 达到最大重写次数，强制通过发布
        print("\n⚠️  已达最大重写次数，强制发布")
        return "publish"
    else:
        # 有修改意见 → 返回重写
        return "writer"

# ===================== 构建图 =====================
workflow = StateGraph(PublishState)
workflow.add_node("writer", writer_node)
workflow.add_node("publish", publish_node)
workflow.set_entry_point("writer")

workflow.add_conditional_edges(
    "writer",
    review_gate,
    {
        "wait": END,
        "publish": "publish",
        "writer": "writer"
    }
)
workflow.add_edge("publish", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ===================== 交互式运行 =====================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "publish_task_001"}}
    topic = "AI Agent 在企业运营中的应用价值"
    
    print(f"📌 任务启动：撰写并发布关于「{topic}」的文章\n")
    
    # 第一步：生成初稿
    result = app.invoke(
        {"topic": topic, "status": "writing", "rewrite_count": 0},
        config=config
    )
    
    # 循环：审核 → 决策 → 重写/发布
    while True:
        print("\n" + "="*50)
        print("⏸️  等待人工审核")
        print("="*50)
        print("📄 当前稿件：")
        print(result["draft"])
        print(f"\n当前第 {result.get('rewrite_count', 1)} 版")
        
        # 【关键】真正的人工输入：程序阻塞，等待控制台输入
        print("\n👤 请输入审核意见：")
        print("  输入 通过  → 发布文章")
        print("  输入 修改意见 → 打回重写")
        human_opinion = input("  你的输入：").strip()
        
        if not human_opinion:
            print("❌ 输入不能为空，请重新输入")
            continue
        
        # 更新状态，写入人工意见
        app.update_state(config, {"review_opinion": human_opinion})
        
        # 恢复执行
        print("\n▶️  提交意见，继续执行...")
        result = app.invoke(None, config=config)
        
        # 清空审核意见，为下一轮审核做准备
        app.update_state(config, {"review_opinion": ""})
        
        # 判断是否结束
        if result.get("status") == "published":
            break
    
    print("\n" + "="*50)
    print("✅ 任务完成")
    print("="*50)
    print(f"最终状态：{result['status']}")
    print(f"最终稿件：\n{result['draft']}")
