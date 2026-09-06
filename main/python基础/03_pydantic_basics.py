"""第 1 周练习③：类型注解 + Pydantic

学习目标——做完后能不看资料回答：
1. 类型注解本身不阻止传错类型，Pydantic 靠什么在运行时拦住？
2. ValidationError 的错误信息长什么样？为什么"错误可读"对 Agent 特别重要？
3. LLM 返回的 JSON 字符串怎么一步变成校验过的模型对象？

运行：python 03_pydantic_basics.py
"""
from pydantic import BaseModel, Field, ValidationError


# ===================== Part A：第一个模型 + 亲眼看一次校验失败 =====================
class ChatMessage(BaseModel):
    role: str      # "system" / "user" / "assistant"
    content: str


def part_a() -> None:
    ok = ChatMessage(role="user", content="你好")
    print(f"合法消息: {ok.role}: {ok.content}")

    # TODO 1：故意传非法数据：content 给个数字 123
    #   用 try/except ValidationError 捕获，打印 e.errors()（一个列表，看第一条就行）
    #   预期：Pydantic 报 "Input should be a valid string"
    #   注意体会：不是等到某处莫名其妙崩，而是在入口就被拦下并告诉你错在哪
    try:
        ChatMessage(role="user", content=123)
    except ValidationError as e:
        print(e.errors()[0]["msg"])


# ===================== Part B：默认值 + Field 约束 =====================
class AgentConfig(BaseModel):
    model_name: str = "deepseek-chat"
    temperature: float = Field(0.0, ge=0.0, le=2.0)  # ge=大于等于，le=小于等于
    max_steps: int = Field(8, ge=1, le=50)


def part_b() -> None:
    # TODO 2：验证三件事，各打印一行：
    #   1) AgentConfig()  不传参数——temperature 是多少？（默认值）
    #   2) AgentConfig(temperature=0.7)  合法，打印它
    #   3) AgentConfig(temperature=9.9)  捕获 ValidationError 并打印错误
    #      ——这就是"用约束把输入收窄"，第 5 周工具参数设计的核心手段
    ok = AgentConfig()
    print(f"默认值 temperature: {ok.temperature}")
    ok = AgentConfig(temperature=0.7)
    print(f"合法 temperature: {ok.temperature}")
    try:
        AgentConfig(temperature=9.9)
    except ValidationError as e:
        print(e.errors()[0]["msg"])


# ===================== Part C：嵌套模型 =====================
class LLMConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.deepseek.com"


class Agent(BaseModel):
    name: str
    llm: LLMConfig          # 模型里嵌模型，自动递归校验
    tools: list[str] = []   # 列表字段
    retries: int = 2


def part_c() -> None:
    config = {
        "name": "searcher",
        "llm": {"api_key": "sk-xxx"},
        "tools": ["web_search", "calculator"],
    }
    # 1) 合法数据直接创建，不需要 try——try 只包"预期会失败"的那一步
    agent = Agent(**config)
    print(f"嵌套默认值 base_url: {agent.llm.base_url}")   # 预期：https://api.deepseek.com
    print(f"工具数量: {len(agent.tools)}")                  # 预期：2

    # 2) 构造坏数据：llm 里的 api_key 删掉，这步才是预期失败的
    bad = {**config, "llm": {}}
    try:
        Agent(**bad)
    except ValidationError as e:
        print(e.errors()[0]["msg"])  # 预期：Field required——注意 loc 会指向哪一层


# ===================== Part D：最贴近 Agent——解析 LLM 的 JSON 输出 =====================
class ToolCall(BaseModel):
    """模拟大模型 Function Calling 返回的工具调用"""
    name: str
    arguments: dict = {}


def part_d() -> None:
    # 模拟模型返回的 JSON 字符串（真实场景里它来自 message.tool_calls 或 JSON Mode）
    llm_output = '''
    {"name": "web_search", "arguments": {"query": "2024 奥运会金牌数"}}
    '''

    # TODO 4：用 ToolCall.model_validate_json(llm_output) 把它变成模型对象，
    #   打印 tool.name 和 tool.arguments["query"]
    # 思考：如果模型把 "name" 拼成 "nmae"，或漏了这个字段，会发生什么？
    #   ——校验在入口拦截 + 错误信息可读，这就是第 5 周"失败错误回传"的地基
    try:
        tool = ToolCall.model_validate_json(llm_output)
        print(f"合法工具调用: {tool.name}({tool.arguments['query']})")
    except ValidationError as e:
        print(e.errors()[0]["msg"])


def main() -> None:
    print("===== Part A =====")
    part_a()
    print("\n===== Part B =====")
    part_b()
    print("\n===== Part C =====")
    part_c()
    print("\n===== Part D =====")
    part_d()
    print("\n全部通过")


if __name__ == "__main__":
    main()
