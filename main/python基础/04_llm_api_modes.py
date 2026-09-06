"""第 1 周练习④：大模型 API 三种调用（流式 + JSON Mode）

普通调用你已经会了（agent基础/01 那批），这里补上第 1 周列表里缺的两个：
1. 流式返回：逐 token 吐出——下周 FastAPI + SSE 打字机效果就靠它
2. JSON Mode：让模型输出合法 JSON——第 5 周工具参数结构化解析的前提

学习目标——跑通后能不看资料回答：
- 流式和普通在 API 层面差什么？为什么流式"先返回再拼"，而不是一次性等完？
- JSON Mode 是"保证输出是合法 JSON"，不是"保证内容正确"——怎么理解？

运行：python 04_llm_api_modes.py
前置：.env 里有 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME（agent基础/ 同款）
"""
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME")

SYSTEM_PROMPT = "你是一个乐于助人的助手，回答要简洁。"


# ===================== Part A：流式返回 =====================
def part_a_stream() -> None:
    print("===== 流式调用（打字机效果预览） =====")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "用一句话介绍什么是 Agent。"},
        ],
        stream=True,  # ★ 关键差异：开流式
    )

    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:  # 有些 chunk 没有内容（比如只带结束信号），跳过
            full_text += delta
            print(delta, end="", flush=True)  # flush 强制立刻输出，否则看不到打字效果
    print("\n---- 流式收到的完整文本 ----")
    print(full_text)


# ===================== Part B：JSON Mode =====================
def part_b_json_mode() -> None:
    print("\n===== JSON Mode 调用 =====")
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},  # ★ 关键差异：要求 JSON
        messages=[
            {"role": "system", "content": "你是一个任务规划助手。"
                                          "只输出 JSON 对象，不要输出任何其他文字。"},
            {"role": "user", "content": "把'调研 Agent 开发框架'这个任务拆成 3 个子任务，"
                                        "返回 JSON，格式为：{\"tasks\": [\"...\", \"...\", \"...\"]}"},
        ],
    )
    raw = resp.choices[0].message.content
    print("模型返回原文（先看它长什么样，再说解析）:")
    print(raw)

    # TODO 1：把 raw 用 json.loads 解析成 dict，打印出 tasks 列表
    # 思考：
    #   a) JSON Mode 保证"能解析成功"，那如果内容结构不对呢？（试试把要求里的 tasks 换成其他字段名）
    #   b) json.loads 失败会抛什么？——这行将来通常换成本地 Pydantic 模型做校验
    ...


# ===================== Part C：流式 + JSON（可选，超前一点） =====================
def part_c_stream_json() -> None:
    """流式逐字收到一段 JSON，最后再整体解析——第 2 周会用到。

    这个 Part 不要求完成 TODO，先跑通看现象，理解"流式是过程、JSON 是结果"。
    """
    print("\n===== 流式 + JSON Mode（预览） =====")
    stream = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        stream=True,
        messages=[
            {"role": "system", "content": "只输出 JSON 对象。"},
            {"role": "user", "content": "返回一个 JSON：{\"name\": \"你的名字\", \"hobby\": \"你喜欢什么\"}"},
        ],
    )
    full = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full += delta
    print("收到的原始 JSON 字符串:", full)
    try:
        obj = json.loads(full)
        print("解析成功:", obj)
    except json.JSONDecodeError as e:
        print("解析失败（JSON Mode 下很少见）:", e)


def main() -> None:
    part_a_stream()
    part_b_json_mode()
    part_c_stream_json()
    print("\n全部跑通")


if __name__ == "__main__":
    main()
