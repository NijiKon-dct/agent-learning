"""ReAct Agent 核心循环：纯 while 思路 + 字符串解析，不依赖任何框架。

第 4 周的学习目标：
1. 读懂这个文件的每一行
2. 合上它，从空白文件自己复现一遍——这是检验"学会"的唯一标准
3. 完成底部的练习任务

循环流程：
    用户问题 → 模型输出(Thought/Action) → 解析 → 执行工具 → Observation 回传
         ↑                                              |
         └────────────────── 循环 ←─────────────────────┘
"""
import re

from llm import chat
from prompt import build_system_prompt
from tools import TOOLS

MAX_STEPS = 8                 # 防死循环·第一道防线：总步数上限
MAX_CONSECUTIVE_REPEATS = 3   # 防死循环·第二道防线：同一动作连续重复 N 次即判定卡死


def extract_final_answer(text: str) -> str | None:
    """提取 Final Answer（允许换行的多行回答），没有则返回 None。"""
    m = re.search(r"Final Answer:\s*([\s\S]+)", text)
    return m.group(1).strip() if m else None


def extract_action(text: str) -> tuple[str, str] | None:
    """提取 (action, action_input)。Action 缺失返回 None；Input 缺失按空串处理。"""
    action_m = re.search(r"Action:\s*(\S+)", text)
    if not action_m:
        return None
    input_m = re.search(r"Action Input:\s*(.*)", text)
    return action_m.group(1).strip(), (input_m.group(1).strip() if input_m else "")


def run(question: str, verbose: bool = True) -> str:
    """运行一次完整的 ReAct 循环，返回最终答案（或终止原因）。"""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": question},
    ]
    last_action: tuple[str, str] | None = None
    repeat_count = 0

    for step in range(1, MAX_STEPS + 1):
        response = chat(messages)
        if verbose:
            print(f"\n===== 第 {step}/{MAX_STEPS} 步 · 模型输出 =====\n{response}")

        # 1. 终止条件一：模型给出最终答案
        final = extract_final_answer(response)
        if final:
            return final

        # 2. 解析 Action；解析失败就把"写给模型看的错误信息"回传，让它重试
        parsed = extract_action(response)
        if parsed is None:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": (
                "Observation: 格式错误——你的输出里没有可识别的 Action 或 Final Answer。"
                "请严格使用 'Thought/Action/Action Input' 或 'Thought/Final Answer' 格式。"
            )})
            continue

        action, action_input = parsed

        # 3. 终止条件二：同一动作连续重复，判定死循环
        if (action, action_input) == last_action:
            repeat_count += 1
        else:
            repeat_count = 1
        last_action = (action, action_input)
        if repeat_count >= MAX_CONSECUTIVE_REPEATS:
            return f"[Agent 强制终止] 动作 {action}({action_input}) 连续重复 {repeat_count} 次，疑似死循环。"

        # 4. 执行工具；工具不存在 / 执行抛错，都转成模型能读懂的 Observation
        if action not in TOOLS:
            observation = (
                f"错误：工具 '{action}' 不存在。可用工具：{', '.join(TOOLS)}。"
                f"请从这些工具里选择，或直接给出 Final Answer。"
            )
        else:
            try:
                observation = TOOLS[action]["func"](action_input)
            except Exception as e:
                observation = (
                    f"工具执行失败：{type(e).__name__}: {e}。"
                    f"请检查 Action Input 是否符合该工具的要求，修正后重试。"
                )

        if verbose:
            print(f"----- Observation -----\n{observation}")

        # 5. 把本轮输出和 Observation 拼回上下文，进入下一轮
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # 6. 终止条件三：步数耗尽
    return f"[Agent 强制终止] 已达到最大步数 {MAX_STEPS}，仍未得到最终答案。"


# ---------------------------------------------------------------------------
# 练习任务（本周做完，下周用 Function Calling 重构时对照）
# 1. 加第 3 个工具：如 read_file(path)，读本地文本文件并返回前 500 字符
# 2. 破坏性实验 A：把 prompt.py 里的格式说明改坏（比如删掉 Action Input 的说明），
#    观察"格式错误回传"如何让模型自我纠正，记录进踩坑笔记
# 3. 破坏性实验 B：把 MAX_STEPS 改成 2、把工具描述写得含糊，
#    观察循环分别在什么情况下提前终止 / 走偏
# 4. 进阶：当前只能检测"连续重复"，试着检测 A→B→A→B 交替循环
# 5. 思考：messages 列表会一直变长，什么时候会出问题？（第 6 周解决：上下文管理）
# ---------------------------------------------------------------------------
