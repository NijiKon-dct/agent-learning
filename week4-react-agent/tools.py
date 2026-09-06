"""工具集：每个工具 = 名字 + 写给模型看的描述 + 一个普通 Python 函数。

工具设计三个要点（第 5 周深入）：
1. 名字要说清楚它是干什么的
2. 描述要写清楚什么时候用、输入长什么样
3. 出错时抛异常即可，错误信息会被 agent.py 包装后回传给模型
"""
from datetime import datetime


def calculator(expression: str) -> str:
    """计算四则运算表达式，例如 '2 + 3 * 4'。"""
    # 学习版实现：字符白名单 + 收紧 eval 环境，避免执行任意代码
    allowed = set("0123456789+-*/().% ")
    if not set(expression) <= allowed:
        raise ValueError("表达式包含不允许的字符，只支持数字和 + - * / ( ) %")
    return str(eval(expression, {"__builtins__": {}}, {}))


def get_current_time(action_input: str = "") -> str:
    """获取当前日期和时间。不需要输入参数。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 工具注册表：prompt.py 会把这里的描述渲染进系统提示词
TOOLS: dict[str, dict] = {
    "calculator": {
        "description": "计算一个数学表达式，如 '2 + 3 * 4'。"
                       "当需要精确数值计算时使用，不要自己心算。",
        "func": calculator,
    },
    "get_current_time": {
        "description": "获取当前的日期和时间。无需输入参数，Action Input 留空即可。"
                       "当用户问'现在几点/今天几号'这类问题时使用。",
        "func": get_current_time,
    },
}
