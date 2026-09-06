"""命令行入口。

用法：
    python main.py "计算 (128 * 4) / 2 - 17"
    python main.py          # 不带参数则进入交互输入
"""
import sys

from agent import run


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("你的问题: ").strip()
    if not question:
        print("问题不能为空")
        return

    print(f"\n问题: {question}")
    answer = run(question)
    print(f"\n========== 最终结果 ==========\n{answer}")


if __name__ == "__main__":
    main()
