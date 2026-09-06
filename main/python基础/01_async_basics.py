"""第 1 周练习①：异步基础三连（协程 / 并发 / 超时取消）

学习目标——跑通并填完 TODO 后，你应该能不看资料回答：
1. async def 定义的东西是什么？为什么必须 asyncio.run() 它才会执行？
2. await 到底在等什么？"挂起"和"阻塞"差在哪？
3. gather 并发跑 3 个 1 秒任务，总耗时是多少？为什么？

运行：python 01_async_basics.py
"""
import asyncio
import time


# ===================== Part A：第一个协程 =====================
async def say_after(delay: float, message: str) -> str:
    """协程：async def 定义；函数体里遇到 await 会"让出"控制权。

    asyncio.sleep 就是"网络等待"的替身——
    真实项目里这一行可能是：等大模型返回、等数据库查询、等 HTTP 响应。
    """
    await asyncio.sleep(delay)
    print(message)
    return message


async def part_a() -> None:
    # TODO 1：依次 await 两个 say_after：
    #   先 1 秒后打印"第一条"，再 1 秒后打印"第二条"
    # 提示：result = await say_after(1, "第一条")
    await say_after(1, "第一条")
    await say_after(1, "第二条")


# ===================== Part B：串行 vs 并发 =====================
async def part_b_serial() -> float:
    """串行执行 3 个 1 秒任务，返回总耗时"""
    start = time.perf_counter()
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    await asyncio.sleep(1)
    return time.perf_counter() - start


async def part_b_concurrent() -> float:
    """并发执行 3 个 1 秒任务，返回总耗时"""
    start = time.perf_counter()
    # gather：3 个任务同时开跑，总耗时 ≈ 最慢的那个（1 秒）
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1),
    )
    return time.perf_counter() - start


# ===================== Part C：超时与取消 =====================
async def slow_llm_call() -> str:
    """模拟一个卡死的模型调用：15 秒后才返回"""
    await asyncio.sleep(15)
    return "终于好了（但你不会看到这行）"


async def part_c() -> None:
    # wait_for：2 秒内没完成就取消任务，抛 TimeoutError
    try:
        await asyncio.wait_for(slow_llm_call(), timeout=2)
    except asyncio.TimeoutError:
        print("调用超时，已取消")


async def main() -> None:
    print("===== Part A =====")
    await part_a()

    print("\n===== Part B =====")
    t_serial = await part_b_serial()
    t_concurrent = await part_b_concurrent()
    print(f"串行耗时: {t_serial:.2f}s（应约 3 秒）")
    print(f"并发耗时: {t_concurrent:.2f}s（应约 1 秒）")
    assert t_serial > 2.5, "串行应该接近 3 秒"
    assert t_concurrent < 1.5, "并发应该接近 1 秒——你的 gather 写对了吗？"

    print("\n===== Part C =====")
    start = time.perf_counter()
    await part_c()
    print(f"超时耗时: {time.perf_counter() - start:.2f}s（应约 2 秒）")


if __name__ == "__main__":
    asyncio.run(main())
