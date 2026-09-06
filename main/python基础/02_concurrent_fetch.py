"""第 1 周练习②：并发抓取 10 个 URL（带超时与重试）——P1 第 1 周核心练习

背景：这就是"Agent 并发调用多个工具"的最小原型。
     比如用户问一个问题，Agent 需要同时查搜索、查数据库、查天气——
     同步写法是等一个再等下一个（耗时相加），异步写法是同时发出去（耗时≈最慢那个）。

验收标准（写完自查）：
1. 总耗时 ≈ 最慢单个请求的耗时，而不是 10 个请求耗时相加
2. 任何一个 URL 失败/超时，不影响其他 URL 返回结果
3. 重试后仍失败的 URL，能在结果里看到实际尝试次数（attempts）

运行：python 02_concurrent_fetch.py
实验：把 TIMEOUT_S 改成 2.0，最后一条 URL（故意延迟 3 秒响应）会触发超时重试，观察输出变化
"""
import asyncio
import time

import httpx
from pydantic import BaseModel

URLS = [
    "https://www.baidu.com",
    "https://www.qq.com",
    "https://www.bilibili.com",
    "https://www.csdn.net",
    "https://gitee.com",
    "https://www.163.com",
    "https://www.sina.com.cn",
    "https://juejin.cn",
    "https://github.com",
    "https://httpbin.org/delay/3",  # 故意的慢请求：3 秒后才响应，用来练超时
]

TIMEOUT_S = 5.0   # 单次请求超时
MAX_RETRIES = 2   # 失败后最多再重试 2 次（总共最多尝试 3 次）


class FetchResult(BaseModel):
    """单个 URL 的抓取结果。

    这里同时是第 1 周的另一半内容：类型注解 + Pydantic 模型。
    Pydantic 会校验赋值类型——试着把 status 传个字符串，看它报什么错。
    """
    url: str
    status: int = 0          # HTTP 状态码；彻底失败时保持 0
    elapsed_ms: float = 0.0  # 本次抓取总耗时（含重试）
    attempts: int = 0        # 实际尝试了几次
    ok: bool = False         # 是否拿到了 HTTP 响应（403 也算拿到，练的是异步不是反爬）
    error: str = ""          # 最后一次失败的错误信息


async def fetch_one(client: httpx.AsyncClient, url: str) -> FetchResult:
    """抓取单个 URL，带超时与重试。

    设计约定：这个函数永远不抛异常——失败信息写进 FetchResult.error 返回。
    原因：一个 URL 出问题不该拖垮整批任务（爆炸半径控制，Agent 工具同理）。
    """
    start = time.perf_counter()
    result = FetchResult(url=url)

    for attempt in range(MAX_RETRIES + 1):
        result.attempts = attempt + 1
        try:
            resp = await client.get(url)          # client 创建时已带超时
            result.status = resp.status_code
            result.ok = True
            break
        except httpx.RequestError as e:           # 超时、连接失败都属于它
            result.error = f"{type(e).__name__}: {e}"
            await asyncio.sleep(0.5)              # 重试前歇一下（生产用指数退避）

    result.elapsed_ms = (time.perf_counter() - start) * 1000
    return result


async def fetch_all(urls: list[str]) -> list[FetchResult]:
    """并发抓取所有 URL，一个失败不能拖垮其他任务。"""
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        # 复用同一个 client（连接池），10 个请求同时发出去
        tasks = [fetch_one(client, url) for url in urls]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    # fetch_one 约定不抛异常；这层保护是留给"约定之外"的意外的——防御性设计
    results: list[FetchResult] = []
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            results.append(FetchResult(url="?", error=f"意外异常: {type(outcome).__name__}: {outcome}"))
        else:
            results.append(outcome)
    return results


async def main() -> None:
    start = time.perf_counter()
    results = await fetch_all(URLS)
    total = time.perf_counter() - start

    print(f"{'URL':<42}{'状态':>5}{'尝试':>5}{'耗时ms':>9}  结果")
    print("-" * 80)
    for r in results:
        status = "OK" if r.ok else r.error[:36]
        print(f"{r.url:<42}{r.status:>5}{r.attempts:>5}{r.elapsed_ms:>9.0f}  {status}")

    sum_s = sum(r.elapsed_ms for r in results) / 1000
    print("-" * 80)
    print(f"并发总耗时: {total:.2f}s   所有请求耗时之和: {sum_s:.2f}s")

    assert len(results) == len(URLS), "结果数量必须等于 URL 数量——有任务丢了？"
    assert total < sum_s, "总耗时应该明显小于耗时之和，否则并发没生效"


if __name__ == "__main__":
    asyncio.run(main())
