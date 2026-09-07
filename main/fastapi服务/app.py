r"""第 2 周 · FastAPI 对话服务 —— 支持 SSE 流式输出 + 鉴权 + 限流

学习计划第 2 周任务：
1. FastAPI 基础：路由 / Pydantic 请求模型 / 响应模型 / 异常处理 ✅
2. POST /chat：接收消息，返回模型回复 ✅
3. POST /chat/stream：SSE 流式接口（打字机效果）✅
4. 加鉴权（X-API-Key Header）和简单限流（内存滑动窗口）✅ —— 本文件新增

鉴权/限流后的调用方式（所有 /chat* 接口都要带 key）：
    curl -X POST http://127.0.0.1:8000/chat/stream \
      -H "Content-Type: application/json" \
      -H "X-API-Key: <你的 SERVICE_API_KEY>" \
      -d "{\"message\": \"你好\"}"

运行：
    python app.py
"""
import asyncio
import json
import os
import time

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

app = FastAPI(title="对话服务", version="0.2.0")


# ===================== 请求 / 响应模型（Pydantic 帮你校验） =====================
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="用户消息")
    # history 可选：多轮对话上下文（[{role, content}, ...]），留到第 3 周接存储
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    model: str


# ===================== 鉴权 + 限流（FastAPI Dependency） =====================
# 依赖是 FastAPI 的复用机制：一段校验逻辑写一次，
# 通过 Depends(verify_request) 挂到任意路由，路由函数签名不用改。
# 请求流程：Header 注入 → 鉴权(不通过=401) → 限流(超了=429) → 路由函数

REQ_WINDOW_SECONDS = 60    # 窗口长度：1 分钟
MAX_REQ_PER_WINDOW = 10    # 窗口内最多请求数（演示用小值，生产按需调大）

# 进程内滑动窗口计数：{api_key: [最近请求的时间戳, ...]}
# ⚠ 生产局限：多进程/多实例不共享、进程重启即清零——
#   这正是第 3 周把状态搬到 Redis 的原因，先记下这个"为什么"
_request_times: dict[str, list[float]] = {}


def verify_request(x_api_key: str = Header(default="")) -> None:
    """鉴权 + 限流二合一依赖。

    x_api_key 来自请求头 X-API-Key；Header(default="") 表示缺头时给空串，
    走到下面自然 401——让"缺 key"和"错 key"统一处理，不给额外信息。
    """
    # ---- 第一段：鉴权（你是谁） ----
    expected = os.getenv("SERVICE_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="服务端未配置 SERVICE_API_KEY（请加入 .env）")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="无效的 API Key")

    # ---- 第二段：限流（你能用多少） ----
    now = time.time()
    # 只保留窗口内的历史时间戳——这是"滑动窗口"和"固定窗口"的区别：
    # 固定窗口在边界会瞬间允许 2 倍流量，滑动窗口不会
    recent = [t for t in _request_times.get(x_api_key, []) if now - t < REQ_WINDOW_SECONDS]
    if len(recent) >= MAX_REQ_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁：{REQ_WINDOW_SECONDS} 秒内最多 {MAX_REQ_PER_WINDOW} 次",
        )
    recent.append(now)
    _request_times[x_api_key] = recent


# ===================== 路由 =====================
@app.get("/health")
def health() -> dict:
    """健康检查：服务是否活着。"""
    return {"status": "ok"}


# ===================== 非流式接口（保留作对比） =====================
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_request)])
async def chat(req: ChatRequest) -> ChatResponse:
    """普通对话接口：一次性等模型返回完整回复（已改 async def）。"""
    messages = [{"role": "system", "content": "你是一个乐于助人的助手，回答简洁。"}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    try:
        # 注意：同步 client 在 async 函数里会阻塞事件循环！
        # 这里用同步是教学简化；第 3 周改 AsyncOpenAI 或用 run_in_executor
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        reply = resp.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"模型调用失败: {type(e).__name__}: {e}")

    return ChatResponse(reply=reply, model=MODEL)


# ===================== SSE 流式接口（本阶段核心） =====================
def build_messages(req: ChatRequest) -> list[dict]:
    """组装消息：系统提示词 + 可选历史 + 当前问题。两个接口共用。"""
    messages = [{"role": "system", "content": "你是一个乐于助人的助手，回答简洁。"}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})
    return messages


async def sse_event_generator(req: ChatRequest):
    """SSE 事件生成器：逐步产出 'data: {...}\n\n' 格式的文本块。

    这是 SSE 的核心——它是个 async 生成器：
    - yield 一次，StreamingResponse 就往 HTTP 连接写一次
    - 中间用 await 让出控制权，不阻塞其他请求
    """
    messages = build_messages(req)

    try:
        # stream=True：把"一次等完"变成"边生成边收"
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, stream=True, temperature=0.7
        )

        # 逐 token 转发给客户端
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                # SSE 协议：每条事件 = "data: <JSON>\n\n"（两个换行分隔）
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                # 教学用：模拟真实网络逐字推送的节奏（生产环境去掉）
                await asyncio.sleep(0.02)

        # 流结束标记，前端以此判断"说完了"
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        # 流中途出错：也要以 SSE 格式把错误推给前端，而不是直接断连
        yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"


@app.post("/chat/stream", dependencies=[Depends(verify_request)])
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式对话接口：打字机效果。

    和普通接口的三个关键差异：
    1. 返回 StreamingResponse，不是 JSON
    2. media_type 必须是 text/event-stream，浏览器/客户端才按流解析
    3. 真正内容由 sse_event_generator 异步逐步产出
    """
    return StreamingResponse(
        sse_event_generator(req),
        media_type="text/event-stream",
        headers={
            # 防代理/浏览器缓冲：关掉才能实时看到字（开发期）
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

