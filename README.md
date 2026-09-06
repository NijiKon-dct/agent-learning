# Agent 开发学习仓库

从零到企业级 Agent 开发的系统学习实践仓库。依据《Agent 开发入门学习路线（2026 · 求职导向）》执行，按周推进、以项目产出物验收。

> 求职导向：目标是让"写过的每一行代码"都能在简历和面试里讲清楚决策与取舍，而不是收藏一堆教程。

---

## 目录结构

```
agent-learning/
├── MD/                        # 学习纲领与复盘
│   ├── Agent开发入门学习路线.md    # 学习路线原文（求职导向，为什么这么学）
│   ├── 学习计划.md                # 16 周执行计划（每周任务清单 + 过关标准）
│   └── read.md                   # Agent 前 6 课核心复盘
│
├── main/
│   ├── python基础/               # P1 工程底座练习
│   │   ├── 01_async_basics.py        # async/await：协程/并发/超时取消
│   │   ├── 02_concurrent_fetch.py    # 并发抓取 10 URL（重试+超时+Pydantic）
│   │   ├── 03_pydantic_basics.py     # Pydantic 校验：嵌套/约束/解析 LLM JSON
│   │   └── 04_llm_api_modes.py       # 大模型 API：流式 + JSON Mode
│   │
│   └── agent基础/               # Agent 核心机制练习（从手写原理到框架）
│       ├── 01_simple_agent.py        # 手写 ReAct 循环（正则解析版）
│       ├── 02_search_agent.py        # 搜索工具 + 滑动窗口记忆
│       ├── 03_function_calling_agent.py  # 原生 Function Calling
│       ├── 04_planning_agent.py      # Plan-and-Execute 规划型 Agent
│       ├── 05_langgraph_agent.py     # LangGraph：State/Node/Edge
│       ├── 06_multi_agent.py         # 多 Agent 协作（三角色写作团队）
│       ├── 07_checkpointer_agent.py  # Checkpoint 持久化 + 会话隔离
│       └── 08_human_in_the_loop.py   # 人工审核介入（暂停/恢复）
│
└── week4-react-agent/         # 手写 ReAct Agent（项目①起点，不依赖任何框架）
    ├── llm.py                 # OpenAI 兼容客户端封装
    ├── prompt.py              # ReAct 输出协议
    ├── tools.py               # 工具注册表
    ├── agent.py               # 核心循环：解析/终止条件/防死循环/错误回传
    └── main.py                # 命令行入口
```

## 学习进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 第 1 周 | Python 异步 + Pydantic + 模型 API 三种调用 | ✅ 完成 |
| 第 2 周 | FastAPI + SSE 流式输出 | 🔄 进行中 |
| P2 | 手写 Agent / Function Calling / Context Engineering | ✅ 练习完成（agent基础 01-03） |
| P3 | LangGraph / Checkpoint / 多 Agent / HIL | ✅ 练习完成（agent基础 05-08，待项目化） |

> 详细任务清单与过关标准见 [MD/学习计划.md](MD/学习计划.md)。按计划规则：**阶段产出物未达标不进入下一阶段**，练习完成 ≠ 项目完成。

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

# 2. 配置密钥（复制 .env.example 为 .env 并填入）
#    当前用到的：OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME / TAVILY_API_KEY

# 3. 运行练习（示例：手写 ReAct Agent）
$env:LLM_API_KEY = "sk-xxx"   # 或用 .env + dotenv
.\venv\Scripts\python week4-react-agent\main.py "计算 1234 * 5678 + 9876"
```

**注意**：`.env` 与 `venv/` 已在 `.gitignore` 中排除，密钥不会进入版本库。

## 学习原则（来自学习路线）

- 先用原始 API 手写一遍，再去学框架——能讲清"框架替我解决了什么"，是区分用过与懂的分水岭
- 每个阶段有硬性产出物，产出物 = 学会的证明
- 踩坑记录与复盘（read.md / 博客）与写代码同等重要
