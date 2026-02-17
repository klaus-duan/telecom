# Telecom AI Agent
![全流程图](flowchart.png)

基于 LangGraph + RAG 的电信智能客服 Agent 系统。

## 项目简介

本项目是一个面向电信业务的智能客服 Agent，采用 LangGraph 构建工作流，结合 RAG（检索增强生成）技术，能够智能理解用户意图、检索业务知识库并生成精准回答。

### 核心特性

- **智能路由**：根据用户问题自动判断是否需要检索知识库（RAG/NO_RAG/TOOL/CLARIFY）
- **工具调用**：支持 Function Calling，可调用知识库检索工具
- **多轮对话**：基于 Redis 维护对话历史，支持上下文理解
- **数据持久化**：对话记录可持久化到 PostgreSQL
- **幂等保护**：防止重复请求，确保接口稳定性

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | FastAPI + LangGraph |
| 大模型 | Qwen (通义千问) |
| 向量数据库 | Milvus |
| 缓存/会话 | Redis |
| 数据持久化 | PostgreSQL |
| 向量嵌入 | BAAI/bge-large-zh |

## 项目结构

```
telecom/
├── app/
│   ├── api/
│   │   └── routes.py          # API 路由定义
│   ├── core/
│   │   ├── config.py          # 配置管理
│   │   ├── schemas.py         # 数据模型
│   │   └── utils.py           # 工具函数
│   ├── graphs/
│   │   └── rag_graph.py       # LangGraph 工作流定义
│   ├── integrations/
│   │   ├── milvus_retriever.py   # Milvus 检索器
│   │   ├── postgres_store.py     # PostgreSQL 存储
│   │   ├── qwen_openai.py        # Qwen 客户端
│   │   └── redis_memory.py       # Redis 会话管理
│   └── main.py                # 应用入口
├── requirements.txt           # 依赖列表
└── .env                       # 环境变量配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件，配置以下变量：

```env
# 应用环境
APP_ENV=dev

# Redis 配置
REDIS_URL=redis://localhost:6379/0
REDIS_PREFIX=dev
SESSION_TTL_SECONDS=7200

# Milvus 配置
MILVUS_URI=https://your-milvus-uri
MILVUS_TOKEN=your-milvus-token
MILVUS_COLLECTION=qa_collection
MILVUS_TOP_K=5

# Qwen API 配置
QWEN_API_KEY=your-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen-plus

# PostgreSQL 配置（可选）
POSTGRES_DSN=postgresql://user:password@localhost:5432/dbname

# 路由模式：heuristic 或 react
ROUTER_MODE=heuristic
```

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 接口

### 对话接口

**POST** `/chat`

请求体：
```json
{
  "conversation_id": "可选，不传则自动生成",
  "request_id": "请求唯一标识",
  "user_id": "用户ID",
  "message": "用户问题"
}
```

响应：
```
conversation_id: xxx
request_id: xxx
answer: 回答内容
```

### 结束对话

**POST** `/end`

将对话历史持久化到 PostgreSQL 并清理 Redis 缓存。

请求体：
```json
{
  "conversation_id": "对话ID"
}
```

## Agent 工作流

```
用户输入
    ↓
[路由节点] → 判断意图 (RAG/NO_RAG/TOOL/CLARIFY)
    ↓
    ├─→ RAG → [检索知识库] → [生成回答]
    ├─→ NO_RAG → [直接回答]
    ├─→ TOOL → [工具调用]
    └─→ CLARIFY → [追问澄清]
    ↓
返回答案
```

## 路由策略

- **heuristic**：基于规则的路由（关键词匹配）
- **react**：基于 LLM 的智能路由

## 许可证

MIT

## 数据库形式
### milvus
**表结构**
milvus_cli > show collection -c qa_collection 

| 属性 | 值 |
|---|---|
| Name | qa_collection |
| Description | |
| Entities | 14256 |
| Is Empty | False |
| Primary Field | id |
| Schema | Description: |
| | |
| | Fields(* is the primary field): |
| | - *id 5 |
| | - question 21 max_length: 2048 |
| | - knowledge 21 max_length: 65535 |
| | - question_emb 101 dim: 1024 |
| Partitions | - _default |
| Indexes | - question_emb |

**数据示例**
| id | question | knowledge | question_emb |
|---|---|---|---|
| 5 | 我想下载上个月的发票怎么操作 | 开发票、查询发票、合并发票、下载发票方法 -电子发票（包括实缴发票、充值发票、月结发票）分为个人和企业两种，点击链接即可线上办理，无需线下办理。链接有所不同： 个人电子发票：点击进入该小程序链接：#小程序://上海电信/lTOQOyYnCF4rLqf 即可线上查询发票、合并开票、开具和下载电子发票（包括实缴发票、充值发票、月结发票）。 企业电子发票：进入该https链接：https://1go.sh.189.cn/billing/yw/zs/xz_pc/yun_m_2022.htm?ptk=123&fs=1&type=xcx 即... | [-0.0015246168,-0.045919843,-0.0075209057,-0.0065605766,0.011926289,0.010762139,-0.00003397221,0.0009114785,0.005358622,-0.0046699075,0.0008535707,0.052...] |

### redis
**表结构**
```shell
redis-cli> KEYS "dev:chat:*"
1) "dev:chat:0d416a9d-c425-455c-9cc8-375ebb64ca02:resp:r1"    #对话消息列表
2) "dev:chat:0d416a9d-c425-455c-9cc8-375ebb64ca02:messages"   #请求ID集合
3) "dev:chat:0d416a9d-c425-455c-9cc8-375ebb64ca02:req_ids"    #第1条请求的响应缓存
4) "dev:chat:0d416a9d-c425-455c-9cc8-375ebb64ca02:resp:r2"    #第2条请求的响应缓存
```

### PostgreSQL
**表结构**

| column_name | data_type | is_nullable | column_default |
|-------------|-----------|-------------|----------------|
| id | bigint | NO | nextval('chat_history_id_seq'::regclass) |
| conversation_id | uuid | NO | *NULL* |
| request_id | text | NO | *NULL* |
| message | text | NO | *NULL* |
| answer | text | NO | *NULL* |
| time | timestamp with time zone | NO | now() |
