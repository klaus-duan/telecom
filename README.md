# Telecom AI Agent
![全流程图](flowchart.png)

## 基于 LangGraph + Agentic RAG 的上海电信智能客服 Multi-Agent 系统的最小实现。

## 项目简介

本项目是一个面向上海电信业务的智能客服 Multi-Agent，采用 LangGraph 构建工作流，结合 RAG，能够智能理解用户意图、检索业务知识库并生成精准回答。

### 核心功能

- **ReAct Agent**：根据用户问题识别意图，结合问题和对话历史检索对应知识。
- **Reflection Agent**：首先判断ReAct Agent提供的信息是否足够回答用户问题，生成回答并自主检查。
- **工具调用**：支持 Function Calling，可调用知识库检索工具。
- **多轮对话**：基于 Redis 维护对话历史，支持上下文理解。
- **数据持久化**：对话记录可持久化到 PostgreSQL。

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | FastAPI + LangGraph |
| 大模型 | Qwen1.5系列 |
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

## 检索实现
召回 + 粗排 + 精排
### 召回 (百万 ~ 亿级 → 几百 ~ 几万)
HNSW和bm25双路召回
- IVF：k-means聚类后，桶内计算相似度
- ✅HNSW：多层近邻图
- ✅bm25

### 粗排（几百 → top5）
bm25 + 余弦相似度。\
bm25对专有名词精准匹配效果好，不依赖语义理解，简单直接。无法处理同义词且丢失语序信息。\
余弦相似度处理同义句、意图理解效果好，理解深层语义。对专有名词不敏感。\
**RRF**等于二者结合：\
<p>
RRF得分 = 1/(k + rank<sub>BM25</sub>) + 1/(k + rank<sub>cosine</sub>)
</p>

**embedding model** : 

- [text2vec-bge-large-chinese](https://huggingface.co/shibing624/text2vec-bge-large-chinese)

| 阈值   | 0.8   | 0.7   | 0.5   |
|--------|-------|-------|-------|
| topk   | 3     | 4     | 5     |
| 召回率 | 0.263 | 0.467 | 0.756 |

- ✅[BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)

| 阈值   | 0.9  | 0.8  | 0.7  | 0.7  | 0.7  | 0.5  | 0.5  | 0.5  | 0.5  |
|--------|------|------|------|------|------|------|------|------|------|
| topk   | 1    | 1    | 1    | 2    | 3    | 1    | 2    | 3    | 5    |
| 召回率 | 0.160|0.381 |0.587 |0.613 |0.621 |0.811 |0.878 |0.908 |0.929 |

**最终选择BAAI/bge-large-zh，top_p=0.5，top_k=5**

### 精排（top5 → top1）
**reranker**：✅[BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)

## 记忆实现

当前方案：只保留8轮的窗口 + 单轮最长300字符限制。

### 预期实现（滚动摘要 + 引用）

n轮对话后，提取摘要和引用；prompt = 摘要 + 近两轮对话 + Query

**示例：**

```plaintext
[1] 用户: 我想查话费
    客服: 您可以发送短信"YE"到10001，或登录电信APP查询
    
[2] 用户: APP怎么下载
    客服: 应用商店搜索"中国电信"，或点击链接 https://189.cn/app
    
[3] 用户: 我套餐多少钱一个月
    客服: 您当前是5G畅享129元套餐，含30G流量和500分钟通话
    
[4] 用户: 有没有便宜点的
    客服: 有99元套餐，含20G流量；或79元套餐，含10G流量
    
[5] 用户: 99元的怎么办理
    客服: 发送短信"BL99"到10001，或APP-套餐变更-选择99元套餐
    
[6] 用户: 办理后要下个月生效吗
    客服: 是的，套餐变更次月生效，本月仍按原套餐计费
    
[7] 用户: 那我现在流量快用完了怎么办
    客服: 可以购买流量包，10元1G，发送"LLB10"到10001
    
[8] 用户: 流量包立即生效吗
    客服: 是的，流量包立即生效，当月有效
    
[9] 用户: 好的我先买流量包
    客服: 已为您记录，发送"LLB10"到10001即可购买
    
[10] 用户: 谢谢
     客服: 不客气，还有其他问题吗
```

redis key：

```
dev:chat:{conversation_id}:messages      → 原始消息列表（List）
dev:chat:{conversation_id}:summary       → 8段式摘要（Hash）
dev:chat:{conversation_id}:refs          → 引用库（Hash）
dev:chat:{conversation_id}:meta          → 元信息（Hash）
```

引用：

```
refs: {
  "套餐:5G畅享129元": {
    "价格": "129元/月",
    "流量": "30GB",
    "通话": "500分钟",
    "来源": "对话第3轮"
  },
  "套餐:99元": {
    "价格": "99元/月",
    "流量": "20GB",
    "来源": "对话第4轮"
  },
  "套餐:79元": {
    "价格": "79元/月",
    "流量": "10GB",
    "来源": "对话第4轮"
  },
  "指令:BL99": {
    "功能": "办理99元套餐",
    "方式": "发送短信到10001 或 APP-套餐变更",
    "来源": "对话第5轮"
  }, ...
}
```

摘要：

```
summary: {
  "用户目标": "查询话费、下载APP、了解套餐价格、寻找便宜方案",
  "当前状态": "已了解[[套餐:99元]]和[[套餐:79元]]选项，准备办理",
  "关键决策": ["考虑从[[套餐:5G畅享129元]]降至[[套餐:99元]]"],
  "技术细节": "短信指令: YE查话费, BL99办套餐; APP下载: https://189.cn/app",
  "错误/问题": "无",
  "文件引用": ["[[套餐:5G畅享129元]]", "[[套餐:99元]]", "[[套餐:79元]]", "[[指令:BL99]]"],
  "待办事项": ["考虑发送[[指令:BL99]]办理套餐变更"],
  "用户偏好": "关注价格，希望降低月租"
}
```

redis最终状态：

```
messages: [9, 10]  （只保留最近2轮原文）

summary: {
  "用户目标": "查询话费、了解套餐、解决流量不足、完成购买决策",
  "当前状态": "已决定购买[[流量包:10元1G]]，待办理[[套餐:99元]]",
  "关键决策": [
    "从[[套餐:5G畅享129元]]降至[[套餐:99元]]（月省30元）",
    "本月先买[[流量包:10元1G]]应急（立即生效）",
    "次月[[套餐:99元]]生效"
  ],
  "技术细节": "指令: YE/BL99/LLB10; APP: https://189.cn/app",
  "错误/问题": "无",
  "文件引用": [
    "[[套餐:5G畅享129元]]",
    "[[套餐:99元]]", 
    "[[套餐:79元]]",
    "[[指令:BL99]]",
    "[[规则:套餐变更生效]]",
    "[[流量包:10元1G]]"
  ],
  "待办事项": [
    "[x] 了解套餐价格（完成）",
    "[x] 决定购买流量包（完成）",
    "[ ] 发送LLB10购买流量包",
    "[ ] 发送BL99变更套餐（次月生效）"
  ],
  "用户偏好": "价格敏感，关注生效时间，需要即时解决方案"
}

refs: {
  "套餐:5G畅享129元": {...},
  "套餐:99元": {...},
  "套餐:79元": {...},
  "指令:BL99": {...},
  "规则:套餐变更生效": {...},
  "流量包:10元1G": {...}
}
```

第11轮提问：“流量包多少钱来着？”
组装prompt：

```
【系统提示】你是上海电信员工晶晶...

【8段摘要】
用户目标: 查询话费、了解套餐、解决流量不足、完成购买决策
当前状态: 已决定购买[[流量包:10元1G]]，待办理[[套餐:99元]]
关键决策: 
  - 从[[套餐:5G畅享129元]]降至[[套餐:99元]]（月省30元）
  - 本月先买[[流量包:10元1G]]应急（立即生效）
技术细节: 指令: YE/BL99/LLB10; APP: https://189.cn/app
待办事项: [ ] 发送LLB10购买流量包, [ ] 发送BL99变更套餐
用户偏好: 价格敏感，关注生效时间

【引用详情】（本次需要展开）
[[流量包:10元1G]]: 价格10元，流量1GB，立即生效，当月有效，指令LLB10

【最近2轮原文】
[9] 用户: 好的我先买流量包
    客服: 已为您记录，发送LLB10到10001即可购买
[10] 用户: 谢谢
     客服: 不客气，还有其他问题吗

【新消息】
用户: 流量包多少钱来着？
```
