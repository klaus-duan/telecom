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
**表结构示例**
```
`KEY` bot:session:{session_id}
```
```json
`VALUE`{
  "session_id": "sess_20260408162714_7f2c9d4e",
  "metadata_type": "cli:direct",
  "created_at": "2026-04-08T16:27:14.688754+08:00",
  "updated_at": "2026-04-08T16:30:59.763396+08:00",

  "messages": [
    // ======================
    // round: 1  第一轮
    // 用户问 → 模型答
    // ======================
    {
      "round": 1,
      "role": "user",
      "content": "你自带的web搜索工具要不要api或者翻墙"
    },
    {
      "round": 1,
      "role": "assistant",
      "content": "不需要 API 密钥，但可能需要翻墙。\n\n**web_search 工具说明：**..."
    },

    // ======================
    // round: 2  第二轮
    // 用户问 → 模型调用工具 → 工具返回 → 模型最终回答
    // 全部都属于 round:2
    // ======================
    {
      "round": 2,
      "role": "user",
      "content": "测试一下,国内国外网站都测试"
    },
    {
      "round": 2,
      "role": "assistant",
      "content": "好的，我来测试一下 web 搜索工具，分别测试国内和国外网站。",
      "tool_calls": [
        {"id": "call_aa1ea9e4", "name": "web_search"},
        {"id": "call_5282842d", "name": "web_search"}
      ]
    },
    {
      "round": 2,
      "role": "tool",
      "tool_call_id": "call_aa1ea9e4",
      "tool_name": "web_search",
      "content": "Results for: Python 人工智能 2026..."
    },
    {
      "round": 2,
      "role": "tool",
      "tool_call_id": "call_5282842d",
      "tool_name": "web_search",
      "content": "Results for: AI machine learning trends 2026..."
    },
    {
      "round": 2,
      "role": "assistant",
      "content": "✅ 测试成功！搜索功能工作正常。..."
    }
  ],

  "summaries": [
    {
      "summary_id": "sum_1",
      "start_round": 1,
      "end_round": 10,
      "summary_md": "# 1～10轮对话摘要\n\n用户先询问Web搜索工具是否需要API与翻墙；随后要求测试国内外网站搜索能力，模型执行两次web_search并返回结果。过程中参考<|Redis最佳实践|>、<|AI对话上下文存储规范|>。",
      "ref_docs": [
        {
          "doc_name": "Redis最佳实践",
          "doc_md": "# Redis最佳实践\n1. key使用冒号分隔\n2. value避免超大JSON\n3. 必须设置过期时间..."
        },
        {
          "doc_name": "AI对话上下文存储规范",
          "doc_md": "# AI对话上下文存储规范\n1. 每10轮生成摘要\n2. 引用文档只存名称，用<|文档名|>标记\n3. 摘要与文档统一使用markdown格式..."
        }
      ]
    }
  ]
}
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
| precision@K | 0.263 | 0.467 | 0.756 |

- ✅[BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)

| 阈值   | 0.9  | 0.8  | 0.7  | 0.7  | 0.7  | 0.5  | 0.5  | 0.5  | 0.5  |
|--------|------|------|------|------|------|------|------|------|------|
| topk   | 1    | 1    | 1    | 2    | 3    | 1    | 2    | 3    | 5    |
| precision@K | 0.160|0.381 |0.587 |0.613 |0.621 |0.811 |0.878 |0.908 |0.929 |

**最终选择BAAI/bge-large-zh，top_p=0.5，top_k=5**

### 精排（top5 → top1）
**reranker**：✅[BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
