# AICIS知识情报助手 产品需求文档 (PRD)

## 1. 产品概述

### 1.1 产品名称与定位

**产品名称**：AICIS知识情报助手

**产品定位**：基于大语言模型（LLM）的智能情报助手，融合 RAG（检索增强生成）技术，支持实时天气查询、多轮会话记忆功能。

**目标用户**：
- 企业内部情报分析人员
- 需要快速获取实时信息并保持多轮对话上下文的用户
- 希望构建智能问答系统的开发者

### 1.2 核心价值主张

- **混合检索能力**：结合密集向量（语义相似）与稀疏向量（BM25词频），提升检索召回率
- **Auto-Merging 机制**：智能合并碎片化检索结果，还原完整段落上下文
- **多工具扩展**：支持天气查询等外部工具接入，可灵活扩展 Agent 工具集
- **会话记忆**：支持多会话管理，自动摘要保持长对话上下文

---

## 2. 用户场景与功能

### 2.1 主要用户场景

| 场景 | 描述 | 关键功能 |
|------|------|----------|
| **多轮对话咨询** | 用户与助手进行多轮自然语言对话 | 会话记忆、流式输出、RAG追踪展示 |
| **天气信息查询** | 用户询问天气时返回实时天气数据 | 高德天气API工具调用 |
| **历史会话管理** | 用户查看、切换、清空历史会话 | 会话列表、会话删除、清空对话 |

### 2.2 功能清单

#### 2.2.1 会话管理
- [x] 新建会话
- [x] 获取会话消息历史
- [x] 获取用户所有会话列表（按更新时间倒序）
- [x] 删除指定会话
- [x] 清空当前对话

#### 2.2.2 RAG 检索增强
- [x] 混合检索（密集向量 + BM25 稀疏向量 + RRF 融合）
- [x] Step-back Expansion（泛化问题扩展）
- [x] HyDE（假设文档生成）
- [x] 查询路由（根据策略选择扩展方式）
- [x] 相关性评分（Grader）
- [x] 重排（Rerank，可选）
- [x] Auto-Merging Retriever（自动合并父子块）
- [x] RAG 步骤追踪（RAG Trace）

#### 2.2.3 Agent 工具
- [x] `get_current_weather`：查询实时天气
- [x] `search_knowledge_base`：知识库检索
- [x] `get_last_rag_context`：获取最近 RAG 上下文

#### 2.2.4 前端功能
- [x] 知识助手主题 UI（Vue 3 CDN 单页应用）
- [x] 流式输出（SSE 实时响应）
- [x] Markdown 渲染与代码高亮
- [x] 打字机效果动画
- [x] RAG 步骤实时展示（检索→重排→合并）
- [x] 响应中断（AbortController）

---

## 3. 系统架构

### 3.1 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| **前端** | Vue 3 (CDN) + Marked.js + Highlight.js | 轻量级单页应用，无需构建 |
| **后端** | FastAPI + Uvicorn | 高性能异步 API 框架 |
| **LLM 框架** | LangChain + LangGraph | Agent 构建、RAG Pipeline 状态图 |
| **向量数据库** | Milvus v2.5.14 | 高性能向量检索 |
| **元数据存储** | etcd v3.5.18 | Milvus 依赖 |
| **对象存储** | MinIO | Milvus 依赖 |
| **本地存储** | JSON 文件 | 会话历史 |
| **环境管理** | python-dotenv | 配置管理 |

### 3.2 系统架构图

```
┌──────────────────────────────────────────────────────────────┐
│                        用户浏览器                             │
│                   Vue 3 单页应用 (http://127.0.0.1:8000/)     │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTP/SSE
┌────────────────────────────▼─────────────────────────────────┐
│                    FastAPI 后端服务                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │  API Router │  │   Agent     │  │     RAG Pipeline         ││
│  │  (会话)     │  │ (LangChain) │  │   (LangGraph StateGraph) ││
│  └─────────────┘  └─────────────┘  └─────────────────────────┘│
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │   Tools     │  │ Embedding   │  │     Milvus Client       ││
│  │ (天气/检索)  │  │  Service    │  │                         ││
│  └─────────────┘  └─────────────┘  └─────────────────────────┘│
└────────────────────────────┬─────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌────────────┐    ┌────────────┐    ┌────────────────┐
   │   Milvus   │    │ Parent     │    │    会话历史     │
   │  VectorDB  │    │ Chunk Store│    │   (JSON)       │
   │  :19530    │    │  (JSON)    │    │                │
   └────────────┘    └────────────┘    └────────────────┘
```

### 3.3 Docker 部署架构

| 服务 | 镜像 | 端口 | 容器名 | 用途 |
|------|------|------|--------|------|
| etcd | quay.io/coreos/etcd:v3.5.18 | 2379 | milvus-etcd | Milvus 元数据存储 |
| minio | minio/minio:RELEASE.2024-05-28T17-19-04Z | 9000/9001 | milvus-minio | Milvus 对象存储 |
| standalone | milvusdb/milvus:v2.5.14 | 19530 | milvus-standalone | 向量数据库主服务 |

---

## 4. 数据模型

### 4.1 会话历史存储 (customer_service_history.json)

```json
{
  "<user_id>": {
    "<session_id>": {
      "messages": [
        {
          "type": "human|ai|system",
          "content": "消息内容",
          "timestamp": "2024-01-01T12:00:00.000Z",
          "rag_trace": { ... }
        }
      ],
      "metadata": {},
      "updated_at": "2024-01-01T12:00:00.000Z"
    }
  }
}
```

---

## 5. API 规格

### 5.1 基础信息

- **Base URL**：`http://127.0.0.1:8000`
- **API 文档**：`http://127.0.0.1:8000/docs`
- **认证**：无（基于 user_id/session_id 的简单隔离）

### 5.2 会话管理 API

#### GET `/sessions/{user_id}/{session_id}`
获取指定会话的所有消息

**响应**：
```json
{
  "messages": [
    {
      "type": "human",
      "content": "用户消息",
      "timestamp": "2024-01-01T12:00:00",
      "rag_trace": null
    }
  ]
}
```

#### GET `/sessions/{user_id}`
获取用户的所有会话列表

**响应**：
```json
{
  "sessions": [
    {
      "session_id": "session_xxx",
      "updated_at": "2024-01-01T12:00:00",
      "message_count": 10
    }
  ]
}
```

#### DELETE `/sessions/{user_id}/{session_id}`
删除指定会话

**响应**：
```json
{
  "session_id": "session_xxx",
  "message": "成功删除会话"
}
```

### 5.3 聊天 API

#### POST `/chat`
普通聊天（非流式）

**请求**：
```json
{
  "message": "用户问题",
  "user_id": "user_xxx",
  "session_id": "session_xxx"
}
```

**响应**：
```json
{
  "response": "AI回答内容",
  "rag_trace": {
    "tool_used": true,
    "tool_name": "search_knowledge_base",
    "retrieval_mode": "hybrid",
    "auto_merge_applied": true,
    "retrieved_chunks": [...]
  }
}
```

#### POST `/chat/stream`
流式聊天（SSE）

**请求**：同 `/chat`

**响应**：SSE 流式输出

---

## 6. RAG Pipeline 详解

### 6.1 流程图

```
User Question
     │
     ▼
┌─────────────┐
│  Query      │
│  Expansion  │ ← step_back / hyde / complex
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │ ← hybrid_search (dense + sparse + RRF)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Grade     │ ← 二值评分 (yes/no)
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Route    │ ← 是否重写查询?
└──────┬──────┘
       │
  ┌────┴────┐
  │ Rewrite │ ──→ 回到 Retrieval
  │  Query  │
  └────┬────┘
       │
       ▼
┌─────────────┐
│   Rerank    │ ← 可选 (RERANK_MODEL)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Auto-Merge   │ ← 合并碎片化子块为父块
└──────┬──────┘
       │
       ▼
  Final Context
       │
       ▼
     LLM Generate
```

### 6.2 关键参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| Auto-Merge 开关 | `AUTO_MERGE_ENABLED` | `true` | 是否启用自动合并 |
| Auto-Merge 阈值 | `AUTO_MERGE_THRESHOLD` | `2` | 子块数量达到阈值时合并 |
| 叶子检索层级 | `LEAF_RETRIEVE_LEVEL` | `3` | 从 Level 3 开始检索 |
| 重排模型 | `RERANK_MODEL` | - | 可选，不配置则跳过重排 |
| 重排端点 | `RERANK_BINDING_HOST` | - | 重排 API 地址 |
| 评分模型 | `GRADE_MODEL` | `gpt-4.1` | 用于文档相关性评分 |

---

## 7. 环境配置

### 7.1 必需配置 (.env)

```env
# ===== Model =====
ARK_API_KEY=your_ark_api_key
MODEL=your_model_name
BASE_URL=https://your-llm-endpoint/v1
EMBEDDER=your_embedding_model

# ===== Milvus =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
```

### 7.2 可选配置

```env
# ===== Rerank =====
RERANK_MODEL=your_rerank_model
RERANK_BINDING_HOST=https://your-rerank-host
RERANK_API_KEY=your_rerank_api_key

# ===== Weather Tool =====
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=your_amap_api_key

# ===== RAG Advanced =====
GRADE_MODEL=gpt-4.1
AUTO_MERGE_ENABLED=true
AUTO_MERGE_THRESHOLD=2
LEAF_RETRIEVE_LEVEL=3
MILVUS_COLLECTION=embeddings_collection
```

---

## 8. 性能指标

### 8.1 目标指标

| 指标 | 目标值 |
|------|--------|
| API 响应时间 (不含 LLM) | < 500ms |
| 向量检索延迟 | < 100ms |
| 同时在线会话数 | > 100 |

---

## 9. 验收标准

### 9.1 功能验收

| 编号 | 验收条件 | 验证方式 |
|------|----------|----------|
| AC-01 | 多轮对话中，AI 能记住之前的上下文 | 进行 3 轮以上对话，验证上下文连贯性 |
| AC-02 | 流式输出正常，前端显示打字机效果 | 观察前端响应是否有流式动画 |
| AC-03 | RAG 步骤正确展示（检索→合并→生成） | 提问触发 RAG，检查前端追踪面板 |
| AC-04 | 会话列表按更新时间倒序排列 | 创建新会话，检查列表顺序 |
| AC-05 | 删除会话后，该会话消息不再显示 | 删除会话，验证列表和详情 |
| AC-06 | 天气查询工具正常返回天气信息 | 使用「北京天气怎么样」提问 |
| AC-07 | 未配置重排时，系统自动降级不报错 | 不设置 RERANK_MODEL，验证功能正常 |

### 9.2 非功能验收

| 编号 | 验收条件 | 验证方式 |
|------|----------|----------|
| NF-01 | Docker Compose 启动后，所有服务健康检查通过 | `docker compose ps` 检查 status |
| NF-02 | Milvus 健康检查通过 (`/healthz`) | `curl http://localhost:9091/healthz` |
| NF-03 | 前端页面在主流浏览器可正常访问 | Chrome/Edge/Firefox 测试 |
| NF-04 | 敏感信息（API Key）不暴露在前端 | 检查 Network 面板，确认无明文 Key |

---

## 10. 附录

### 10.1 术语表

| 术语 | 说明 |
|------|------|
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| HyDE | Hypothetical Document Embedding，假设文档嵌入 |
| RRF | Reciprocal Rank Fusion，排名融合 |
| Auto-Merge | 自动合并，将多个相关子块合并为父级块 |
| Dense Vector | 密集向量，由深度学习模型生成的语义向量 |
| Sparse Vector | 稀疏向量，基于词频（如 BM25）的向量 |

### 10.2 参考资料

- [Milvus 文档](https://milvus.io/docs)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
