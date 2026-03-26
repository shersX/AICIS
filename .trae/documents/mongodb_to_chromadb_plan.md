# MongoDB 数据迁移至 ChromaDB 向量知识库

## 任务目标

将 MongoDB 中的 30000 条新闻数据迁移至 ChromaDB，构建基于 summary 向量化的问答知识库。

## 数据字段

* `_id` - MongoDB 文档ID

* `publishTime` - 发布时间

* `originName` - 来源名称

* `title` - 标题

* `url` - 原文链接

* `summary` - 摘要（用于向量化）

## 技术方案

### 推荐方案：先入库再向量化（分离操作）

将数据入库与向量化分离，分两个独立阶段执行：

### 阶段一：数据入库（快速入库）

* 分批从 MongoDB 读取数据（每批 500-1000 条）

* 直接存入 ChromaDB，仅存储 metadata（不包含向量）

* 添加 `vectorized` 字段标记是否已向量化

* 阶段一完成后，所有原始数据和 metadata 已入库

### 阶段二：批量向量化（后台执行）

* 分批处理，每批 20 条（符合 API 限制）

* 使用硅基流动 API 对 summary 进行向量化

* 更新 ChromaDB 中对应记录的向量

* 标记 `vectorized = True`

* 设置合适的批次间隔，避免触发速率限制

### 优势分析

* 数据入库快速完成，不受 API 限制影响

* 向量化可独立重试，不影响原始数据完整性

* 可灵活调整向量化速度和重试策略

* 支持断点续传

## 环境准备

* 安装必要依赖库（pymongo, chromadb, requests 等）

* 确认 MongoDB 连接信息（host, port, database, collection）

* 确认 ChromaDB 持久化路径

* 获取硅基流动 API Key

## 数据连接模块

* 创建 MongoDB 连接配置

* 创建 ChromaDB 客户端配置（持久化模式）

## 硅基流动 API 向量化

* API 地址：`https://api.siliconflow.cn/v1/embeddings`

* Embedding 模型：`BAAI/bge-m3`

* 每批处理：20 条

## 关键配置项

```python
MONGODB_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "database": "news_db",
    "collection": "news"
}

CHROMADB_CONFIG = {
    "persist_directory": "./chromadb_data",
    "collection_name": "news_knowledge_base"
}

SILICONFLOW_CONFIG = {
    "api_url": "https://api.siliconflow.cn/v1/embeddings",
    "api_key": "your-api-key",
    "model": "BAAI/bge-m3"
}

BATCH_SIZE_IMPORT = 500   # 入库阶段每批数量
BATCH_SIZE_EMBED = 20     # 向量化阶段每批数量
EMBEDDING_INTERVAL = 1    # 每批间隔（秒），避免速率限制
```

## 元数据存储结构

每条记录存储到 ChromaDB 时，metadata 包含：

* `_id`: MongoDB 文档ID

* `publishTime`: 发布时间

* `originName`: 来源名称

* `title`: 标题

* `url`: 原文链接

* `summary`: 摘要文本

* `vectorized`: 是否已向量化（布尔值）

## 执行流程

### 流程图

```
┌─────────────────────────────────────┐
│          阶段一：数据入库            │
│  MongoDB → ChromaDB (仅metadata)   │
│  批量: 500条/批                     │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│          阶段二：批量向量化          │
│  ChromaDB → 硅基流动API → ChromaDB │
│  批量: 20条/批，间隔1秒            │
└─────────────────────────────────────┘
```

### 阶段一伪代码

```python
def import_data_to_chromadb():
    for batch in fetch_mongodb_batches(batch_size=500):
        documents = []
        metadatas = []
        ids = []

        for doc in batch:
            documents.append(doc["summary"])
            metadatas.append({
                "_id": str(doc["_id"]),
                "publishTime": doc.get("publishTime"),
                "originName": doc.get("originName"),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "summary": doc.get("summary"),
                "vectorized": False
            })
            ids.append(str(doc["_id"]))

        chromadb.add(ids=ids, documents=documents, metadatas=metadatas)
```

### 阶段二伪代码

```python
def vectorize_summary():
    unvectorized = chromadb.get(where={"vectorized": False})
    for i in range(0, len(unvectorized), 20):
        batch = unvectorized[i:i+20]
        embeddings = call_siliconflow_api([doc.summary for doc in batch])

        for doc_id, embedding in zip(batch.ids, embeddings):
            chromadb.update(ids=[doc_id], embeddings=[embedding])

        time.sleep(1)
```

