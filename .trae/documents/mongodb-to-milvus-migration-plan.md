# MongoDB 新闻数据向量化迁移方案

## 一、需求概述

将 MongoDB 数据库 `kg_information_base` 中 `1_intelligence` collection 的新闻数据迁移到 Milvus 向量数据库，对 `summary` 字段进行向量化。

### 源数据字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` | 混合类型 | 大多数为数字（如 60110），个别为 ObjectId |
| `publishTime` | INT64 | 毫秒时间戳（如 1758610800000） |
| `originName` | String | 来源名称 |
| `title` | String | 新闻标题 |
| `url` | String | 新闻链接 |
| `summary` | String | 新闻摘要，**需要向量化**（部分数据不存在，直接跳过） |

### 配置信息（已确认）
- **数据量**: 约 30,000 条（实际以有效 summary 数量为准）
- **向量 API**: 硅基流动 (SiliconFlow)
- **模型**: BAAI/bge-m3（1024 维）
- **API Key 变量名**: `SILICONFLOW_API_KEY`
- **混合检索**: 需要（密集向量 + 稀疏向量 BM25）
- **MongoDB 认证**: 已在 `.env` 配置
- **Milvus 数据库**: `rag_sum_col`
- **Milvus Collection 名称**: `rag_summary`
- **相似度度量**: 余弦相似度（COSINE）

---

## 二、技术方案

### 2.1 架构设计

```
MongoDB (kg_information_base.1_intelligence)
         ↓
    统计有效数据: count_documents({"summary": {"$exists": True, "$ne": ""}})
         ↓
    读取数据 (_id, publishTime, originName, title, url, summary)
    过滤条件: {"summary": {"$exists": True, "$ne": ""}}
         ↓
    ┌─────────────────────────────────────┐
    │  硅基流动 API (BAAI/bge-m3)          │
    │  → summary_vector (密集向量)          │
    │                                     │
    │  BM25 算法                           │
    │  → summary_sparse_vector (稀疏向量)   │
    └─────────────────────────────────────┘
         ↓
    Milvus (数据库: rag_sum_col, Collection: rag_summary)
```

### 2.2 Milvus Collection Schema

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | VARCHAR(64) | 主键，`_id` 统一转为字符串 |
| `publish_time` | INT64 | 发布时间戳（毫秒） |
| `origin_name` | VARCHAR(255) | 来源名称 |
| `title` | VARCHAR(500) | 新闻标题 |
| `url` | VARCHAR(1024) | 新闻链接 |
| `summary` | VARCHAR(5000) | 新闻摘要 |
| `summary_vector` | FLOAT_VECTOR(1024) | BGE-M3 密集向量 |
| `summary_sparse_vector` | SPARSE_FLOAT_VECTOR | BM25 稀疏向量 |

### 2.3 索引配置

```python
# 密集向量索引 - HNSW + 余弦相似度
index_params.add_index(
    field_name="summary_vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)

# 稀疏向量索引
index_params.add_index(
    field_name="summary_sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={"drop_ratio_build": 0.2}
)
```

---

## 三、关键问题处理

### 3.1 `_id` 字段处理
- **问题**: `_id` 有两种格式（数字和 ObjectId）
- **方案**: 统一转换为字符串存储
```python
id_str = str(doc["_id"])  # 60110 → "60110", ObjectId(...) → "507f1f77bcf86cd799439011"
```

### 3.2 `summary` 字段缺失处理
- **问题**: 部分 `summary` 不存在
- **方案**: 直接跳过，使用 MongoDB 查询过滤
```python
# 统计有效数据量
total = collection.count_documents({"summary": {"$exists": True, "$ne": ""}})

# 查询时过滤
all_docs = list(collection.find(
    {"summary": {"$exists": True, "$ne": ""}}, 
    {"_id": 1, "publishTime": 1, "originName": 1, "title": 1, "url": 1, "summary": 1}
))
```

### 3.3 BM25 稀疏向量说明
- **目标**: 对 `summary` 字段进行 BM25 稀疏向量化
- **用途**: 关键词匹配，与密集向量互补
- **字段名**: `summary_sparse_vector`

### 3.4 Milvus 数据库配置
- **数据库名**: `rag_sum_col`
- **Collection 名**: `rag_summary`
```python
# 连接时指定数据库
client = MilvusClient(uri="http://localhost:19530")
client.use_database("rag_sum_col")  # 切换到指定数据库
```

---

## 四、实施步骤

### Step 1: 更新环境变量配置
在 `.env` 中确认/添加以下配置：
```env
# 硅基流动 API
SILICONFLOW_API_KEY=your-api-key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBED_MODEL=BAAI/bge-m3

# MongoDB（已配置）
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=xxx
MONGODB_PASSWORD=xxx

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DATABASE=rag_sum_col
```

### Step 2: 修改 embedding.py
添加硅基流动 API 支持：
- 新增 `SiliconFlowEmbedding` 类
- 支持批量向量化
- 错误处理和重试机制

### Step 3: 重写迁移脚本 migrate_news.py
- 修改 MongoDB 连接（使用认证）
- 修改字段映射（适配新 schema）
- 处理 `_id` 混合格式
- 过滤无 `summary` 的数据
- 修改向量字段名称
- 使用余弦相似度
- 指定 Milvus 数据库 `rag_sum_col`

### Step 4: 执行迁移
```bash
python migrate_news.py
```

### Step 5: 验证结果
- 检查数据完整性
- 测试向量检索功能

---

## 五、代码实现细节

### 5.1 硅基流动 Embedding 服务

```python
class SiliconFlowEmbedding:
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = os.getenv("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3")
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        response = requests.post(f"{self.base_url}/embeddings", headers=headers, json=data)
        return [item["embedding"] for item in response.json()["data"]]
```

### 5.2 迁移脚本核心逻辑

```python
def migrate(self, batch_size: int = 50):
    # 1. 连接 MongoDB
    collection = mongo_client["kg_information_base"]["1_intelligence"]
    
    # 2. 统计有效数据量
    total = collection.count_documents({"summary": {"$exists": True, "$ne": ""}})
    print(f"有效数据量: {total}")
    
    # 3. 读取数据（过滤无 summary 的数据）
    all_docs = list(collection.find(
        {"summary": {"$exists": True, "$ne": ""}}, 
        {"_id": 1, "publishTime": 1, "originName": 1, "title": 1, "url": 1, "summary": 1}
    ))
    
    # 4. 拟合语料库（BM25）
    summaries = [doc.get("summary", "") for doc in all_docs]
    self.embedding_service.fit_corpus(summaries)
    
    # 5. 连接 Milvus 并切换数据库
    self.milvus_client = MilvusClient(uri="http://localhost:19530")
    self.milvus_client.use_database("rag_sum_col")
    
    # 6. 分批迁移
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        batch_summaries = [doc.get("summary", "") for doc in batch]
        
        # 生成向量
        dense_embeddings = self.embedding_service.get_embeddings(batch_summaries)
        sparse_embeddings = self.embedding_service.get_sparse_embeddings(batch_summaries)
        
        insert_data = []
        for j, doc in enumerate(batch):
            insert_data.append({
                "id": str(doc["_id"]),
                "publish_time": doc.get("publishTime", 0),
                "origin_name": doc.get("originName", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "summary": batch_summaries[j],
                "summary_vector": dense_embeddings[j],
                "summary_sparse_vector": sparse_embeddings[j]
            })
        
        self.milvus_client.insert("rag_summary", insert_data)
        print(f"已插入 {min(i + batch_size, total)}/{total}")
```

---

## 六、风险与注意事项

| 风险 | 缓解措施 |
|------|----------|
| API 限流 | 控制批量大小（建议 50 条/批），添加请求间隔 |
| 网络超时 | 添加重试机制，记录失败数据 |
| 内存溢出 | 分批处理，及时释放内存 |
| 数据不一致 | 迁移前备份，迁移后验证 |

### 预估时间
- 有效数据量以实际统计为准
- 批量大小 50 条
- 每批次约 2-3 秒（含 API 调用）
- **总耗时约 20-30 分钟**

---

## 七、文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `.env` | 确认 | 确保硅基流动 API 配置正确 |
| `embedding.py` | 修改 | 添加硅基流动 API 支持 |
| `migrate_news.py` | 重写 | 适配新的数据结构和 schema |

---

## 八、后续扩展（可选）

1. **增量迁移**: 基于 `publishTime` 判断新增数据
2. **数据更新**: 支持更新已存在的向量
3. **检索服务**: 封装检索 API，支持混合检索
