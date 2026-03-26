"""Milvus 客户端 - 支持密集向量+稀疏向量混合检索"""
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

load_dotenv()


class MilvusManager:
    """Milvus 连接和集合管理 - 支持混合检索"""

    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.database = os.getenv("MILVUS_DATABASE", "rag_sum_col")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "rag_summary")
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
        
        databases = self.client.list_databases()
        if self.database not in databases:
            self.client.create_database(self.database)
        self.client.use_database(self.database)

    def init_collection(self, dense_dim: int = 1024):
        """
        初始化 Milvus 集合 - 同时支持密集向量和稀疏向量
        字段定义与实际新闻数据集合一致
        :param dense_dim: 密集向量维度
        """
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

            # 主键
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

            # 新闻元数据字段
            schema.add_field("publish_time", DataType.INT64)
            schema.add_field("origin_name", DataType.VARCHAR, max_length=255)
            schema.add_field("title", DataType.VARCHAR, max_length=512)
            schema.add_field("url", DataType.VARCHAR, max_length=1024)
            schema.add_field("summary", DataType.VARCHAR, max_length=4000)

            # 密集向量（来自 embedding 模型）
            schema.add_field("summary_vector", DataType.FLOAT_VECTOR, dim=dense_dim)

            # 稀疏向量（来自 BM25）
            schema.add_field("summary_sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

            # 为两种向量分别创建索引
            index_params = self.client.prepare_index_params()

            # 密集向量索引 - 使用 HNSW（更适合混合检索）
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

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

    def insert(self, data: list[dict]):
        """插入数据到 Milvus"""
        return self.client.insert(self.collection_name, data)

    def query(self, filter_expr: str = "", output_fields: list[str] = None, limit: int = 10000):
        """查询数据"""
        default_fields = ["id", "publish_time", "origin_name", "title", "url", "summary"]
        return self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields or default_fields,
            limit=limit
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        混合检索 - 使用 RRF 融合密集向量和稀疏向量的检索结果
        
        :param dense_embedding: 密集向量
        :param sparse_embedding: 稀疏向量 {index: value, ...}
        :param top_k: 返回结果数量
        :param rrf_k: RRF 算法参数 k，默认60
        :return: 检索结果列表
        """
        output_fields = [
            "id",
            "summary",
            "title",
            "origin_name",
            "url",
            "publish_time",
        ]
        
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="summary_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="summary_sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        
        reranker = RRFRanker(k=rrf_k)
        
        print(f"[MILVUS] hybrid_search: collection={self.collection_name}, database={self.database}")
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_search, sparse_search],
            ranker=reranker,
            limit=top_k,
            output_fields=output_fields
        )
        print(f"[MILVUS] hybrid_search 返回: {len(results)} 组")
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "summary": hit.get("summary", ""),
                    "title": hit.get("title", ""),
                    "origin_name": hit.get("origin_name", ""),
                    "url": hit.get("url", ""),
                    "publish_time": hit.get("publish_time", 0),
                    "score": hit.get("distance", 0.0)
                })
        
        return formatted_results

    def dense_retrieve(self, dense_embedding: list[float], top_k: int = 5, filter_expr: str = "") -> list[dict]:
        """
        仅使用密集向量检索（降级模式，用于稀疏向量不可用时）
        """
        print(f"[MILVUS] dense_retrieve: collection={self.collection_name}, database={self.database}, top_k={top_k}")
        results = self.client.search(
            collection_name=self.collection_name,
            data=[dense_embedding],
            anns_field="summary_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=[
                "id",
                "summary",
                "title",
                "origin_name",
                "url",
                "publish_time",
            ],
            filter=filter_expr,
        )
        print(f"[MILVUS] dense_retrieve 返回: {len(results)} 组")
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "summary": hit.get("entity", {}).get("summary", ""),
                    "title": hit.get("entity", {}).get("title", ""),
                    "origin_name": hit.get("entity", {}).get("origin_name", ""),
                    "url": hit.get("entity", {}).get("url", ""),
                    "publish_time": hit.get("entity", {}).get("publish_time", 0),
                    "score": hit.get("distance", 0.0)
                })
        
        return formatted_results

    def delete(self, filter_expr: str):
        """删除数据"""
        return self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        return self.client.has_collection(self.collection_name)

    def drop_collection(self):
        """删除集合（用于重建 schema）"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
