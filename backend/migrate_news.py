"""MongoDB 新闻数据迁移到 Milvus 向量数据库"""
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from pymilvus import MilvusClient, DataType

from embedding import SiliconFlowEmbedding

load_dotenv()


class NewsMigrator:
    """新闻数据迁移器 - MongoDB 到 Milvus"""

    def __init__(self):
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = os.getenv("MILVUS_PORT", "19530")
        self.milvus_database = os.getenv("MILVUS_DATABASE", "rag_sum_col")
        self.collection_name = "rag_summary"
        
        self.embedding_service = SiliconFlowEmbedding()
        self.milvus_client = None

    def _get_mongo_client(self) -> MongoClient:
        """创建 MongoDB 客户端"""
        host = os.getenv("MONGODB_HOST", "localhost")
        port = int(os.getenv("MONGODB_PORT", "27017"))
        username = os.getenv("MONGODB_USERNAME", "")
        password = os.getenv("MONGODB_PASSWORD", "")
        
        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}"
        else:
            uri = f"mongodb://{host}:{port}"
        
        return MongoClient(uri)

    def _get_milvus_client(self) -> MilvusClient:
        """创建 Milvus 客户端并切换到指定数据库"""
        client = MilvusClient(uri=f"http://{self.milvus_host}:{self.milvus_port}")
        
        databases = client.list_databases()
        if self.milvus_database not in databases:
            client.create_database(self.milvus_database)
            print(f"创建数据库: {self.milvus_database}")
        
        client.use_database(self.milvus_database)
        return client

    def init_collection(self, dense_dim: int = 1024):
        """初始化 Milvus collection"""
        if self.milvus_client is None:
            self.milvus_client = self._get_milvus_client()
        
        if self.milvus_client.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' 已存在")
            return
        
        schema = self.milvus_client.create_schema(
            auto_id=False,
            enable_dynamic_field=False
        )
        
        schema.add_field("id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("publish_time", DataType.INT64)
        schema.add_field("origin_name", DataType.VARCHAR, max_length=255)
        schema.add_field("title", DataType.VARCHAR, max_length=500)
        schema.add_field("url", DataType.VARCHAR, max_length=2048)
        schema.add_field("summary", DataType.VARCHAR, max_length=65535)
        schema.add_field("summary_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("summary_sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        
        index_params = self.milvus_client.prepare_index_params()
        
        index_params.add_index(
            field_name="summary_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )
        
        index_params.add_index(
            field_name="summary_sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2}
        )
        
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"创建 Collection: {self.collection_name}")

    def migrate(self, batch_size: int = 50, request_delay: float = 0.5):
        """
        执行迁移
        :param batch_size: 批量大小
        :param request_delay: API 请求间隔（秒）
        """
        mongo_client = self._get_mongo_client()
        
        try:
            db = mongo_client["kg_information_base"]
            collection = db["1_intelligence"]
            
            filter_query = {"summary": {"$exists": True, "$ne": ""}}
            projection = {
                "_id": 1,
                "publishTime": 1,
                "originName": 1,
                "title": 1,
                "url": 1,
                "summary": 1
            }
            
            total = collection.count_documents(filter_query)
            print(f"有效数据量: {total}")
            
            if total == 0:
                print("没有有效数据，退出")
                return
            
            print("读取数据...")
            all_docs = list(collection.find(filter_query, projection))
            
            print("拟合语料库（BM25）...")
            summaries = [doc.get("summary", "") for doc in all_docs]
            self.embedding_service.fit_corpus(summaries)
            
            if self.milvus_client is None:
                self.milvus_client = self._get_milvus_client()
            
            print("开始迁移...")
            success_count = 0
            failed_batches = []
            
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i:i + batch_size]
                batch_summaries = [doc.get("summary", "") for doc in batch]
                
                try:
                    dense_embeddings = self.embedding_service.get_embeddings(batch_summaries)
                    sparse_embeddings = self.embedding_service.get_sparse_embeddings(batch_summaries)
                    
                    insert_data = []
                    for j, doc in enumerate(batch):
                        publish_time = doc.get("publishTime", 0)
                        if publish_time is None:
                            publish_time = 0
                        
                        insert_data.append({
                            "id": str(doc["_id"]),
                            "publish_time": int(publish_time),
                            "origin_name": doc.get("originName", "") or "",
                            "title": doc.get("title", "") or "",
                            "url": doc.get("url", "") or "",
                            "summary": batch_summaries[j],
                            "summary_vector": dense_embeddings[j],
                            "summary_sparse_vector": sparse_embeddings[j]
                        })
                    
                    self.milvus_client.insert(self.collection_name, insert_data)
                    success_count += len(batch)
                    print(f"已插入 {min(i + batch_size, total)}/{total}")
                    
                    if request_delay > 0:
                        time.sleep(request_delay)
                        
                except Exception as e:
                    failed_batches.append({
                        "start": i,
                        "end": i + batch_size,
                        "error": str(e)
                    })
                    print(f"批次 {i}-{i + batch_size} 失败: {e}")
            
            print("\n迁移完成!")
            print(f"成功: {success_count}/{total}")
            
            if failed_batches:
                print(f"失败批次: {len(failed_batches)}")
                for fb in failed_batches:
                    print(f"  - {fb['start']}-{fb['end']}: {fb['error']}")
                    
        finally:
            mongo_client.close()

    def verify(self):
        """验证迁移结果"""
        if self.milvus_client is None:
            self.milvus_client = self._get_milvus_client()
        
        stats = self.milvus_client.get_collection_stats(self.collection_name)
        print(f"Collection 统计: {stats}")


if __name__ == "__main__":
    migrator = NewsMigrator()
    
    migrator.init_collection(dense_dim=1024)
    
    migrator.migrate(batch_size=50, request_delay=3)
    
    migrator.verify()
