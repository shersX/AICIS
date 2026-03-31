"""MongoDB 新闻数据迁移到 Milvus 向量数据库"""
import os
import time
import signal
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from pymilvus import MilvusClient, DataType
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from urllib.parse import quote_plus

from embedding import SiliconFlowEmbedding

load_dotenv()


class NewsMigrator:
    """新闻数据迁移器 - MongoDB 到 Milvus"""

    def __init__(self, state_file: str = "data/bm25_state.json"):
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = os.getenv("MILVUS_PORT", "19530")
        self.milvus_database = os.getenv("MILVUS_DATABASE", "rag_sum_col")
        self.collection_name = "rag_summary"
        
        self.embedding_service = SiliconFlowEmbedding(state_file=state_file)
        self.milvus_client = None

    def _get_mongo_client(self) -> MongoClient:
        """创建 MongoDB 客户端"""
        host = os.getenv("MONGODB_HOST", "localhost")
        port = int(os.getenv("MONGODB_PORT", "27017"))
        username = quote_plus(os.getenv("MONGODB_USERNAME", ""))
        password = quote_plus(os.getenv("MONGODB_PASSWORD", ""))
        
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

    def _get_max_publish_time(self) -> int:
        """从 Milvus 获取最大的 publishTime"""
        if self.milvus_client is None:
            self.milvus_client = self._get_milvus_client()
        
        try:
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["publish_time"],
                limit=1,
                sort_by="publish_time",
                sort_order="desc"
            )
            if results:
                return results[0].get("publish_time", 0)
        except Exception:
            pass
        return 0

    def _check_doc_exists(self, doc_id: str) -> tuple[bool, str]:
        """检查文档是否存在于 Milvus，返回 (是否存在, summary)"""
        if self.milvus_client is None:
            self.milvus_client = self._get_milvus_client()
        
        try:
            results = self.milvus_client.get(
                collection_name=self.collection_name,
                ids=[doc_id],
                output_fields=["summary"]
            )
            if results:
                return True, results[0].get("summary", "")
        except Exception:
            pass
        return False, ""

    def _delete_docs(self, doc_ids: list[str]):
        """删除 Milvus 中的文档"""
        if self.milvus_client is None:
            self.milvus_client = self._get_milvus_client()
        
        self.milvus_client.delete(
            collection_name=self.collection_name,
            ids=doc_ids
        )

    def _insert_docs(self, insert_data: list[dict]):
        """插入文档到 Milvus"""
        if self.milvus_client is None:
            self.milvus_client = self._get_milvus_client()
        
        self.milvus_client.insert(self.collection_name, insert_data)

    def migrate_full(self, batch_size: int = 50, request_delay: float = 0.5):
        """
        全量迁移
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
            self.embedding_service.fit_corpus(summaries, incremental=False)
            
            if self.milvus_client is None:
                self.milvus_client = self._get_milvus_client()
            
            print("开始迁移...")
            success_count = 0
            failed_batches = []
            max_publish_time = 0
            
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
                        
                        if publish_time > max_publish_time:
                            max_publish_time = publish_time
                        
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
                    
                    self._insert_docs(insert_data)
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
            
            self.embedding_service.set_last_sync_time(max_publish_time)
            self.embedding_service.save_state()
            
            print("\n迁移完成!")
            print(f"成功: {success_count}/{total}")
            
            if failed_batches:
                print(f"失败批次: {len(failed_batches)}")
                for fb in failed_batches:
                    print(f"  - {fb['start']}-{fb['end']}: {fb['error']}")
                    
        finally:
            mongo_client.close()

    def migrate_incremental(self, batch_size: int = 50, request_delay: float = 0.5):
        """
        增量迁移
        :param batch_size: 批量大小
        :param request_delay: API 请求间隔（秒）
        """
        mongo_client = self._get_mongo_client()
        
        try:
            db = mongo_client["kg_information_base"]
            collection = db["1_intelligence"]
            
            last_sync_time = self.embedding_service.get_last_sync_time()
            print(f"上次同步时间: {last_sync_time}")
            
            filter_query = {
                "summary": {"$exists": True, "$ne": ""},
                "publishTime": {"$gt": last_sync_time}
            }
            projection = {
                "_id": 1,
                "publishTime": 1,
                "originName": 1,
                "title": 1,
                "url": 1,
                "summary": 1
            }
            
            candidates = list(collection.find(filter_query, projection))
            total = len(candidates)
            print(f"候选数据量: {total}")
            
            if total == 0:
                print("没有新数据，退出")
                return
            
            new_docs = []
            update_docs = []
            
            print("检查数据变更...")
            for doc in candidates:
                doc_id = str(doc["_id"])
                exists, existing_summary = self._check_doc_exists(doc_id)
                
                if not exists:
                    new_docs.append(doc)
                elif existing_summary != doc.get("summary", ""):
                    update_docs.append(doc)
            
            print(f"新增: {len(new_docs)}, 更新: {len(update_docs)}")
            
            if not new_docs and not update_docs:
                print("没有需要处理的数据")
                return
            
            all_process_docs = new_docs + update_docs
            
            if all_process_docs:
                print("增量拟合语料库（BM25）...")
                summaries = [doc.get("summary", "") for doc in all_process_docs]
                self.embedding_service.fit_corpus(summaries, incremental=True)
            
            if update_docs:
                print("删除需要更新的文档...")
                update_ids = [str(doc["_id"]) for doc in update_docs]
                for i in range(0, len(update_ids), batch_size):
                    batch_ids = update_ids[i:i + batch_size]
                    self._delete_docs(batch_ids)
            
            success_count = 0
            failed_batches = []
            max_publish_time = last_sync_time
            
            all_process_docs = new_docs + update_docs
            
            print("开始迁移...")
            for i in range(0, len(all_process_docs), batch_size):
                batch = all_process_docs[i:i + batch_size]
                batch_summaries = [doc.get("summary", "") for doc in batch]
                
                try:
                    dense_embeddings = self.embedding_service.get_embeddings(batch_summaries)
                    sparse_embeddings = self.embedding_service.get_sparse_embeddings(batch_summaries)
                    
                    insert_data = []
                    for j, doc in enumerate(batch):
                        publish_time = doc.get("publishTime", 0)
                        if publish_time is None:
                            publish_time = 0
                        
                        if publish_time > max_publish_time:
                            max_publish_time = publish_time
                        
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
                    
                    self._insert_docs(insert_data)
                    success_count += len(batch)
                    print(f"已处理 {min(i + batch_size, len(all_process_docs))}/{len(all_process_docs)}")
                    
                    if request_delay > 0:
                        time.sleep(request_delay)
                        
                except Exception as e:
                    failed_batches.append({
                        "start": i,
                        "end": i + batch_size,
                        "error": str(e)
                    })
                    print(f"批次 {i}-{i + batch_size} 失败: {e}")
            
            self.embedding_service.set_last_sync_time(max_publish_time)
            self.embedding_service.save_state()
            
            print("\n增量迁移完成!")
            print(f"成功: {success_count}/{len(all_process_docs)}")
            
            if failed_batches:
                print(f"失败批次: {len(failed_batches)}")
                for fb in failed_batches:
                    print(f"  - {fb['start']}-{fb['end']}: {fb['error']}")
                    
        finally:
            mongo_client.close()

    def migrate(self, batch_size: int = 50, request_delay: float = 0.5, force_full: bool = False):
        """
        智能迁移：自动判断全量或增量
        :param batch_size: 批量大小
        :param request_delay: API 请求间隔（秒）
        :param force_full: 是否强制全量迁移
        """
        if force_full or self.embedding_service.get_last_sync_time() == 0:
            print("执行全量迁移...")
            self.migrate_full(batch_size, request_delay)
        else:
            print("执行增量迁移...")
            self.migrate_incremental(batch_size, request_delay)

    def verify(self):
        """验证迁移结果"""
        if self.milvus_client is None:
            self.milvus_client = self._get_milvus_client()
        
        stats = self.milvus_client.get_collection_stats(self.collection_name)
        print(f"Collection 统计: {stats}")


def run_scheduled_migration(interval_hours: int = 24):
    """
    运行定时增量迁移服务
    :param interval_hours: 间隔时间（小时），默认 24 小时
    """
    migrator = NewsMigrator()
    scheduler = BlockingScheduler()
    
    def job_wrapper():
        print(f"\n{'='*50}")
        print(f"开始定时增量迁移 ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{'='*50}")
        migrator.init_collection(dense_dim=1024)
        migrator.migrate(batch_size=50, request_delay=3)
        migrator.verify()
        print(f"{'='*50}")
        print(f"增量迁移完成 ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{'='*50}\n")
    
    scheduler.add_job(
        job_wrapper,
        trigger=IntervalTrigger(hours=interval_hours),
        id='daily_incremental_migrate',
        name='每日增量迁移',
        replace_existing=True
    )
    
    def signal_handler(signum, frame):
        print("\n收到退出信号，正在关闭调度器...")
        scheduler.shutdown(wait=True)
        print("调度器已关闭")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"定时增量迁移服务启动，每隔 {interval_hours} 小时执行一次")
    print("按 Ctrl+C 或发送 SIGTERM 信号退出")
    
    job_wrapper()
    
    scheduler.start()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MongoDB 新闻数据迁移到 Milvus")
    parser.add_argument("--mode", choices=["once", "schedule"], default="once",
                        help="运行模式: once=单次执行, schedule=定时调度")
    parser.add_argument("--interval", type=int, default=24,
                        help="调度间隔（小时），仅在 schedule 模式下有效")
    parser.add_argument("--force-full", action="store_true",
                        help="强制全量迁移")
    
    args = parser.parse_args()
    
    if args.mode == "schedule":
        run_scheduled_migration(interval_hours=args.interval)
    else:
        migrator = NewsMigrator()
        migrator.init_collection(dense_dim=1024)
        migrator.migrate(batch_size=50, request_delay=3, force_full=args.force_full)
        migrator.verify()
