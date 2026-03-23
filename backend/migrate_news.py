import os
import re
import math
from collections import Counter
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType

load_dotenv()


class NewsMigrator:
    def __init__(self, embedding_service):
        self.milvus_client = MilvusClient(uri="http://localhost:19530")
        self.collection_name = "news_collection"
        self.embedding_service = embedding_service
        self.k1 = 1.5
        self.b = 0.75
        self._vocab = {}
        self._vocab_counter = 0
        self._doc_freq = Counter()
        self._total_docs = 0
        self._avg_doc_len = 0

    def init_collection(self, dense_dim=1024):
        if not self.milvus_client.has_collection(self.collection_name):
            schema = self.milvus_client.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )
            schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
            schema.add_field("title", DataType.VARCHAR, max_length=500)
            schema.add_field("summary", DataType.VARCHAR, max_length=5000)
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

            index_params = self.milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="dense_embedding",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 256}
            )
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        english_pattern = re.compile(r'[a-zA-Z]+')
        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1
        return tokens

    def fit_corpus(self, summaries: list[str]):
        self._total_docs = len(summaries)
        total_len = 0
        for summary in summaries:
            tokens = self._tokenize(summary)
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freq[token] += 1
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_counter
                    self._vocab_counter += 1
        self._avg_doc_len = total_len / max(self._total_docs, 1)

    def _get_sparse_embedding(self, text: str) -> dict:
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector = {}

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1

            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)
            if df == 0:
                idf = math.log((self._total_docs + 1) / 1)
            else:
                idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)

            score = idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1)))
            if score > 0:
                sparse_vector[idx] = float(score)
        return sparse_vector

    def migrate(self, mongodb_uri: str, db_name: str, collection_name: str, batch_size: int = 50):
        from pymongo import MongoClient

        mongo_client = MongoClient(mongodb_uri)
        db = mongo_client[db_name]
        collection = db[collection_name]

        print("📖 从MongoDB读取数据...")
        all_docs = list(collection.find({}, {"_id": 1, "title": 1, "summary": 1}))
        print(f"   共 {len(all_docs)} 条")

        if len(all_docs) == 0:
            print("⚠️ 没有数据，退出")
            mongo_client.close()
            return

        summaries = [doc.get("summary", "") for doc in all_docs]
        print("🔧 拟合语料库...")
        self.fit_corpus(summaries)

        print("🚀 开始迁移...")
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i + batch_size]
            insert_data = []
            batch_summaries = [doc.get("summary", "") for doc in batch]

            dense_embeddings = self.embedding_service.get_embeddings(batch_summaries)

            for j, doc in enumerate(batch):
                summary = doc.get("summary", "")
                insert_data.append({
                    "id": str(doc["_id"]),
                    "title": doc.get("title", ""),
                    "summary": summary,
                    "dense_embedding": dense_embeddings[j],
                    "sparse_embedding": self._get_sparse_embedding(summary)
                })

            self.milvus_client.insert(self.collection_name, insert_data)
            print(f"   已插入 {min(i + batch_size, len(all_docs))}/{len(all_docs)}")

        print("✅ 迁移完成!")
        mongo_client.close()


if __name__ == "__main__":
    from embedding import EmbeddingService

    migrator = NewsMigrator(embedding_service=EmbeddingService())
    migrator.init_collection(dense_dim=1024)

    migrator.migrate(
        mongodb_uri="mongodb://localhost:27017",
        db_name="your_news_db",
        collection_name="your_news_collection",
        batch_size=50
    )