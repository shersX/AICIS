"""文本向量化服务 - 支持密集向量和稀疏向量（BM25）"""
import os
import json
import math
import time
import requests
import jieba
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

STOPWORDS = set([
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
    "或", "一个", "没有", "我们", "你们", "他们", "它们", "这个",
    "那个", "之", "也", "在", "有", "中", "为", "上", "个", "我",
    "他", "她", "它", "这", "那", "什么", "怎么", "如何", "为什么",
    "因为", "所以", "但是", "然而", "如果", "虽然", "可以", "可能",
    "应该", "需要", "能", "会", "要", "把", "被", "让", "给", "从",
    "到", "以", "等", "时", "地", "得", "着", "过", "来", "去",
    "又", "还", "再", "才", "只", "不", "很", "更", "最", "太",
    "非常", "比较", "相当", "一些", "这些", "那些", "这样", "那样",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but",
    "if", "or", "because", "until", "while", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "what", "which",
    "who", "whom", "this", "that", "am", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "its", "itself", "they", "them",
    "their", "theirs", "themselves"
])


class SiliconFlowEmbedding:
    """硅基流动 Embedding 服务 - 支持 BAAI/bge-m3 模型"""

    def __init__(self, state_file: str = "data/bm25_state.json"):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = os.getenv("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3")
        
        self.k1 = 1.5
        self.b = 0.75
        
        self._vocab = {}
        self._vocab_counter = 0
        
        self._doc_freq = Counter()
        self._total_docs = 0
        self._avg_doc_len = 0
        self._total_tokens = 0
        
        self._max_retries = 3
        self._retry_delay = 5.0
        
        self._state_file = state_file
        self._last_sync_time = 0
        
        self._load_state()

    def _load_state(self):
        """从 JSON 文件加载 BM25 状态"""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self._vocab = state.get("vocab", {})
                    self._vocab_counter = state.get("vocab_counter", 0)
                    self._doc_freq = Counter(state.get("doc_freq", {}))
                    self._total_docs = state.get("total_docs", 0)
                    self._avg_doc_len = state.get("avg_doc_len", 0)
                    self._total_tokens = state.get("total_tokens", 0)
                    self._last_sync_time = state.get("last_sync_time", 0)
                print(f"已加载 BM25 状态: {self._total_docs} 文档, {len(self._vocab)} 词汇")
            except Exception as e:
                print(f"加载 BM25 状态失败: {e}")

    def save_state(self):
        """保存 BM25 状态到 JSON 文件"""
        state = {
            "vocab": self._vocab,
            "vocab_counter": self._vocab_counter,
            "doc_freq": dict(self._doc_freq),
            "total_docs": self._total_docs,
            "avg_doc_len": self._avg_doc_len,
            "total_tokens": self._total_tokens,
            "last_sync_time": self._last_sync_time
        }
        state_dir = os.path.dirname(self._state_file)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir)
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"已保存 BM25 状态: {self._total_docs} 文档, {len(self._vocab)} 词汇")

    def get_last_sync_time(self) -> int:
        """获取上次同步时间"""
        return self._last_sync_time

    def set_last_sync_time(self, timestamp: int):
        """设置上次同步时间"""
        self._last_sync_time = timestamp

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        调用硅基流动 API 生成密集向量
        :param texts: 待转换的文本列表
        :return: 向量列表
        """
        if not texts:
            return []
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings", 
                    headers=headers, 
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return [item["embedding"] for item in result["data"]]
            except requests.exceptions.RequestException as e:
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise Exception(f"硅基流动 API 调用失败（重试 {self._max_retries} 次后）: {str(e)}")

    def tokenize(self, text: str) -> list[str]:
        """
        jieba 分词器 - 支持中英文混合和数字，过滤停用词
        :param text: 输入文本
        :return: 分词结果
        """
        if not text:
            return []
        
        text = text.lower()
        
        tokens = []
        words=jieba.lcut(text)
        for word in words:
            word = word.strip()
            if word and word not in STOPWORDS and len(word) > 1:
                tokens.append(word)
        
        return tokens

    def fit_corpus(self, texts: list[str], incremental: bool = False):
        """
        拟合语料库，计算 IDF 和平均文档长度
        :param texts: 文档列表
        :param incremental: 是否增量更新
        """
        if incremental:
            new_docs = len(texts)
            new_total_len = 0
            
            for text in texts:
                tokens = self.tokenize(text)
                new_total_len += len(tokens)
                
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    self._doc_freq[token] += 1
                    
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
            
            self._total_tokens += new_total_len
            self._total_docs += new_docs
            self._avg_doc_len = self._total_tokens / self._total_docs if self._total_docs > 0 else 1
        else:
            self._total_docs = len(texts)
            self._total_tokens = 0
            
            for text in texts:
                tokens = self.tokenize(text)
                self._total_tokens += len(tokens)
                
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    self._doc_freq[token] += 1
                    
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
            
            self._avg_doc_len = self._total_tokens / self._total_docs if self._total_docs > 0 else 1

    def get_sparse_embedding(self, text: str) -> dict:
        """
        生成 BM25 稀疏向量
        :param text: 输入文本
        :return: 稀疏向量 {index: value, ...}
        """
        tokens = self.tokenize(text)
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
            
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
            score = idf * numerator / denominator
            
            if score > 0:
                sparse_vector[idx] = float(score)
        
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """
        批量生成 BM25 稀疏向量
        :param texts: 文本列表
        :return: 稀疏向量列表
        """
        return [self.get_sparse_embedding(text) for text in texts]

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """
        同时生成密集向量和稀疏向量
        :param texts: 文本列表
        :return: (密集向量列表, 稀疏向量列表)
        """
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings
