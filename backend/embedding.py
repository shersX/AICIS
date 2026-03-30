"""文本向量化服务 - 支持密集向量和稀疏向量（BM25）"""
import os
import re
import math
import time
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

class SiliconFlowEmbedding:
    """硅基流动 Embedding 服务 - 支持 BAAI/bge-m3 模型"""

    def __init__(self):
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
        
        self._max_retries = 3
        self._retry_delay = 5.0

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
        简单分词器 - 支持中英文混合
        :param text: 输入文本
        :return: 分词结果
        """
        if not text:
            return []
        
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

    def fit_corpus(self, texts: list[str]):
        """
        拟合语料库，计算 IDF 和平均文档长度
        :param texts: 文档列表
        """
        self._total_docs = len(texts)
        total_len = 0
        
        for text in texts:
            tokens = self.tokenize(text)
            total_len += len(tokens)
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freq[token] += 1
                
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_counter
                    self._vocab_counter += 1
        
        self._avg_doc_len = total_len / self._total_docs if self._total_docs > 0 else 1

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
