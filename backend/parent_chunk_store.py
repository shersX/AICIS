"""父级分块文档存储（用于 Auto-merging Retriever）"""
from pathlib import Path
from typing import Dict, List


class ParentChunkStore:
    """基于本地 JSON 的父级分块存储。"""

    def __init__(self, store_path: Path | None = None):
        base_dir = Path(__file__).resolve().parent
        self.store_path = store_path or (base_dir.parent / "data" / "parent_chunks.json")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, dict]:
        if not self.store_path.exists():
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = f.read()
            import json
            parsed = json.loads(data) if data.strip() else {}
            if isinstance(parsed, dict):
                return parsed
            return {}
        except Exception:
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        import json
        tmp_path = self.store_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp_path.replace(self.store_path)

    def upsert_documents(self, docs: List[dict]) -> int:
        """写入/更新父级分块，返回写入条数。"""
        return 0

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []
        store = self._load()
        return [store[item] for item in chunk_ids if item in store]

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除父级分块，返回删除条数。"""
        return 0
