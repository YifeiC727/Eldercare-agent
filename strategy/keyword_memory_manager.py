# Persistent keyword memory manager using TinyDB
from typing import List
from datetime import datetime, timedelta
from tinydb import TinyDB, Query

class KeywordMemoryManager:
    def __init__(self, db_path="keyword_memory.json", expire_days=7):
        # Initialize memory database and expiration policy
        self.db = TinyDB(db_path)
        self.expire_days = expire_days

    def add_keyword(self, keyword: str, source: str):
        # Store keyword and its source with a timestamp
        now = datetime.now().isoformat()
        self.db.insert({
            "keyword": keyword,
            "source": source,
            "timestamp": now
        })

    def get_recent_keywords(self) -> List[str]:
        # Retrieve non-expired keywords from DB
        now = datetime.now()
        valid_time = now - timedelta(days=self.expire_days)
        Keyword = Query()
        valid_records = self.db.search(Keyword.timestamp.test(
            lambda t: datetime.fromisoformat(t) >= valid_time
        ))
        return list({r["keyword"] for r in valid_records})

    def get_keyword_sources(self, keyword: str) -> List[str]:
        # Retrieve all text sources where a given keyword appeared
        Keyword = Query()
        return [r["source"] for r in self.db.search(Keyword.keyword == keyword)]

    def clear_expired(self):
        # Delete expired keyword entries
        now = datetime.now()
        valid_time = now - timedelta(days=self.expire_days)
        self.db.remove(Query().timestamp.test(
            lambda t: datetime.fromisoformat(t) < valid_time
        ))

    def clear_all(self):
        # Wipe the memory database
        self.db.truncate()