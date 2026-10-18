import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()


class HistoryMongoTool:
    """MongoDB 会话历史读写封装。"""

    def __init__(self):
        """建立数据库连接，并确保按会话查询的索引存在。"""
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            # 适配“按会话取最近消息”的排序与过滤
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise


def clear_history(session_id: str) -> int:
    """删除指定会话的全部记录，返回删除条数。"""
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}")
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None
) -> str:
    """写入会话记录；传入 message_id 时更新对应记录。

    :return: 新增记录的 ObjectId 字符串，或更新记录的原 message_id
    """
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "ts": ts
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document}
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """批量回填关联商品名，仅覆盖 item_names 为空的记录。"""
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},
                "$or": [
                    {"item_names": {"$exists": False}},
                    {"item_names": []},
                    {"item_names": None}
                ]
            },
            {"$set": {"item_names": item_names}}
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logging.error(f"Error updating history item_names: {e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """返回指定会话最近 limit 条记录，按时间正序排列。"""
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}

        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        messages = list(cursor)

        return messages
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []


_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """返回进程内共享的 HistoryMongoTool 实例，按需创建。"""
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    # 预热失败不阻断导入，首次调用时由 get_history_mongo_tool 重试
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")

if __name__ == "__main__":
    sid = "000015_hybrid"
    save_chat_message(sid, "user", "你好 (Hybrid)")
    save_chat_message(sid, "assistant", "你好！我是基于原生 Mongo + LangChain 对象的助手。")
    save_chat_message(sid, "user", "这个万用表怎么换电池？", item_names=["混合万用表"])

    print("--- 查询 LangChain 对象记录 ---")
    messages = get_recent_messages(sid, limit=5)
    print(f"查询到的记录数: {len(messages)}")
    for m in messages:
        print(f" {m}  ")
