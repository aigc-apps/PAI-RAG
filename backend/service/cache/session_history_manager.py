"""
会话历史管理模块

负责在 Redis 中保存和恢复用户与特定模型的对话历史。
使用 user_id + model + session_id 作为唯一标识。
保留最近 5 轮对话（用户消息 + 助手回复），7 天过期。
"""

import json
import re
from typing import List
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from service.cache.redis_cache import cache_manager

# Strip the "[System Time: ...]\n" prefix the agent injects into the live user
# message so it never gets persisted into session history.
_SYS_TIME_PREFIX_RE = re.compile(r"^\[System Time:[^\]]*\]\n")


def _clean_user_message(msg):
    """Return a copy of a user message with the injected System Time prefix removed."""
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        cleaned = _SYS_TIME_PREFIX_RE.sub("", msg["content"])
        if cleaned != msg["content"]:
            return {**msg, "content": cleaned}
    return msg


def session_history_key(user_id: str, session_id: str) -> str:
    """
    生成会话历史的 Redis key

    Args:
        user_id: 用户ID
        model: 模型名称
        session_id: 会话ID

    Returns:
        Redis key 字符串
    """
    return f"session:uid:{user_id}:sid:{session_id}"


class SessionHistoryManager:
    """会话历史管理器"""

    MAX_HISTORY_ROUNDS = 5
    MAX_HISTORY_MESSAGES = 40
    MAX_OLD_TOOL_RESULT_CHARS = 500
    TOOL_RESULT_TRUNCATED_MARKER = "\n...[truncated]"
    TTL_SECONDS = 7 * 24 * 60 * 60

    def __init__(self):
        self.cache = cache_manager.get_cache()

    async def save_messages(
        self,
        user_id: str,
        session_id: str,
        user_message: ChatCompletionMessageParam,
        assistant_message: ChatCompletionMessageParam,
        tool_messages: List[ChatCompletionMessageParam] | None = None,
    ) -> None:
        """
        保存一轮对话到 Redis。

        一轮完整的对话包含:
        user_message → [assistant(tool_calls) → tool(result)]* → assistant(final_text)

        Args:
            user_id: 用户ID
            session_id: 会话ID
            user_message: 用户消息
            assistant_message: 助手最终回复消息
            tool_messages: 中间的 tool 交互消息列表
                           (assistant with tool_calls + tool results)
        """
        if not user_id or not session_id:
            logger.debug(
                f"user_id={user_id} or session_id={session_id} is empty, "
                "skip saving session history"
            )
            return

        try:
            key = session_history_key(user_id, session_id)

            existing_history = await self._get_history_list(key)

            # Persist ONLY the clean Q/A pair: the user question (without the
            # injected System Time prefix) and the assistant's final answer.
            # Intermediate tool_calls/tool-result steps (`tool_messages`) are this
            # turn's ephemeral working state and are NOT stored — replaying them on
            # later turns bloats context, confuses the model, and (on retry)
            # produces duplicated/garbled history.
            existing_history.append(_clean_user_message(user_message))
            existing_history.append(assistant_message)

            existing_history = self._trim_to_rounds(existing_history, self.MAX_HISTORY_ROUNDS)

            history_json = json.dumps(existing_history, ensure_ascii=False)
            await self.cache.set(key, history_json, ttl=self.TTL_SECONDS)

            logger.info(
                f"Saved session history: user={user_id}, "
                f"session={session_id}, total_messages={len(existing_history)}, "
                f"tool_messages={len(tool_messages) if tool_messages else 0}"
            )
        except Exception as e:
            logger.error(f"Failed to save session history: {e}", exc_info=True)

    @classmethod
    def _trim_to_rounds(
        cls,
        messages: List[ChatCompletionMessageParam],
        max_rounds: int,
    ) -> List[ChatCompletionMessageParam]:
        user_indices = [
            i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if len(user_indices) > max_rounds:
            cut_from = user_indices[-max_rounds]
            messages = messages[cut_from:]

        while len(messages) > cls.MAX_HISTORY_MESSAGES:
            user_indices = [
                i for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            if len(user_indices) <= 1:
                break
            messages = messages[user_indices[1]:]
            logger.info(
                f"History over {cls.MAX_HISTORY_MESSAGES} messages, "
                f"reduced to {len(user_indices) - 1} rounds ({len(messages)} msgs)"
            )

        cls._truncate_old_tool_results(messages)
        return messages

    @classmethod
    def _truncate_old_tool_results(
        cls,
        messages: List[ChatCompletionMessageParam],
    ) -> None:
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                last_user_idx = i
                break

        for i in range(last_user_idx):
            msg = messages[i]
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > cls.MAX_OLD_TOOL_RESULT_CHARS:
                msg["content"] = content[:cls.MAX_OLD_TOOL_RESULT_CHARS] + cls.TOOL_RESULT_TRUNCATED_MARKER

    async def get_history_messages(
        self,
        user_id: str,
        session_id: str,
    ) -> List[ChatCompletionMessageParam]:
        """
        从 Redis 获取历史消息

        Args:
            user_id: 用户ID
            model: 模型名称
            session_id: 会话ID

        Returns:
            历史消息列表，如果没有历史则返回空列表
        """
        if not user_id or not session_id:
            logger.debug(
                f"user_id={user_id} or session_id={session_id} is empty, "
                "skip loading session history"
            )
            return []

        try:
            key = session_history_key(user_id, session_id)
            history = await self._get_history_list(key)

            if history:
                logger.info(
                    f"Loaded session history: user={user_id},"
                    f"session={session_id}, messages_count={len(history)}"
                )
            else:
                logger.debug(
                    f"No session history found: user={user_id}, "
                    f"session={session_id}"
                )
                history = []
            return history
        except Exception as e:
            logger.error(f"Failed to load session history: {e}", exc_info=True)
            return []

    async def clear_history(
        self,
        user_id: str,
        model: str,
        session_id: str
    ) -> None:
        """
        清除指定会话的历史

        Args:
            user_id: 用户ID
            model: 模型名称
            session_id: 会话ID
        """
        if not user_id or not session_id:
            return

        try:
            key = session_history_key(user_id, session_id)
            await self.cache.delete(key)
            logger.info(
                f"Cleared session history: user={user_id}, model={model}, "
                f"session={session_id}"
            )
        except Exception as e:
            logger.error(f"Failed to clear session history: {e}", exc_info=True)

    async def _get_history_list(self, key: str) -> List[ChatCompletionMessageParam]:
        """
        从 Redis 获取并解析历史消息列表

        Args:
            key: Redis key

        Returns:
            解析后的消息列表
        """
        history_json = await self.cache.get(key)
        if history_json:
            try:
                return json.loads(history_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode session history JSON: {e}")
                return []
        return []


# 全局单例
session_history_manager = SessionHistoryManager()
