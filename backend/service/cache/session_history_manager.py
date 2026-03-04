"""
会话历史管理模块

负责在 Redis 中保存和恢复用户与特定模型的对话历史。
使用 user_id + model + session_id 作为唯一标识。
保留最近 5 轮对话（用户消息 + 助手回复），7 天过期。
"""

import json
from typing import List
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from service.cache.redis_cache import cache_manager


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

    # 配置常量
    MAX_HISTORY_ROUNDS = 5  # 保存最近 5 轮对话
    TTL_SECONDS = 7 * 24 * 60 * 60  # 7 天过期

    def __init__(self):
        self.cache = cache_manager.get_cache()

    async def save_messages(
        self,
        user_id: str,
        session_id: str,
        user_message: ChatCompletionMessageParam,
        assistant_message: ChatCompletionMessageParam,
    ) -> None:
        """
        保存一轮对话（用户消息 + 助手回复）到 Redis

        Args:
            user_id: 用户ID
            model: 模型名称
            session_id: 会话ID
            user_message: 用户消息
            assistant_message: 助手回复消息
        """
        if not user_id or not session_id:
            logger.debug(
                f"user_id={user_id} or session_id={session_id} is empty, "
                "skip saving session history"
            )
            return

        try:
            key = session_history_key(user_id, session_id)

            # 获取现有历史
            existing_history = await self._get_history_list(key)

            # 添加新的一轮对话（用户消息 + 助手回复）
            existing_history.append(user_message)
            existing_history.append(assistant_message)

            # 保持最近的 N 轮对话（每轮包含用户 + 助手两条消息）
            max_messages = self.MAX_HISTORY_ROUNDS * 2
            if len(existing_history) > max_messages:
                existing_history = existing_history[-max_messages:]

            # 序列化并保存到 Redis
            history_json = json.dumps(existing_history, ensure_ascii=False)
            await self.cache.set(key, history_json, ttl=self.TTL_SECONDS)

            logger.info(
                f"Saved session history: user={user_id}, "
                f"session={session_id}, total_messages={len(existing_history)}"
            )
        except Exception as e:
            logger.error(f"Failed to save session history: {e}", exc_info=True)

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
            key = session_history_key(user_id, model, session_id)
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
