import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.tool.faq_config_service import FAQConfigService, get_default_faq_config
from db.models.faq_config import FAQConfigCreate
from db.models.chatbot import ChatBotEntity


class TestGetDefaultFaqConfig:
    def test_returns_dict_with_required_keys(self):
        config = get_default_faq_config()
        assert config["active"] is True
        assert "similarity_threshold" in config
        assert "embedding_model" in config
        assert config["kb_id"] is None


class TestFAQConfigService:
    @pytest.fixture
    def service(self, mock_session):
        return FAQConfigService(session=mock_session)

    async def test_get_or_create_existing(self, service, mock_session):
        chatbot = MagicMock(spec=ChatBotEntity)
        chatbot.id = "cb1"
        chatbot.faq_config = {"active": True, "similarity_threshold": 0.9, "embedding_model": "bge-m3"}
        result = await service.get_or_create_faq_config(chatbot)
        assert isinstance(result, FAQConfigCreate)

    async def test_get_or_create_new(self, service, mock_session):
        chatbot = MagicMock(spec=ChatBotEntity)
        chatbot.id = "cb1"
        chatbot.faq_config = None
        mock_session.refresh = AsyncMock()
        result = await service.get_or_create_faq_config(chatbot)
        assert isinstance(result, FAQConfigCreate)
        mock_session.add.assert_called()

    async def test_update_faq_config(self, service, mock_session):
        chatbot = MagicMock()
        chatbot.id = "cb1"
        chatbot.faq_config = {"active": True, "similarity_threshold": 0.9}
        mock_session.refresh = AsyncMock(side_effect=lambda c: setattr(c, 'faq_config', {"active": False, "similarity_threshold": 0.85}))
        update_data = FAQConfigCreate(active=False, similarity_threshold=0.85)
        result = await service.update_faq_config(chatbot, update_data)
        mock_session.add.assert_called()
