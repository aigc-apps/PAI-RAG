import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.model.llm_service import LlmService
from db.models.llm import LlmModelEntity, LlmModelCreate
from tests.service.conftest import make_mock_result


TENANT = "test_tenant"


class TestLlmService:
    @pytest.fixture
    def service(self, mock_session):
        return LlmService(session=mock_session)

    async def test_get_llm_found(self, service, mock_session):
        llm = LlmModelEntity(id="l1", model_id="qwen", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=llm)
        result = await service.get_llm("l1", TENANT)
        assert result == llm

    async def test_get_llm_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_llm("nonexistent", TENANT)
        assert result is None

    async def test_get_llm_by_model_id(self, service, mock_session):
        llm = LlmModelEntity(id="l1", model_id="qwen-plus", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=llm)
        result = await service.get_llm_by_model_id("qwen-plus", TENANT)
        assert result.model_id == "qwen-plus"

    async def test_list_llms(self, service, mock_session):
        llms = [LlmModelEntity(id=f"l{i}", model_id=f"m{i}", tenant_id=TENANT) for i in range(3)]
        # First exec for count, second for paginated results
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=3)
            else:
                return make_mock_result(all_values=llms)

        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_llms(TENANT)
        assert result.total == 3
        assert len(result.items) == 3

    @patch("service.model.llm_service.encrypt_key", return_value="encrypted")
    async def test_create_llm(self, mock_encrypt, service, mock_session):
        data = LlmModelCreate(
            model_id="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            api_key="test-key",
            provider_name="dashscope",
        )
        mock_session.refresh = AsyncMock()
        result = await service.create_llm(data, TENANT)
        assert result is not None
        mock_session.add.assert_called_once()

    @patch("service.model.llm_service.encrypt_key", return_value="encrypted")
    async def test_create_llm_unsupported_provider(self, mock_encrypt, service, mock_session):
        data = LlmModelCreate(
            model_id="m1",
            base_url="http://example.com",
            model="m1",
            api_key="key",
            provider_name="totally_unsupported_provider_xyz",
        )
        with pytest.raises(ValueError, match="not supported"):
            await service.create_llm(data, TENANT)

    async def test_update_llm_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        data = LlmModelCreate(model_id="m1", model="m1")
        with pytest.raises(ValueError, match="does not exist"):
            await service.update_llm("bad_id", data, TENANT)

    @patch("service.model.llm_service.encrypt_key", return_value="encrypted")
    async def test_update_llm_success(self, mock_encrypt, service, mock_session):
        llm = LlmModelEntity(id="l1", model_id="old", model="old", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=llm)
        mock_session.refresh = AsyncMock()
        data = LlmModelCreate(model_id="new", model="new", temperature=0.5)
        result = await service.update_llm("l1", data, TENANT)
        assert result.model_id == "new"
        assert result.temperature == 0.5

    async def test_delete_llm_success(self, service, mock_session):
        llm = LlmModelEntity(id="l1", model_id="m1", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=llm)
        await service.delete_llm("l1", TENANT)
        mock_session.delete.assert_called_once_with(llm)

    async def test_delete_llm_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError, match="does not exist"):
            await service.delete_llm("bad_id", TENANT)

    async def test_get_all_llms(self, service, mock_session):
        llms = [LlmModelEntity(id=f"l{i}", model_id=f"m{i}", tenant_id=TENANT) for i in range(2)]
        mock_session.exec.return_value = make_mock_result(all_values=llms)
        result = await service.get_all_llms(TENANT)
        assert len(result) == 2

    async def test_get_provider_names(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(all_values=["dashscope", "openai_like"])
        result = await service.get_provider_names(TENANT)
        assert "dashscope" in result
        assert "openai_like" in result

    async def test_get_provider_names_adds_openai_like_default(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(all_values=["dashscope"])
        result = await service.get_provider_names(TENANT)
        assert "openai_like" in result
