import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from db.models.vectordb import VectorDbConfig
from service.knowledgebase.vectordb_service import VectordbService
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


def _clear_hologres_env(monkeypatch):
    # Env declares hologres as the vector db type (as in a real deployment) but the
    # credentials are meant to be supplied through the UI rather than the environment.
    monkeypatch.setenv("VECTOR_DB_TYPE", "hologres")
    monkeypatch.setenv("HOLOGRES_HOST", "env-host")
    for key in (
        "HOLOGRES_USER",
        "PAIRAG_RAG__INDEX__VECTOR_STORE__username",
        "PAIRAG_RAG__INDEX__VECTOR_STORE__type",
    ):
        monkeypatch.delenv(key, raising=False)


class TestVectordbService:
    @pytest.fixture
    def service(self, mock_session):
        return VectordbService(session=mock_session)

    async def test_create_hologres_config_with_partial_env(
        self, service, mock_session, monkeypatch
    ):
        """A fully-specified hologres config saved through the UI must not be
        blocked by a partially-configured environment (regression: the env-based
        fallback used to assert ``Hologres user不能为空`` and fail the save)."""
        _clear_hologres_env(monkeypatch)
        mock_session.exec.return_value = make_mock_result(first_value=None)  # no existing config

        config = VectorDbConfig(
            type="hologres",
            config={
                "host": "user-host",
                "port": "80",
                "user": "user-provided-user",
                "password": "user-provided-pass",
                "database": "user-db",
            },
        )

        saved = await service.create_or_update_vectordb_config(config, tenant_id=TENANT)

        assert saved.type == "hologres"
        assert saved.config["user"] == "user-provided-user"

    async def test_get_config_with_partial_env_falls_back_to_local(
        self, service, mock_session, monkeypatch
    ):
        """Loading the page when no config exists yet and the environment is only
        partially configured must not raise; it should fall back to local."""
        _clear_hologres_env(monkeypatch)
        mock_session.exec.return_value = make_mock_result(first_value=None)

        vector_config = await service.get_vectordb_config(tenant_id=TENANT)

        assert vector_config.type == "local"
