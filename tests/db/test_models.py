import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from datetime import datetime


class TestThreadEntity:
    def test_defaults(self):
        from db.models.thread import ThreadEntity
        t = ThreadEntity()
        assert t.id is not None
        assert t.user_id == "PAI-RAG Assistant"
        assert isinstance(t.created_at, datetime)
        assert isinstance(t.updated_at, datetime)

    def test_serialize_dt(self):
        from db.models.thread import ThreadEntity
        t = ThreadEntity()
        result = t.serialize_dt(datetime(2024, 1, 1, 12, 0, 0), None)
        assert result == "2024-01-01T12:00:00Z"


class TestMessageEntity:
    def test_defaults(self):
        from db.models.message import MessageEntity
        m = MessageEntity(thread_id="t1", role="user")
        assert m.id is not None
        assert m.content == []
        assert m.attachments == []
        assert m.token_usage is None

    def test_serialize_dt(self):
        from db.models.message import MessageEntity
        m = MessageEntity(thread_id="t1", role="user")
        result = m.serialize_dt(datetime(2024, 6, 15, 8, 30, 0), None)
        assert result == "2024-06-15T08:30:00Z"


class TestLlmModelEntity:
    def test_defaults(self):
        from db.models.llm import LlmModelEntity
        # Provide id explicitly since default_factory uses validated_data
        llm = LlmModelEntity(id="test-id", model_id="qwen-plus", model="qwen-plus")
        assert llm.context_window == 110000
        assert llm.temperature == 0.1
        assert llm.enabled is True
        assert llm.vision_support is False
        assert llm.enable_thinking is False
        assert llm.max_tokens == 8000


class TestKbFileEntity:
    def test_defaults(self):
        from db.models.knowledgebase.file import KbFileEntity
        f = KbFileEntity(kb_id="kb1", file_name="test.pdf")
        assert f.id is not None
        assert f.file_version == 0
        assert f.active is True
        assert f.file_metadata == {}

    def test_get_file_content(self):
        from db.models.knowledgebase.file import KbFileEntity
        f = KbFileEntity(kb_id="kb1", file_name="test.pdf", file_content="hello")
        content = f.get_file_content()
        assert "test.pdf" in content
        assert "hello" in content

    def test_get_file_content_truncation_note(self):
        from db.models.knowledgebase.file import KbFileEntity
        f = KbFileEntity(kb_id="kb1", file_name="test.pdf", file_content="x" * 100, file_content_length=2000)
        content = f.get_file_content()
        assert "truncated" in content

    def test_serialize_dt(self):
        from db.models.knowledgebase.file import KbFileEntity
        f = KbFileEntity(kb_id="kb1", file_name="test.pdf")
        result = f.serialize_dt(datetime(2024, 3, 15), None)
        assert result == "2024-03-15T00:00:00Z"


class TestKbFileTaskEntity:
    def test_defaults(self):
        from db.models.knowledgebase.file_task import KbFileTaskEntity
        task = KbFileTaskEntity(kb_id="kb1", file_id="f1")
        assert task.id is not None
        assert task.file_part == 0
        assert task.file_version == 0
        assert task.status == "pending"


class TestTraceModelEntity:
    def test_is_enabled_true(self):
        from db.models.trace import TraceModelEntity
        t = TraceModelEntity(
            id="tr1",
            endpoint="http://trace.example.com",
            token="tok",
            service_name="svc",
            enabled=True,
        )
        assert t.is_enabled()

    def test_is_enabled_false_missing_service(self):
        from db.models.trace import TraceModelEntity
        t = TraceModelEntity(
            id="tr1",
            endpoint="http://trace.example.com",
            token="tok",
            service_name=None,
            enabled=True,
        )
        assert not t.is_enabled()

    def test_is_enabled_false_disabled(self):
        from db.models.trace import TraceModelEntity
        t = TraceModelEntity(
            id="tr1",
            endpoint="http://trace.example.com",
            token="tok",
            service_name="svc",
            enabled=False,
        )
        assert not t.is_enabled()


class TestGuardrailConfigEntity:
    def test_defaults(self):
        from db.models.guardrail import GuardrailConfigEntity
        g = GuardrailConfigEntity(id="g1", region_name="cn-hangzhou")
        assert g.region_id == "cn-hangzhou"


class TestWebSearchConfigEntity:
    def test_defaults(self):
        from db.models.websearch import WebSearchConfigEntity
        w = WebSearchConfigEntity(id="w1")
        assert w.search_count == 10


class TestCodeSandboxConfigEntity:
    def test_defaults(self):
        from db.models.code_sandbox import CodeSandboxConfigEntity
        c = CodeSandboxConfigEntity(id="cs1")
        assert c.type == "aliyun-fc"
        assert c.enabled is False
        assert c.timeout_default == 50


class TestMcpServerEntity:
    def test_defaults(self):
        from db.models.mcp import McpServerEntity
        m = McpServerEntity(id="mcp1", name="test", url="http://mcp.example.com")
        assert m.type == "sse"
        assert m.enabled is True
        assert m.need_token is False


class TestRetrievalConfig:
    def test_defaults(self):
        from db.models.knowledgebase.knowledgebase import RetrievalConfig
        rc = RetrievalConfig()
        assert rc.enable_rerank is False
        assert rc.similarity_threshold == 0


class TestKnowledgebaseCreate:
    def test_name_validation_empty(self):
        from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate
        with pytest.raises(ValueError, match="cannot be empty"):
            KnowledgebaseCreate(name="")

    def test_name_validation_too_long(self):
        from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate
        with pytest.raises(ValueError, match="exceed 100"):
            KnowledgebaseCreate(name="a" * 101)

    def test_name_validation_special_chars(self):
        from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate
        with pytest.raises(ValueError, match="letters, numbers"):
            KnowledgebaseCreate(name="invalid name!")

    def test_valid_name(self):
        from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate
        kb = KnowledgebaseCreate(name="valid-name_123", embedding_model="bge-m3")
        assert kb.name == "valid-name_123"
