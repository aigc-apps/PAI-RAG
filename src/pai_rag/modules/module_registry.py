import hashlib
import json
from typing import Dict, Any
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
import pai_rag.modules as modules
import logging

MODULE_CONFIG_KEY_MAP = {
    "IndexModule": "index",
    "EmbeddingModule": "embedding",
    "LlmModule": "llm",
    "NodeParserModule": "node_parser",
    "RetrieverModule": "retriever",
    "PostprocessorModule": "postprocessor",
    "SynthesizerModule": "synthesizer",
    "QueryEngineModule": "query_engine",
    "ChatStoreModule": "chat_store",
    "ChatEngineFactoryModule": "chat_engine",
    "LlmChatEngineFactoryModule": "llm_chat_engine",
    "DataReaderFactoryModule": "data_reader",
    "AgentModule": "agent",
    "ToolModule": "tool",
    "DataLoaderModule": "data_loader",
    "OssCacheModule": "cache",
    "BM25IndexModule": "bm25",
}


logger = logging.getLogger(__name__)


class ModuleRegistry:
    def __init__(self):
        self._cache_by_config = {}
        self._mod_cls_map = {}
        self._mod_deps_map = {}
        self._mod_deps_map_inverted = {}

        self._mod_instance_map = {}

        for m_name in modules.ALL_MODULES:
            self._mod_instance_map[m_name] = {}

            m_cls = getattr(modules, m_name)
            self._mod_cls_map[m_name] = m_cls()

            deps = m_cls.get_dependencies()
            self._mod_deps_map[m_name] = deps

            for dep in deps:
                if dep not in self._mod_deps_map_inverted:
                    self._mod_deps_map_inverted[dep] = []
                self._mod_deps_map_inverted[dep].append(m_name)

    def _get_param_hash(self, params: Dict[str, Any]):
        repr_str = json.dumps(params, default=repr, sort_keys=True).encode("utf-8")
        return hashlib.sha256(repr_str).hexdigest()

    def get_module_with_config(self, module_key, config):
        key = repr(config)
        if key in self._cache_by_config and module_key in self._cache_by_config[key]:
            return self._cache_by_config[key][module_key]

        else:
            mod = self._create_mod_lazily(module_key, config)
            if key not in self._cache_by_config:
                self._cache_by_config[key] = {}

            self._cache_by_config[key][module_key] = mod
            return mod

    def init_modules(self, config):
        key = repr(config)

        mod_cache = {}
        mod_stack = []
        mod_ref_count = {}
        for mod, deps in self._mod_deps_map.items():
            ref_count = len(deps)
            mod_ref_count[mod] = ref_count
            if ref_count == 0:
                mod_stack.append(mod)

        while mod_stack:
            mod = mod_stack.pop()
            mod_obj = self._create_mod_lazily(mod, config, mod_cache)
            mod_cache[mod] = mod_obj
            if key not in self._cache_by_config:
                self._cache_by_config[key] = {}
            self._cache_by_config[key][mod] = mod_obj

            # update module ref count that depends on on
            ref_mods = self._mod_deps_map_inverted.get(mod, [])
            for ref_mod in ref_mods:
                mod_ref_count[ref_mod] -= 1
                if mod_ref_count[ref_mod] == 0:
                    mod_stack.append(ref_mod)

        if len(mod_cache) != len(modules.ALL_MODULES):
            # dependency circular error!
            raise ValueError(
                f"Circular dependency detected. Please check module dependency configuration. Module initialized: {mod_cache}. Module ref count: {mod_ref_count}"
            )
        logger.info(f"RAG modules init successfully. {mod_cache.keys()}")
        return

    def _create_mod_lazily(self, mod_name, config, mod_cache=None):
        if mod_cache and mod_name in mod_cache:
            return mod_cache[mod_name]

        logger.info(f"Get module {mod_name}.")

        mod_config_key = MODULE_CONFIG_KEY_MAP[mod_name]
        mod_deps = self._mod_deps_map[mod_name]
        mod_cls = self._mod_cls_map[mod_name]

        params = {MODULE_PARAM_CONFIG: config.get(mod_config_key, None)}
        for dep in mod_deps:
            params[dep] = self._create_mod_lazily(dep, config, mod_cache)

        instance_key = self._get_param_hash(params)
        if mod_name == "IndexModule":
            print(instance_key, params)

        if instance_key not in self._mod_instance_map[mod_name]:
            logger.info(f"Creating new instance for module {mod_name} {instance_key}.")
            self._mod_instance_map[mod_name][instance_key] = mod_cls.get_or_create(
                params
            )
        return self._mod_instance_map[mod_name][instance_key]


module_registry = ModuleRegistry()
