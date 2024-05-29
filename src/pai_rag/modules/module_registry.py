from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
import pai_rag.modules as modules

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
    "ToolModule": "tool"
}


class ModuleRegistry:
    def __init__(self):
        self._mod_instance_map = {}
        self._mod_cls_map = {}
        self._mod_deps_map = {}
        self._mod_deps_map_inverted = {}

        for m_name in modules.ALL_MODULES:
            m_cls = getattr(modules, m_name)
            self._mod_cls_map[m_name] = m_cls()

            deps = m_cls.get_dependencies()
            self._mod_deps_map[m_name] = deps

            for dep in deps:
                if dep not in self._mod_deps_map_inverted:
                    self._mod_deps_map_inverted[dep] = []
                self._mod_deps_map_inverted[dep].append(m_name)

    def get_module(self, module_key: str):
        return self._mod_instance_map[module_key]

    def init_modules(self, config):
        mod_stack = []
        mods_inited = []
        mod_ref_count = {}
        for mod, deps in self._mod_deps_map.items():
            ref_count = len(deps)
            mod_ref_count[mod] = ref_count
            if ref_count == 0:
                mod_stack.append(mod)

        while mod_stack:
            mod = mod_stack.pop()
            mod_obj = self._init_mod(mod, config)
            mods_inited.append(mod)
            self._mod_instance_map[mod] = mod_obj

            # update module ref count that depends on on
            ref_mods = self._mod_deps_map_inverted.get(mod, [])
            for ref_mod in ref_mods:
                mod_ref_count[ref_mod] -= 1
                if mod_ref_count[ref_mod] == 0:
                    mod_stack.append(ref_mod)

        if len(mods_inited) != len(modules.ALL_MODULES):
            # dependency circular error!
            raise ValueError(
                f"Circular dependency detected. Please check module dependency configuration. Module initialized: {mods_inited}. Module ref count: {mod_ref_count}"
            )
        print(f"RAG modules init successfully. {mods_inited}")
        return

    def _init_mod(self, mod_name, config):
        mod_config_key = MODULE_CONFIG_KEY_MAP[mod_name]
        mod_deps = self._mod_deps_map[mod_name]
        mod_cls = self._mod_cls_map[mod_name]

        params = {MODULE_PARAM_CONFIG: config[mod_config_key]}
        for dep in mod_deps:
            params[dep] = self.get_module(dep)
        return mod_cls.get_or_create(params)


module_registry = ModuleRegistry()
