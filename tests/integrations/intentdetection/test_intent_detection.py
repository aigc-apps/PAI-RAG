from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.llms.pai.llm_config import DashScopeLlmConfig
from pai_rag.integrations.router.pai.pai_router import PaiIntentRouter, IntentConfig

fc_llm_config = DashScopeLlmConfig(model="qwen-max")
fc_llm = PaiLlm(fc_llm_config)

intents = {
    "rag": "关于一些通用的信息检索，比如搜索旅游攻略、搜索美食攻略、搜索注意事项等信息。",
    "tool": "实时性的信息查询，比如查询航班信息、查询高铁信息、查询天气等时效性很强的信息。",
}


def test_single_detectors():
    intent_config = IntentConfig(descriptions=intents)
    intent_detector = PaiIntentRouter(intent_config=intent_config, llm=fc_llm)
    query_1 = "去上海玩的攻略有什么？"
    intent_1 = intent_detector.select(str_or_query_bundle=query_1)
    assert intent_1 == "rag"

    query_1 = "8月10号从北京出发去上海，机票价格有哪些？"
    intent_1 = intent_detector.select(str_or_query_bundle=query_1)
    assert intent_1 == "tool"
