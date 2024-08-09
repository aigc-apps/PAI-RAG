import os
from dotenv import load_dotenv
import pytest

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding

from pai_rag.integrations.nodes.raptor_nodes_enhance import RaptorProcessor

load_dotenv()


@pytest.mark.skipif(
    os.getenv("DASHSCOPE_API_KEY") is None, reason="no llm api key provided"
)
async def test_enhance_nodes():
    load_dotenv(verbose=True)
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DEFAULT_EMBED_BATCH_SIZE = 10

    # set embedding
    llm = DashScope(model_name="qwen-turbo", temperature=0.1)
    embed_model = DashScopeEmbedding(
        api_key=DASHSCOPE_API_KEY, embed_batch_size=DEFAULT_EMBED_BATCH_SIZE
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    # create nodes
    nodes = []
    text_list = [
        "小红今天在家做了番茄炒蛋。",
        "小红在家做了一道菜，叫蚂蚁上树。",
        "小红在家照着菜谱做了一道红烧肉，只是酱油放多了。",
        "今天刮大风，天气很冷，小明躲在被窝里睡觉。",
        "今天天气很糟糕，零下2度，小明躺在床上看电视剧。",
        "今天下小雨，小明到附近的商场去购物。",
        "现在梅雨季节，天天下雨，小明趴在窗口看着外面潮湿的世界发呆。",
        "开始下雪了，小明在家穿着厚厚的毛衣，一边喝咖啡一遍读书。",
        "小红特别喜欢做菜，经常在家里联系各种菜系的做法。",
        "小红晚上在家煲了一碗鸽子汤，还做了小葱拌豆腐。",
        "小红喜欢看各种美食类节目，比如舌尖上的中国。",
        "小红爱吃川菜，经常托朋友从四川带正宗的川辣椒自己研究川菜的做法。",
    ]
    for i, text_i in enumerate(text_list):
        nodes.append(TextNode(text=text_i, id_=f"chunk_{i}"))

    # raptor init
    raptor = RaptorProcessor(
        tree_depth=2, max_clusters=50, threshold=0.1, embed_model=embed_model
    )

    # use transform directly
    nodes_with_embeddings = raptor(nodes)

    assert len(nodes_with_embeddings) - len(nodes) > 0

    # use transformation in async
    nodes_with_embeddings = await raptor.acall(nodes)

    assert len(nodes_with_embeddings) - len(nodes) > 0
