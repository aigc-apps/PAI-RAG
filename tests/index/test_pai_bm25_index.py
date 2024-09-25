from pai_rag.integrations.index.pai.local.local_bm25_index import LocalBm25IndexStore
from llama_index.core.schema import TextNode

texts = [
    "人工智能平台PAI是面向开发者和企业的云原生机器学习/深度学习工程平台，服务覆盖AI开发全链路，内置140+种优化算法，具备丰富的行业场景插件。",
    "面向大规模深度学习及融合智算场景的PaaS平台产品，支持公共云Serverless版、单租版以及混合云产品形态，一站式提供AI工程化全流程平台及软硬一体联合优化的异构融合算力。",
    "公共云Serverless版: Serverless平台产品，一键快速拉起AI计算任务，复杂异构系统自动运维，轻松管理。与云上的计算、存储、网络等各类产品无缝衔接。",
    "公共云单租版:云上建立客户专属集群，单个客户独享一套AI平台和运维服务。便捷运营管理，云产品互通，使用云上标准的计算、存储、网络服务。",
    "飞天混合云版:支持混合云标准架构，提供完整的计算、网络、存储、账号(ASCM)，标准SDK/OpenAPI，物理资源独立部署，支持服务商基于客户场景构建业务。",
    "模型开发: 在模型开发阶段，可通过PAI-Designer、PAI-DSW、PAI-QuickStart 三款工具来完成建模。",
    "模型训练: 在模型训练阶段，可通过PAI-DLC发起大规模的分布式训练任务；按照使用场景和算力类别，可以分为使用灵骏智算支持大模型的训练任务，和使用阿里云通用算力节点支持通用的训练任务。",
    "模型部署: 在模型部署阶段，PAI-EAS提供在线预测服务，PAI-Blade提供推理优化服务。",
]


def test_bm25():
    persist_path = "./tmp/bm25_test"

    bm25 = LocalBm25IndexStore(persist_path)

    nodes = [TextNode(id_=i, text=text) for i, text in enumerate(texts)]
    bm25.add_docs(nodes)

    nodes_with_score = bm25.query("模型开发")
    assert nodes_with_score[0].node.get_content() == texts[5]

    nodes_with_score = bm25.query("模型训练")
    assert nodes_with_score[0].node.get_content() == texts[6]
