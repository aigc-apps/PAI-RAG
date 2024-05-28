SIMPLE_PROMPTS = "参考内容如下：\n{context_str}\n作为个人知识答疑助手，请根据上述参考内容回答下面问题，答案中不允许包含编造内容。\n用户问题:\n{query_str}"
GENERAL_PROMPTS = '基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context_str}\n=====\n用户问题:\n{query_str}'
EXTRACT_URL_PROMPTS = "你是一位智能小助手，请根据下面我所提供的相关知识，对我提出的问题进行回答。回答的内容必须包括其定义、特征、应用领域以及相关网页链接等等内容，同时务必满足下方所提的要求！\n=====\n 知识库相关知识如下:\n{context_str}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {query_str}"
ACCURATE_CONTENT_PROMPTS = "你是一位知识小助手，请根据下面我提供的知识库中相关知识，对我提出的若干问题进行回答，同时回答的内容需满足我所提的要求! \n=====\n 知识库相关知识如下:\n{context_str}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {query_str}"

PROMPT_MAP = {
    SIMPLE_PROMPTS: "Simple",
    GENERAL_PROMPTS: "General",
    EXTRACT_URL_PROMPTS: "Extract URL",
    ACCURATE_CONTENT_PROMPTS: "Accurate Content",
}
