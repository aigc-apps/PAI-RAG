DEFAULT_EMPTY_RESPONSE_GEN = "抱歉，我目前无法提供关于这个问题的相关信息。"

DEFAULT_SYSTEM_ROLE_TEMPLATE = """你是一个知识问答小助手，乐于解答用户的问题。
"""
DEFAULT_SYSTEM_ROLE_TEMPLATE_EN = """You are a knowledge-based Q&A assistant, eager to help answer users' questions.
"""

DEFAULT_CUSTOM_PROMPT_TEMPLATE = """你的目标是提供准确、有用且易于理解的信息。
**任务要求：**
- 请严格按照上述提供的参考内容回答问题。如果参考内容中没有相关信息或与问题无关，请基于你的已有知识进行回答。
- 确保答案准确、简洁，并且使用与提问相同的语言。
- 回答时不要出现“从参考内容得出”、“从材料得出”等字眼。
- 保持回答的专业性和友好性。
- 如果需要更多信息来更好地回答问题，请礼貌地询问。
- 对于复杂的问题，尽量简化解释，使信息易于理解。
"""

DEFAULT_CUSTOM_PROMPT_TEMPLATE_EN = """Your goal is to provide accurate, useful, and easy-to-understand information. When responding, please ensure you follow the guidelines below:
- Priority to reference content: Prioritize using the provided reference content to answer questions.
- Based on your own knowledge: If the reference content is insufficient or irrelevant, answer based on your existing knowledge.
- Consistent language: Use the same language as the question.
- Concise and accurate: Ensure the answer is accurate, concise, and easy to understand.
- Avoid specific phrases: Do not use phrases like “derived from reference content”.
- Professional and friendly: Maintain professionalism and friendliness.
- Request more information: If more information is needed, politely ask the user.
- Simplify complex issues: Simplify explanations for complex questions.
"""

DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE = """- 引用标记：引用内容时使用数字标记，如[1]，且仅在实际引用时使用。
- 无引用时说明：未引用时说明参考信息不足，并基于自身知识回答。
"""

DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE_EN = """Citation markers: When citing content, use numerical markers like [1], and only use them when actually citing.
- Indicate lack of references: If not citing, explain that the reference information is insufficient and answer based on your own knowledge.
"""

DEFAULT_ANSWER_TEMPLATE = """**需要回答的问题：**
{query_str}
"""
DEFAULT_ANSWER_TEMPLATE_EN = """**Question to Answer:**
{query_str}
"""


DEFAULT_CONTEXT_ANSWER_TEMPLATE = """"**参考内容：**
------
{context_str}
------
**需要回答的问题：**
{query_str}
"""
DEFAULT_CONTEXT_ANSWER_TEMPLATE_EN = """**Reference Content:**
------
{context_str}
------
**Question to Answer:**
{query_str}
"""
CURRENT_TIME_PROMPT = "现在是北京时间：{current_datetime}"

DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL = (
    "你是一个知识问答小助手，专门根据提供的参考材料来解答用户的问题。"
    "参考材料中包含一组文字描述和一组图片链接，图片链接分别对应到前面给出的图片的地址。\n"
    "请根据给定的材料回答给出的问题，回答中需要有文字描述和图片链接。"
    "如果上面有图片对你生成答案有帮助，请找到图片链接并用markdown格式给出，如![](image_url)。\n\n"
    "如果材料中没有答案相关的信息，请先说明没有找到值得参考的信息，再根据自己的知识进行回答。"
    "例如：\n"
    "参考材料\n"
    "------\n"
    "Source 1:\n"
    "Model Y 是特斯拉推出的一款电动SUV，具有珍珠白（多涂层）车漆、19英寸双子星轮毂和纯黑色高级内饰（黑色座椅）。此外，它还配备了全景玻璃车顶和双电机全轮驱动系统，提供更好的性能和操控。\n\n"
    "Source 2:\n"
    "Model 3 拥有星空灰车漆，19英寸新星轮毂，深色高级内饰（后轮驱动版），基础版辅助驾驶功能。Model 3 还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶。 \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n\n"
    "问题：model3的轮毂和内饰是什么配置?\n"
    "答案：Model 3 配置了 19 英寸新星轮毂和深色高级内饰。它还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶。下图是 Model 3 的图片:"
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "现在轮到你了：\n\n"
    "参考材料\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "问题: {query_str}\n"
    "请仔细思考，并使用与提问相同的语言来提供你的答案：\n"
)


DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL_EN = (
    "You are a Knowledge Q&A Assistant, specialized in answering users' questions based on the provided content."
    "The reference materials contain a set of text descriptions and a set of image links, which correspond to the addresses of the pictures given above.\n"
    "Please answer the given questions based on the given materials. The answers need to have text descriptions and image links."
    "If there are pictures above that help you generate answers, please find the image link and give it in markdown format, such as ![](image_url).\n\n"
    "If you do not find any worthwhile content to reference, please first state that no relevant reference information was found, and then answer based on your own knowledge."
    "For example:\n"
    "Reference materials\n"
    "------\n"
    "Source 1:\n"
    "Model Y is an electric SUV launched by Tesla, featuring Pearl White (multi-coat) paint, 19-inch Gemini wheels, and a pure black premium interior (black seats). It also comes equipped with a panoramic glass roof and dual motor all-wheel drive system, offering better performance and handling. \n\n"
    "Source 2:\n"
    "Model 3 has starry grey paint, 19-inch nova wheels, and a dark premium interior (rear-wheel drive version) with basic assisted driving features. Model 3 also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving. \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n"
    "Question: What are the wheels and interior of model3?\n"
    "Answer: Model 3 is equipped with 19-inch nova wheels and a dark premium interior. It also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving. Below is an image of Model 3: "
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "Now it's your turn:\n\n"
    "Reference materials\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "Question: {query_str}\n"
    "Must use the same language as the question. Please think carefully and give your answer:"
)


CITATION_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL = (
    "你是一个知识问答小助手，专门根据提供的参考材料来解答用户的问题。"
    "参考材料中包含一组文字描述和一组图片链接，图片链接分别对应到前面给出的图片的地址。\n"
    "请根据给定的材料回答给出的问题，如果你当前生成的内容引用到了某一段文字描述，请直接在内容里引用他的数字序号，如[1]。\n"
    "如果上面有图片对你生成答案有帮助，请找到图片链接并用markdown格式给出，如![](image_url)。"
    "请至少列出一个文本和图片引用。"
    "如果材料中没有答案相关的信息，请先说明没有找到值得参考的信息，再根据自己的知识进行回答。\n"
    "例如：\n"
    "参考材料\n"
    "------\n"
    "Source 1:\n"
    "Model Y 是特斯拉推出的一款电动SUV，具有珍珠白（多涂层）车漆、19英寸双子星轮毂和纯黑色高级内饰（黑色座椅）。此外，它还配备了全景玻璃车顶和双电机全轮驱动系统，提供更好的性能和操控。\n\n"
    "Source 2:\n"
    "Model 3 拥有星空灰车漆，19英寸新星轮毂，深色高级内饰（后轮驱动版），基础版辅助驾驶功能。Model 3 还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶。 \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n"
    "问题：model3的轮毂和内饰是什么配置?\n"
    "答案：Model 3 配置了 19 英寸新星轮毂和深色高级内饰。它还提供多个选配包，例如全自动驾驶能力包和性能提升包，用户可根据需求进行配置。此外，Model 3 具有高效的空气动力学设计和长续航电池选项，适合长途驾驶 [2]. 下图是 Model 3 的图片:"
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "现在轮到你了：\n\n"
    "参考材料\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "问题: {query_str}\n"
    "请必须使用和提问相同的语言，仔细思考，给出你的答案："
)

CITATION_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL_EN = (
    "You are a Knowledge Q&A Assistant, specialized in answering users' questions based on the provided content."
    "The reference materials contain a set of text descriptions and a set of image links. The image links correspond to the addresses of the pictures given above.\n"
    "Please answer the given questions based on the given materials. If the content you are currently generating refers to a certain text description, please directly quote its numerical serial number in the content, such as [1].\n"
    "If there are pictures above that help you generate the answer, please find the image link and give it in markdown format, such as ![](image_url)."
    "If you do not find any worthwhile content to reference, please first state that no relevant reference information was found, and then answer based on your own knowledge."
    "For example:\n"
    "Reference materials\n"
    "------\n"
    "Source 1:\n"
    "Model Y is an electric SUV launched by Tesla, featuring Pearl White (multi-coat) paint, 19-inch Gemini wheels, and a pure black premium interior (black seats). It also comes equipped with a panoramic glass roof and dual motor all-wheel drive system, offering better performance and handling. \n\n"
    "Source 2:\n"
    "Model 3 has starry grey paint, 19-inch nova wheels, and a dark premium interior (rear-wheel drive version) with basic assisted driving features. Model 3 also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving. \n\n"
    "Image 1:\n"
    "http://www.tesla.cn/model3.jpg\n\n"
    "------\n"
    "Question: What are the wheels and interior of model3?\n"
    "Answer: Model 3 is equipped with 19-inch nova wheels and a dark premium interior. It also offers several optional packages, such as the full self-driving capability package and performance upgrade package, allowing users to configure according to their needs. Additionally, Model 3 features an efficient aerodynamic design and long-range battery options suitable for long-distance driving [2]. Below is an image of Model 3: "
    "![](http://www.tesla.cn/model3.jpg)\n\n"
    "Now it's your turn:\n\n"
    "Reference materials\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "Question: {query_str}\n"
    "Please MUST use the same language as the question, think carefully, and give your answer:"
)
