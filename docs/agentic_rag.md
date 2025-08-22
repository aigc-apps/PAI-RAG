# 简介
Agentic RAG Chat UI 是一个基于检索增强生成（RAG）技术的智能对话界面，融合了Agent能力，能够帮助用户构建基于自有知识库的智能对话机器人。

# 特点
+ 支持多知识库管理
+ 灵活的模型配置和MCP工具配置
+ 强大的Agent能力扩展
+ 完善的链路追踪与调试
+ 细粒度的权限控制

# 前置条件
- 已成功完成[安装指南](quick_start.md)中的所有步骤，服务正常运行
- 前端可通过 http://localhost:8680 访问

# 核心功能
## 知识库管理
### 1. 新建知识库

启动PAI-RAG服务后，打开浏览器访问 http://localhost:8680

点击左侧"知识库" → 点击"+新建知识库"，进入知识库配置页面

### 2. 配置知识库

```json
知识库名称: PAI_RAG产品用户手册 (自定义，建议名字能清楚表示知识库的类别)
知识库描述:（可选，建议清晰描述知识库类别及用途）
切片大小: 1000(自定义,默认1000)
切片重叠: 50(自定义,默认50)
向量模型: BAAI/bge-m3 (自定义,默认BAAI/bge-m3)
Top-K:5(自定义,默认5)
相似度阈值:0.7(自定义,默认0.2)
检索策略: 向量检索(三种策略选一种,默认向量检索)
开启重排序:可勾选(默认不勾选，勾选后，需选择重排序模型)
```

![](images/chat/kb_settings.jpg)

### 3. 点击"创建", 创建知识库
### 4. 文件管理

   点击"上传文件", 选择本地文件，进行上传。

   上传结束后，可在"状态"一栏下查看上传状态。

![](images/chat/kb_status.jpg)

#### a. 点击"文件预览"，可查看原文件。

![](images/chat/show_file.jpg)

#### b. 点击"源链接"，添加文件的外部链接。

![](images/chat/file_url_setting.jpg)

#### c. 点击"查看切片"，可查看文件切片。可点击按钮决定chunk是否激活，未激活的chunk不会被检索召回。

![](images/chat/file_chunk.jpg)

可编辑chunk内容

![](images/chat/edit_file_chunk.jpg)

#### d. 文件权限查看并设置

 点击“权限”，查看并设置文档权限。

![](images/chat/file_permission_management.jpg)

#### e. 文件元数据查看并设置

点击“元数据”，查看并设置文档元数据。

点击“编辑”，点击“添加”，可添加元数据。前提：在知识库设置页面添加了元数据。

![](images/chat/edit_file_metadata.jpg)

#### f. 删除文件

点击"删除",删除文件。

### 5. 知识库设置

创建知识库后，可编辑知识库。点击知识库卡片，编辑以下字段，并点击"保存设置"

```json
知识库名称: PAI_RAG产品用户手册 (自定义)
知识库描述:（可选）
切片大小: 1000(自定义,默认1000)
切片重叠: 50(自定义,默认50)
向量模型: BAAI/bge-m3 (自定义,默认BAAI/bge-m3)
Top-K:5(自定义,默认5)
相似度阈值:0.7(自定义,默认0.2)
检索策略: 向量检索(三种策略选一种,默认向量检索)
开启重排序: 可勾选(默认不勾选，勾选后，需选择重排序模型)
元数据配置: 可点击右侧"添加元数据", 
```

![](images/chat/kb_setting_2.jpg)

知识库可添加元数据配置。

![](images/chat/add_metadata.jpg)

### 6. 检索测试

测试chunk检索。在输入框输入检索文字。（可选）点击元数据，选择逻辑操作符，点击新增过滤规则，选择自定义元数据名称，选择规则，在输入框填写具体值。（可选）输入user_id。

点击开始查询，得到查询chunk。

![](images/chat/kb_retriever.jpg)

![](images/chat/kb_retriever_2.jpg)

### 7. 删除知识库

点击知识库左下角"垃圾桶"图标，删除知识库  
![](images/chat/kb_card.jpg)

## 模型管理
### LLM模型管理
#### 1. 添加LLM模型

点击左下角 Settings（设置图标）→ 选择 Model（模型）选项卡→ 进入模型配置页面→ 点击LLM→点击添加LLM模型

![](images/chat/add_model.jpg)

#### 2. 配置LLM模型

有思考与非思考两种模式的模型，可通过思考模型选项来控制是否思考。

```json
模型ID: qwen-test (可自定义)
Endpoint URL: https://dashscope.aliyuncs.com/compatible-mode/v1 (根据实际模型情况填写, OpenAI兼容的API端口，一般以**/v1**结尾)
API Key: your_api_key (填写实际密钥)
模型名称: qwen-max (根据实际模型名称填写)
多模态模型: 如果是多模态模型，则勾选，否则不勾选（默认不勾选）
思考模型: 如果是思考模型，则勾选，否则不勾选（默认不勾选）
```

![](images/chat/llm_setting.jpg)

#### 3. 完成添加

点击新增，完成添加

![](images/chat/llm_card.jpg)

#### 4. 编辑和删除

点击模型卡片右下角的"垃圾箱"删除模型，点击模型卡片右下角的"编辑"编辑模型。

### Embedding模型管理
#### 1. 添加Embedding模型

点击左下角 Settings（设置图标）→ 选择 Model（模型）选项卡→ 进入模型配置页面→ 点击Embedding→点击添加Embedding模型。在启动时默认添加了BAAI/bge-m3 Embedding模型。

![](images/chat/add_embedding_model.jpg)

#### 2. 配置Embedding模型

本地模式：根据模型名称自动从modelscope上下载

```json
模型ID: custom_embedding (可自定义)
模型名称: BAAI/bge-reranker-v2-m3 (modelscope上模型名称)
模型类型: 本地
向量维度: （可填）
向量Batch大小:（可填）
默认向量模型:（可勾选，默认不勾选）
```

![](images/chat/embedding_model_local_setting.jpg)

API模式：根据API调用模型

```json
模型ID: custom_embedding (可自定义)
Endpoint URL: https://dashscope.aliyuncs.com/compatible-mode/v1 (根据实际模型情况填写)
API Key: your_api_key (填写实际密钥)
模型名称: text-embedding-v4 (根据实际模型名称填写)
向量维度: （可填）
向量Batch大小:（可填）
默认向量模型:（可勾选，默认不勾选）
```

![](images/chat/embedding_model_api_setting.jpg)

#### 3. 完成添加

点击新增，完成添加

#### 4. 编辑和删除

点击模型卡片右下角的"垃圾箱"删除模型，点击模型卡片右下角的"编辑"编辑模型。

### Reranker模型管理
#### 1. 添加Reranker模型

点击左下角 Settings（设置图标）→ 选择 Model（模型）选项卡→ 进入模型配置页面→ 点击Reranker→点击添加Reranker模型。

#### 2. 配置Reranker模型

```json
模型ID: custom_reranker (可自定义)
模型名称: Qwen3-Reranker-0.6B (根据实际模型名称填写)
Base URL: https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-0.6B (根据实际模型情况填写)
API Key: your_api_key (填写实际密钥)
```

![](images/chat/reranker_setting.jpg)

#### 3. 完成添加

点击新增，完成添加

#### 4. 编辑和删除

点击模型卡片右下角的"垃圾箱"删除模型，点击模型卡片右下角的"编辑"编辑模型。

## MCP管理
### 1. 新建MCP工具

点击左下角 Settings（设置图标）→ 选择 MCP选项卡 → 进入MCP配置页面 → 点击添加MCP

![](images/chat/add_mcp.jpg)

### 2. 配置MCP工具。填写完下面字段后，点击添加，添加成功。

```json
MCP名称: amaps（自定义）
MCP链接: https://mcp-server-amap-jitptfyoyw.cn-hangzhou.fcapp.run/sse(根据实际MCP情况填写)
MCP类型: sse
Bearer Token:验证信息（可选）
默认启用: 勾选（默认勾选）
```

![](images/chat/mcp_setting.jpg)

![](images/chat/edit_mcp.jpg)

### 3.编辑MCP工具。点击MCP右侧的操作编辑图标，对MCP进行编辑。
### 4. 删除MCP。点击MCP右侧的操作删除图标，删除MCP。

## 搜索管理
### 1. 配置搜索工具

点击左下角 Settings（设置图标）→ 选择 搜索 → 进入搜索配置页面 → 配置搜索 → 点击保存搜索配置保存

![](images/chat/search_setting.jpg)

## Prompt管理
### 1. 查看Prompt

点击左下角 Settings（设置图标）→ 选择 Prompt → 进入Prompt配置页面

pai_rag有一套默认prompt，用户可根据需求修改prompt。

![](images/chat/prompts.jpg)

### 2. 修改Prompt

选中待修改Prompt，进行编辑。编辑好后，点击右下角保存所有Prompt，更新Prompt。prompt里{}的内容不要改动。

## 链路追踪管理
### 1. 配置链路追踪

点击左下角 Settings（设置图标）→ 选择 链路追踪 → 进入链路追踪配置页面 → 配置链路追踪 → 保存链路追踪配置

```json
Endpoint: http://tracing-analysis-dc-hz.aliyuncs.com:8090（根据实际tracing链接填写）
Token: private_token_abc(根据实际Token情况填写)
ServiceName: test_gpu（EAS服务名称）
默认启用: 勾选（默认未勾选，勾选表示启用。如果想要停用，不勾选即可）
```

![](images/chat/tracing_setting.jpg)

### 2. 在EAS上查看链路追踪

登录阿里云官网 → 进入人工智能平台PAI → 点击左侧 模型在线服务EAS → 选择对应region → 找到填写的EAS名称对应的服务 → 点击服务 → 点击链路追踪 → 可看到trace内容

![](images/chat/eas_trace.jpg)

## 权限控制
权限控制可添加角色配置和用户-角色关系，可对知识库进行多粒度的权限管控。

### 1. 查看权限控制

点击左下角 Settings（设置图标）→ 选择 权限控制 → 进入权限控制页面

![](images/chat/permission_setting.jpg)

### 2. 角色配置

点击角色配置 → 选择添加角色 → 填写角色名称和角色描述 → 点击保存

![](images/chat/role_setting_1.jpg)

![](images/chat/role_setting_2.jpg)

### 3. 用户-角色关系

点击用户-角色关系 → 选择添加用户角色 → 填写用户ID和选择角色名称 → 点击保存

![](images/chat/user_setting_1.jpg)

![](images/chat/user_setting_2.jpg)

## chatbot应用管理
chat应用是一种简化调用模式的应用。一次性完成一个应用配置后，后续调用该应用进行对话无需重复设置各种配置，旨在降低用户使用门槛，快速发送请求。如需修改对话配置，直接修改应用配置，保存即可。

### 1. 新建应用

点击左侧"应用" → 点击"+新建应用"，进入应用配置

### 2. 应用配置

按照如下配置配置应用后，点击创建，创建应用。

```json
App ID: chatbot (自定义)
描述:（可填）
基模型选择: qwen-max(配置模型后选择)
启动联网搜索: 关闭（默认关闭，配置联网搜索后，可选择打开）
Agentic模式: 关闭（默认关闭，可选择打开，打开后是Agentic模式）
知识库选择: 知识库测试(配置模型后选择，可多选)
MCP选择: amaps(配置MCP后选择，可多选)
```

![](images/chat/chatbot_setting.jpg)

### 3. 编辑和删除

点击应用卡片左下角的"垃圾箱"删除模型，点击应用卡片编辑应用。

![](images/chat/chatbot_card.jpg)



## 对话
### 1. 新建对话

点击左侧栏对话 → 新建对话，新建立一个对话

### 2. 在对话页面，选择模型或者应用进行对话。

![](images/chat/chat_model_selector.jpg)

### 3. 选择模型进行对话。

选择模型进行对话的时候，对话输入栏里会有四种可选配置。根据需求，从对话输入栏里选择配置。

![](images/chat/chat_setting.jpg)

#### a. 深度思考：选中深度思考，对话采用agentic模式，进行多轮思考。不选中深度思考，对话采用普通单轮chat模式。

#### b. 搜索：选中搜索，在配置了搜索的情况下，对话可调用搜索工具。不选中搜索，对话不调用搜索工具。

#### c. mcp: 点击mcp，弹框展示已经配置并启用的MCP工具，选择所需的MCP工具激活，点击保存，对话可调用激活的MCP。不选中MCP激活，对话不调用MCP工具。

![](images/chat/chat_mcp_choices.jpg)

#### d. 知识库：点击知识库，弹框显示已经配置的知识库，选择所需的知识库激活，点击保存，对话可进行RAG

![](images/chat/chat_kb_choices.jpg)

#### e.上传附件

点击输入框左上角的上传附件，可上传本地文件。支持10M以下文件上传，一次性上传不超过5个。

![](images/chat/chat_attachments.jpg)

### 4. 选择应用进行对话。

选择模型进行对话的时候，对话输入栏里没有配置可选，配置是在应用中选择的。如果想要修改或者查看对话配置，请点击左侧栏应用，选择对应的应用查看。配置好chat应用后，在对话页面上方，选择对应应用进行对话。

#### a. 上传附件

点击输入框左上角的上传附件，可上传本地文件

![](images/chat/chatbot_attachment.jpg)

### 5. 问答效果。

#### a.非Agentic模式：使用搜索工具

![](images/chat/chat_with_search_tool.jpg)

#### b.非Agentic模式：使用知识库

![](images/chat/chat_with_kb.jpg)

#### c.非Agentic模式：使用附件上传

![](images/chat/chat_with_attachments.jpg)

#### d.Agentic模式：使用知识库

![](images/chat/agentic_chat_with_kb.jpg)

#### e. Agentic模式：使用mcp工具

![](images/chat/agentic_chat_with_mcp.jpg)


