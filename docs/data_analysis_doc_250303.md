# 自动化数据分析

RAG支持两种数据分析类型：database（连接MySQL数据库）和 datafile（上传表格文件）。以下内容将分别介绍如何使用这两种分析类型进行自动化数据分析。

# 大模型配置

首先需要配置 LLM 信息，在 web 界面 Settings 右上方 Large Language Model 处，如下图所示：

- 使用DashScope（通义API），推荐使用qwen-max，配置示例：

  - LLM Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - API Key: [DashScope模型服务灵积](https://dashscope.console.aliyun.com/apiKey)
  - Model Name: qwen-max, qwen-turbo等

- 使用PaiEas部署的开源大语言模型，配置示例：
  - LLM Base URL: 填写EAS上调用信息中的公网/私网地址（如果使用vllm, 地址后加/v1）`http://1730xxx63.cn-beijing.pai-eas.aliyuncs.com/api/predict/deepseek_v3/v1`
  - API Key: 填写EAS上调用信息的token
  - Model Name: 填写部署时设置的模型名称，可从EAS上服务配置的 system_eas_llm_open_source_model_name 中找到

点击下方 Save Llm Setting 更新模型信息。

![llm_config](/docs/figures/data_analysis/llm_config.png)

点击 web 界面上方 DBChat ，进入到数据分析页面。

![data_analysis_overview](/docs/figures/data_analysis/DBChat_overview.png)

# 数据库分析配置

## 数据库连接

连接数据库，选择左上方数据分析类型为 database，出现数据库连接配置界面，如下图：

![db_config](/docs/figures/data_analysis/db_config_update.png)

- Dialect 是数据库类别，当前支持 mysql，默认 mysql。

- Username 和 Passoword 分别为用户名和密码。

- Host 是本地或远程数据库 url，Port 是端口，默认 3306。

- Database 是需要分析的目标数据库名称。

- Tables 是需要分析的数据表，格式为：table_A, table_B,... ，默认为空，使用目标数据库中所有数据表。

- Table Comment 是针对目标数据库中每张表的补充描述，可以是表格的整体描述或者对表中字段的解释，格式为：{"table_A":"table_A是xxx，字段a表示xxx，字段b数据的格式为xxx","table_B":"这张表主要用于xxx"}，注意：需要使用英文输入法下的字典格式（英文双引号，冒号，逗号），默认为空，该功能主要用于临时调试，观察数据分析效果，如果获得理想效果或者描述信息较多，建议：1. 将相关描述在数据库中作为相应table或column的comment持久化添加；2. 将相关描述信息整理成csv文档上传（下文会介绍文档格式和上传方式）。

- Enhancement options for larger database 该折叠选项下的功能如下图所示：

  ![db_enhanced_features](/docs/figures/data_analysis/db_enhanced_features.png)

  - Enhance db retrieval by embedding 当目标数据库中表列数量较多，如所有表的字段总和大于50列，建议开启该功能，会基于用户问题对数据库结构信息和值信息进行向量检索，快速筛选出可能有用的表字段，避免输入llm的提示词信息过长，导致超过最大允许长度或生成sql效果下降。其中，基于值的检索会获取数据库中文本字段的非重复值，设置了最大列数量（Max Column Number）和 每列最大值数量的上限，上限越大查询的非重复值范围越广，可能效果更好，但延迟会更高。如果目标数据库中表列数量较少，则无需使用此功能。

  - Enable db schema selection by llm 该功能也是应对多表多列的场景，利用大语言模型根据用户查询筛选可能有用的表列。

  - Enable db query history/example 该功能支持用户上传 json 格式的 query-sql 文档，通过提供数据库中历史查询或者相似查询，使用基于向量检索获得与问题相似度较高的查询和sql提供参考，类似于FAQ，有助提高生成sql的效果，提供的数据格式如下：

    ```json
    [
      {
        "query": "找出体重大于10的宠物的数量。",
        "SQL": "SELECT count(*) FROM pets WHERE weight  >  10"
      },
      {
        "query": "找出每种宠物的最大重量。列出最大重量和宠物类型。",
        "SQL": "SELECT max(weight) ,  petType FROM pets GROUP BY petType"
      },
      {
        "query": "找出20岁以上学生拥有的宠物数量。",
        "SQL": "SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.age  >  20"
      }
    ]
    ```

  - Enable db description upload 该功能支持用户上传 csv 格式的数据表描述文档（多份），通过提供数据库中每张表的所有列描述，帮助大模型更好理解字段含义，有助有助提高sql生成效果。提供的csv文件需与目标数据库表名保持一致，可上传多张表的描述，每份csv数据格式如下：

    | original_column_name | column_name                                                               | column_description                                                        | data_format | value_description                                                                                                                                 |
    | -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
    | cds                  |                                                                           | California Department Schools                                             | text        | useless                                                                                                                                           |
    | rtype                |                                                                           | rtype                                                                     | text        | useless                                                                                                                                           |
    | sname                | school name                                                               | school name                                                               | text        |                                                                                                                                                   |
    | dname                | district name                                                             | district name                                                             | text        |                                                                                                                                                   |
    | cname                | county name                                                               | county name                                                               | text        |                                                                                                                                                   |
    | enroll12             | enrollment (1st-12nd grade)                                               | enrollment (1st-12nd grade)                                               | integer     |                                                                                                                                                   |
    | NumTstTakr           | Number of Test Takers                                                     | Number of Test Takers in this school                                      | integer     | number of test takers in each school                                                                                                              |
    | AvgScrRead           | average score                                                             | average score                                                             | integer     | average score for reading                                                                                                                         |
    | NumGE1500            | Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500 | Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500 | integer     | Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500 \n\n commonsense evidence: \n Excellence Rate = NumGE1500 / NumTstTakr" |

    其中，

    - original_column_name 表示数据库中存储的列（字段）名。
    - column_name 可以表示字段的别名，可不填写。
    - column_description 表示字段的描述信息，可以包含中/英文描述。
    - data_type 表示字段的数据类型，可以是字符串、整数、浮点数等，可不填写。
    - value_description 表示字段的值的描述信息，可以包含中/英文描述，如果字段描述和字段值描述区分不大，仅在一处填写即可。

数据库的所有连接信息配置妥当后，点击如下图的 Load DB Info 按钮，即可完成数据库的连接配置。如果需要更新数据表、字段的描述信息，在相应位置更新完毕后再次点击 Load DB Info 即可。

![db_load_info](/docs/figures/data_analysis/db_info_load.png)

## 提示词配置

支持自定义用于生成sql和生成最终回复prompt，如下图所示：

![prompt_config](/docs/figures/data_analysis/prompt_config.png)

如有自定义需求，可在上图模版的下划虚线之间添加或修改。下划虚线以外大括号{}的内容为输入参数，需要保留。如需恢复，可以点击Prompt Reset选择需要恢复的prompt，如下图：

![prompt_reset](/docs/figures/data_analysis/prompt_reset.png)

## 查询及优化

目标数据库信息以及提示词配置完成后，可直接在右侧chatbot中开始提问，回答结果的Reference中可以看到查询的数据库表名称，生成的sql语句，以及该sql语句是否有效执行。
**注意：** 这里有效执行是指sql语句语法有效，不代表业务逻辑一定正确。Reference可以作为查询效果优化的"debug"工具。

目前默认支持对话记忆功能，简单的问答效果如下所示：

![db_chat](/docs/figures/data_analysis/db_chat_with_memo.png)

如果发现问答效果不理想，可以尝试如下优化方式：

- 如果目标数据库中表的数量较多或通过referece发现查询的数据表有误
  - 尝试在 Tables 中限定查询表的范围。
  - 在 Table Comment 中增加表的解释，帮助大模型更好理解不同数据表之间的关系和作用。
- 如果数据库中表列名称比较抽象，如以简单字母命名或者以某专业领域术语命名，请务必增加解释，辅助模型模型理解具体领域知识
  - 通过数据库中 comment 添加。
  - 通过 Table Comment 添加。
  - 通过提供字段描述的 csv 并通过 Enhancement options 中的 Enable db description upload 上传。
- 通过reference观察到已经到生成的sql语句不满足某些业务逻辑或部分业务逻辑本身较复杂
  - 如果业务逻辑相对通用，可通过prompt template中增加相关业务逻辑的提示或者给出相关示例。
  - 按 Enhancement options 中的 Enable db query history/example 准备 query-sql样例数据，作为 few-shot learning 给模型提供更直观的参考。
- 如果目标数据库表列数量较多，如字段数量总数超过50个
  - 进一步优化各个表的字段描述。
  - 开启 Enhancement options 中的前两项，通过embedding和llm的选表表列减少prompt中的信息干扰。
- 通过Reference观察到生成的sql语句包含了非sql的其他内容，如开头多了"sql"等（部分小模型指令遵循问题），可以在nl2sql prompt中增加简单限制。
- 如sql生成正确，需要更加个性化的回答（如以什么语言风格，加上具体单位等要求），可通过synthesizer prompt调整。

# 表格文件分析配置

表格文件配置相对简单，选择左上方的分析类型为：datafile，出现以下界面：

![datafile_config](/docs/figures/data_analysis/datafile_config.png)

点击左侧中部的上传，一次上传一份格式规整的表格文件（excel或csv格式），上传成功后，左侧下方会出现文件的前几行预览，如下图所示：

![datafile_chat](/docs/figures/data_analysis/datafile_chat.png)

上传表格文件后可以直接在右侧chatbot中提问，如需更换表格，重新上传所需表格即可。

# FAQ

<details>
<summary>数据库连接不上</summary>

### 网络和连接配置问题

- 检查提供的主机名、端口号、用户名和密码是否正确：确保所有信息准确无误。
- 防火墙和安全组设置：确认防火墙规则和安全组允许通过指定端口进行访问。
- VPC 和私网设置：如果在同一 VPC 内，直接使用私网地址；若需公网访问，确保已开通公网访问。

### 权限和认证问题

- 用户权限：确认用户具有足够的权限访问目标数据库，必要时使用 GRANT 命令赋予权限。
- 白名单设置：确保客户端 IP 地址在数据库的访问白名单中。

### 数据库服务状态

- MySQL 服务是否正在运行：检查 MySQL 服务的状态，确保其正常运行。
- 监听端口是否正确：确认 MySQL 服务正在监听正确的端口。

</details>
