# 模型配置

在web界面 Settings 中，选择需要的LLM，如果选择DashScope（通义API），推荐使用qwen-max模型；如果选择PaiEas开源部署，推荐使用qwen2-72b-instruct模型

点击下方Save更新使用的模型

![llm_selection](/docs/figures/data_analysis/llm_selection.png)

点击web界面上方Data Analysis，进入到数据分析页面，支持两种类型的数据分析：连接数据库（mysql）分析 和 上传表格文件（excel/csv）分析

![data_analysis_overview](/docs/figures/data_analysis/data_analysis_overview.png)

# 数据库分析配置

## 数据库连接

连接数据库，选择左上方数据分析类型为 database，出现数据库连接配置界面，如下图：

![db_config](/docs/figures/data_analysis/db_config.png)

其中，

- Dialect为数据库类别，当前支持mysql，默认mysql
- Username和Passoword分别为用户名和密码
- Host为本地或远程数据库url，Port为接口，默认3306
- DBname为需要分析的目标数据库名称
- Tables为需要分析的数据表，格式为：table_A, table_B,... ，默认为空，使用目标数据库中所有数据表
- Descriptions为针对目标数据库中每张表的补充描述，比如对表中字段的进一步解释，可以提升数据分析效果，格式为：{"table_A":"字段a表示xxx，字段b数据的格式为yyy","table_B":"这张表主要用于zzz"}，注意：需要使用英文输入法下的字典格式（英文双引号，冒号，逗号），默认为空
- 支持三种prompt template用于生成sql查询，点击选项后可以看到具体的prompt template，也可在custom中自定义

**注意：** 如自定义prompt，请尽量参考template修改，其中中大括号{}的内容为输入参数，需要保留

确认输入无误后，可直接在右侧chatbot中开始提问，回答结果的Reference中可以看到查询的数据库表名称，生成的sql语句，以及该sql语句是否有效执行。**注意：** 这里有效执行是指sql语句语法有效，不代表业务逻辑一定正确

Reference可以作为查询效果优化的"debug"工具

![db_chat](/docs/figures/data_analysis/db_chat.png)

## 查询效果优化

### Description

针对数据表中字段含义不清晰，或者字段存储内容格式不清晰等问题，可以在Descriptions中增加相应描述，帮助llm更准确提取数据表内容，此处以公开数据集Spider中my_pets数据库为例，其中pets表数据如下：

![table_example](/docs/figures/data_analysis/table_example.png)

问答效果对比：

当描述为空时，对问题“有几只狗”生成的sql查询语句为：SELECT COUNT(\*) FROM pets WHERE PetType = '狗'，查询不到

![db_query_no_desc](/docs/figures/data_analysis/db_query_no_desc.png)

增加简单描述后，生成的sql查询语句为：SELECT COUNT(\*) FROM pets WHERE PetType = 'dog'，可以准确回答

![db_query_desc](/docs/figures/data_analysis/db_query_desc.png)

如果查询效果有明显改善，可以将相应的补充描述在数据库中作为相应table或column的comment持久化添加

### Prompt

此外，还可以通过调整prompt template，优化查询效果

- Reference观察到生成的sql语句包含了非sql的其他内容，如开头多了"sql"等（部分小模型可能会发生），可以在prompt中增加简单限制
- Reference观察到生成的sql语句不满足某些业务逻辑，可以在prompt中给出示例，通过few-shot learning，也可以很快提升效果

# 表格文件分析配置

表格文件配置相对简单，选择左上方的分析类型为：datafile，出现以下界面

![sheet_upload](/docs/figures/data_analysis/sheet_upload.png)

点击左侧中部的上传，一次上传一份表格文件（excel或csv格式），上传成功后，左侧下方会出现文件的前几行预览，如下图所示：

![sheet_data_preview](/docs/figures/data_analysis/sheet_data_preview.png)

上传表格文件后可以直接在右侧chatbot中提问，如需更换表格，重新上传所需表格即可
