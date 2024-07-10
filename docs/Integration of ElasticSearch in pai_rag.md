# 开通ElasticSearch

阿里云Elasticsearch是基于[开源Elasticsearch](https://www.elastic.co/cn/elasticsearch/features)构建的全托管Elasticsearch云服务，在100%兼容开源功能的同时，支持开箱即用、按需付费。不仅提供云上开箱即用的Elasticsearch、Logstash、Kibana、Beats在内的Elastic Stack生态组件，还与Elastic官方合作提供免费X-Pack（白金版高级特性）商业插件，集成了安全、SQL、机器学习、告警、监控等高级特性，被广泛应用于实时日志分析处理、信息检索、以及数据的多维查询和统计分析等场景。

更详细的产品介绍请参考：[https://help.aliyun.com/zh/es/product-overview/what-is-alibaba-cloud-elasticsearch?spm=a2c4g.11186623.0.i3](https://help.aliyun.com/zh/es/product-overview/what-is-alibaba-cloud-elasticsearch?spm=a2c4g.11186623.0.i3)

如何快速入门，购买阿里云elasticsearch，以及快速访问和配置，请参考：[https://help.aliyun.com/zh/es/user-guide/elasticsearch-1/?spm=a2c4g.11186623.0.0.35fb486ft9sq0P](https://help.aliyun.com/zh/es/user-guide/elasticsearch-1/?spm=a2c4g.11186623.0.0.35fb486ft9sq0P)

![aliyun_es_overview.png](/docs/figures/elastic/aliyun_es_overview.png)

说明：PAI-RAG服务中使用ElasticSearch的前提是开通阿里云ElasticSearch实例。

# PAI-RAG webui中配置ElasticSearch

PAI-RAG 拉起服务后，按照如下步骤配置ElasticSearch

1. **在Please check your Vector Store中选择ElasticSearch**

![pairag_select_es.png](/docs/figures/elastic/pairag_select_es.png)

2. **如下4个参数需要手动配置**

![pairag_es_param_list.png](/docs/figures/elastic/pairag_es_param_list.png)

&emsp; 2.1. **ElasticSearch Url**

a) 登录阿里云Elasticsearch控制台，[https://elasticsearch.console.aliyun.com/cn-hangzhou/home](https://elasticsearch.console.aliyun.com/cn-hangzhou/home)

b) 在左侧导航栏，单击 **Elasticsearch实例**，右侧页面可以看到已经开通的实例ID，如下图：

![aliyun_es_instance.png](/docs/figures/elastic/aliyun_es_instance.png)

c) 点击上图红框中的实例ID，可以看到实例的基本信息，如下图所示：

![aliyun_es_instance_info.png](/docs/figures/elastic/aliyun_es_instance_info.png)

d) 以公网为例：复制上图红框中的公网地址和公网端口，在ElasticSearch Url中填写：http://<公网地址>:<公网端口>

e) 私网操作步骤类似，需确保es和pai-rag服务位于同一VPC内

&emsp; 2.2 **Index Name**

首先，进入云端ES实例页面，点击左边栏 **配置与管理** 中 **ES集群配置**，将下图所示的自动创建索引设置为：允许自动创建索引。然后，可在PAI-RAG 界面Index Name中自定义字符串，如：“es-test”。

![aliyun_es_instance_autoindex.png](/docs/figures/elastic/aliyun_es_instance_autoindex.png)

&emsp; 2.3 **ES User**

默认填写 elastic。

&emsp; 2.4 **ES password**

填写开通es实例时候设置的登陆密码。

![aliyun_es_password.png](/docs/figures/elastic/aliyun_es_password.png)

如果忘记密码，可在Elasticsearch实例详情页的安全配置中重置。

![aliyun_es_password_reset.png](/docs/figures/elastic/aliyun_es_password_reset.png)

以上4个参数配置完成后，点击界面右下角的Connet ElasticSearch，看到Connetion Info中如下提示表示ElasticSearch配置成功。

![pairag_es_connect.png](/docs/figures/elastic/pairag_es_connect.png)

# 基础使用

PAI-RAG webui中，

- 首先在setting中完成embedding mode配置、LLM配置，成功连接es store
- 然后在Upload中完成文档上传以及相关chunk配置
- 即可在Chat中正常使用es全文检索

支持三种检索模式，如下图所示：

- 向量检索（Embedding Only）
- 关键词检索（Keyword Only）
- 向量和关键词混合检索（Hybrid）

![pairag_retrieval_mode.png](/docs/figures/elastic/pairag_retrieval_mode.png)

PAI-RAG默认使用分词器类型为ik-smart，如何人工添加分词和停用词请查看下一章节

# 分词/停用词表个性化配置

阿里云ElasticSearch内置的ik中文分词器可以应对大部分常用词，但并未覆盖一些互联网新词以及一些特殊词汇，此时，通过人工维护分词词典和停用词词典，可以获得更好的检索效果。

**前置步骤**

本地准备好一份分词表和停用词表

- 新建“new_word.dic”, 使用文本编辑器打开，并按行添加新词。比如，使用内置分词表，会把“云服务器”分为“云”和“服务器”两个词，按照业务需求希望将其作为一个词，可以将“云服务器”加入新词表中，如下图：

![new_word_dict.png](/docs/figures/elastic/new_word_dict.png)

停用词表的准备同理。

准备好自定义的分词表和停用词表后，需要将词表上传到指定位置：

1. 登录[阿里云Elasticsearch控制台](https://elasticsearch.console.aliyun.com/#/home)
2. 在左侧导航栏，单击 **Elasticsearch实例**
3. 进入目标实例
   1. 在顶部菜单栏处，选择资源组和地域。
   2. 在 **Elasticsearch实例** 中单击目标实例ID
4. 在左侧导航栏，选择 **配置与管理 > 插件配置**，如下图所示：

![aliyun_es_plugin.png](/docs/figures/elastic/aliyun_es_plugin.png)

说明：Elasticsearch 7.16及以上版本的实例和部分地域的基于云原生管控的实例不支持IK词典冷更新。

热更新生效方式：第一次上传词典文件时，会对整个集群的词典进行更新，需要重启集群才能生效；二次上传同名文件不会触发集群重启，在运行过程中直接加载词库。

5. 在 **热更新** 页面，单击右下方的 **配置**。
6. 在 **IK主分词词库** 下方，选择词典的更新方式，并按照以下说明上传词典文件。

![aliyun_es_upload_dic.png](/docs/figures/elastic/aliyun_es_upload_dic.png)

**说明:** IK热更新不支持修改系统自带的主词典，如果您需要修改系统主词典请使用IK冷更新的方式。

阿里云Elasticsearch支持 **上传DIC文件** 和 **添加OSS文件** 两种词典更新方式：

- **上传DIC文件**：单击 **上传DIC文件**，选择一个本地文件进行上传。
- **添加OSS文件**：输入Bucket名称和文件名称，单击 **添加**。请确保Bucket与当前Elasticsearch实例在同一地域下，且文件为DIC文件（以下步骤以 **new_word.dic** 文件进行说明）。且源端（OSS）的文件内容发生变化后，需要重新手动配置上传才能生效，不支持自动同步更新。

上传成功后，会显示如下：（停用词的上传步骤同上）

![aliyun_es_ik_hot_update.png](/docs/figures/elastic/aliyun_es_ik_hot_update.png)

**警告:** 以下操作会重启实例，为保证您的业务不受影响，请确认后在业务低峰期进行操作。

7. 滑动到页面底端，勾选 **该操作会重启实例，请确认后操作**（第一次上传词典文件，需要重启），单击 **保存** 。保存后，集群会进行滚动重启，等待滚动重启结束后，词典会自动生效。

词典使用一段时间后，如果需要扩充或者减少词典中的内容，请继续执行以下步骤修改上传的**new_word.dic** 文件。

8. 进入词典热更新页面，先删除之前上传的同名词典文件，重新上传修改过的 **new_word.dic** 同名词典文件。

因为修改的是已存在的同名词典文件的内容，所以本次上传修改过的同名词典文件不需要滚动重启整个集群。

9. 单击 **保存**。

由于阿里云Elasticsearch节点上的插件具有自动加载词典文件的功能，所以每个节点获取词典文件的可能时间不同，请耐心等待词典生效。大概两分钟后再使用更新之后的词典，为了保证准确性，可登录Kibana控制台进行验证，具体可参考：[https://help.aliyun.com/zh/es/user-guide/use-the-analysis-ik-plug-in?spm=a2cba.elasticsearch_plugin.c_plugin.2.4bd6a68cQZ3Jo6](https://help.aliyun.com/zh/es/user-guide/use-the-analysis-ik-plug-in?spm=a2cba.elasticsearch_plugin.c_plugin.2.4bd6a68cQZ3Jo6)

10. 配置完成后，重新拉起PAI-RAG服务，参考前述webui配置（其中，检索模式选择keyword或hybrid），则可以使用更新词表后的ES全文检索。

# 索引管理

1. 进入目标实例，点击左侧菜单选项中的“可视化控制”，开通Kibana

![aliyun_es_kibana.png](/docs/figures/elastic/aliyun_es_kibana.png)

2. 以公网访问为例，点击上图的右下角的修改配置，可以看到如下界面，然后点击红框中的修改，添加需要的ip白名单，以逗号分隔

![aliyun_es_kibana_whitelist.png](/docs/figures/elastic/aliyun_es_kibana_whitelist.png)

3. 添加完成后，回退上层实例界面，点击右下方的公网入口，如下图：

![aliyun_es_kibana_entry.png](/docs/figures/elastic/aliyun_es_kibana_entry.png)

4. 输入创建es实例的用户名和密码，点击登录，进入界面如下图所示：

![aliyun_es_kibana_menu.png](/docs/figures/elastic/aliyun_es_kibana_menu.png)

5. 点击上图左上角的隐藏菜单，下拉找到Management，可以进行索引管理，如下图所示：

![aliyun_es_kibana_management.png](/docs/figures/elastic/aliyun_es_kibana_management.png)

6. 按照上图红框中操作，在Management中“索引”页面，可以看到章节2中创建的Index Name (es_test)，点击具体索引名称，则可以对相应的索引进行各种管理操作，如查看设置、清空、关闭等。

![aliyun_es_kibana_index_management.png](/docs/figures/elastic/aliyun_es_kibana_index_management.png)

7. 点击上图右侧索引名称（es_test）旁红框图标，可进入到该索引的详情页面，查看切片信息、各类字段如content、embedding等，如下图所示：

![aliyun_es_kibana_index_detail.png](/docs/figures/elastic/aliyun_es_kibana_index_detail.png)
