# 开通ElasticSearch

阿里云Elasticsearch是基于[开源Elasticsearch](https://www.elastic.co/cn/elasticsearch/features)构建的全托管Elasticsearch云服务，在100%兼容开源功能的同时，支持开箱即用、按需付费。不仅提供云上开箱即用的Elasticsearch、Logstash、Kibana、Beats在内的Elastic Stack生态组件，还与Elastic官方合作提供免费X-Pack（白金版高级特性）商业插件，集成了安全、SQL、机器学习、告警、监控等高级特性，被广泛应用于实时日志分析处理、信息检索、以及数据的多维查询和统计分析等场景。

更详细的产品介绍请参考：[https://help.aliyun.com/zh/es/product-overview/what-is-alibaba-cloud-elasticsearch?spm=a2c4g.11186623.0.i3](https://help.aliyun.com/zh/es/product-overview/what-is-alibaba-cloud-elasticsearch?spm=a2c4g.11186623.0.i3)

如何快速入门，购买阿里云elasticsearch，以及快速访问和配置，请参考：[https://help.aliyun.com/zh/es/user-guide/elasticsearch-1/?spm=a2c4g.11186623.0.0.35fb486ft9sq0P](https://help.aliyun.com/zh/es/user-guide/elasticsearch-1/?spm=a2c4g.11186623.0.0.35fb486ft9sq0P)
![截屏2024-06-14 10.21.13.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718331689925-1ec64e5e-0d91-4390-b2b5-b202c2d248d5.png#clientId=u3ecc3105-3d48-4&from=ui&height=380&id=u81a81e45&originHeight=1202&originWidth=1170&originalType=binary&ratio=2&rotation=0&showTitle=false&size=387669&status=done&style=none&taskId=uf02461d4-9c6b-435b-8144-b496262732a&title=&width=370)
说明：PAI-RAG服务中使用ElasticSearch的前提是开通阿里云ElasticSearch实例

# PAI-RAG webui中配置ElasticSearch

PAI-RAG 拉起服务后，按照如下步骤配置ElasticSearch

1. **在Please check your Vector Store中选择ElasticSearch**

![截屏2024-06-14 10.43.31.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718333064456-1739daf7-ca7a-4783-9f9c-411b0eb46548.png#clientId=u3ecc3105-3d48-4&from=ui&id=u57149659&originHeight=1804&originWidth=3034&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1255127&status=done&style=none&taskId=u21454120-6dbe-448e-8e5e-e003956affd&title=)

2. **如下4个参数需要手动配置**

![截屏2024-06-14 10.44.50.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718333126572-d6d30157-dbe9-4d2a-ae71-321b9af4edfb.png#clientId=u3ecc3105-3d48-4&from=ui&id=uf112dfdd&originHeight=1808&originWidth=3040&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1254350&status=done&style=none&taskId=ue41d6a30-5b95-495a-812f-92ec4086e3d&title=)
**2.1 ElasticSearch Url**
a) 登录阿里云Elasticsearch控制台，[https://elasticsearch.console.aliyun.com/cn-hangzhou/home](https://elasticsearch.console.aliyun.com/cn-hangzhou/home)
b) 在左侧导航栏，单击**Elasticsearch实例**，右侧页面可以看到已经开通的实例ID，如下图：
![截屏2024-06-14 10.31.07.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718332714043-089d6458-a094-4a60-abfd-447b0de30c65.png#clientId=u3ecc3105-3d48-4&from=ui&id=L16BV&originHeight=804&originWidth=2392&originalType=binary&ratio=2&rotation=0&showTitle=false&size=506148&status=done&style=none&taskId=u3ed30bde-3e1c-4c79-8add-ca1db8aeffc&title=)
c) 点击上图红框中的实例ID，可以看到实例的基本信息，如下图所示：
![截屏2024-06-14 10.39.29.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718332934567-2bac9707-f570-44ec-8a14-25196effdb87.png#clientId=u3ecc3105-3d48-4&from=ui&id=LoiXX&originHeight=1054&originWidth=2426&originalType=binary&ratio=2&rotation=0&showTitle=false&size=528975&status=done&style=none&taskId=ud0d287f6-993f-4a7e-b138-3dd250dfa57&title=)
d) 以公网为例：复制上图红框中的公网地址和公网端口，在ElasticSearch Url中填写：http://<公网地址>:<公网端口>
e) 私网操作步骤类似，需确保es和pai-rag服务位于同一VPC内
**2.2 Index Name**
首先，进入云端ES实例页面，点击左边栏**配置与管理 **中** ES集群配置**，将下图所示的自动创建索引设置为：允许自动创建索引，然后，可在PAI-RAG 界面Index Name中自定义字符串，如：“es-test”
![截屏2024-06-14 17.11.15.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718356404299-95d0ace7-1589-43e1-91fa-779283c61689.png#clientId=u9631f05e-2c0e-4&from=ui&id=u6b06b0d1&originHeight=574&originWidth=3470&originalType=binary&ratio=2&rotation=0&showTitle=false&size=431473&status=done&style=none&taskId=u5754e238-d82d-4004-90fe-23b5749eab3&title=)
**2.3 ES User**
默认填写elastic
**2.4 ES password**
开通es实例时候设置的登陆密码，
![CA7212F0-3183-4304-BA88-3CF1E98F7412.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718334707000-185d3f93-daec-4003-b274-801450a18ef7.png#clientId=u3ecc3105-3d48-4&from=ui&height=275&id=ufc4de49a&originHeight=670&originWidth=1464&originalType=binary&ratio=2&rotation=0&showTitle=false&size=238543&status=done&style=none&taskId=u9aaca1cd-67c1-472d-847f-4865b1527bf&title=&width=600)
如果忘记密码，可在Elasticsearch实例详情页的安全配置中重置
![截屏2024-06-14 11.12.20.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718334846530-29b10106-ce19-4d62-8e56-f7704e144b9f.png#clientId=u3ecc3105-3d48-4&from=ui&height=351&id=u87fe649a&originHeight=774&originWidth=1334&originalType=binary&ratio=2&rotation=0&showTitle=false&size=251020&status=done&style=none&taskId=ue3725e0b-23da-4d73-bf07-2c6b7ef0d8b&title=&width=605)

以上4个参数配置完成后，点击界面右下角的Connet ElasticSearch，看到Connetion Info中如下提示表示ElasticSearch配置成功
![截屏2024-06-14 11.16.29.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718618291035-f5f4def8-66d2-4a5e-ba7a-5f7fe092dcdc.png#clientId=u34bcaa4d-a122-4&from=ui&height=352&id=ud0ad5c18&originHeight=916&originWidth=1526&originalType=binary&ratio=2&rotation=0&showTitle=false&size=392512&status=done&style=none&taskId=u36c9f1e6-1555-4f8a-8671-4f1a58231cf&title=&width=586)

# 基础使用

PAI-RAG webui中，

- 首先在setting中完成embedding mode配置、LLM配置，成功连接es store
- 然后在Upload中完成文档上传以及相关chunk配置
- 即可在Chat中正常使用es全文检索

支持三种检索模式，，如下图所示：

- 向量检索（Embedding Only）
- 关键词检索（Keyword Only）
- 向量和关键词混合检索（Hybrid）

![截屏2024-06-14 11.20.48.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718335290001-d5dd5ae0-1090-48a2-b800-112bd942a72b.png#clientId=u3ecc3105-3d48-4&from=ui&height=415&id=u35a321db&originHeight=1750&originWidth=2308&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1044900&status=done&style=none&taskId=u80506fdb-cb4f-4075-bd3d-db6c730cb8e&title=&width=547)
PAI-RAG默认使用分词器类型为ik-smart，如何人工添加新词和停用词请查看下一章节

# 分词/停用词表个性化配置

阿里云ElasticSearch内置的ik中文分词器可以应对大部分常用词，但并未覆盖一些互联网新词以及一些特殊词汇，此时，通过人工维护分词词典和停用词词典，可以获得更好的检索效果。

前置步骤：
本地准备好一份分词表和停用词表

- 新建“new_word.dic”, 使用文本编辑器打开，并按行添加新词。比如，使用内置分词表，会把“云服务器”分为“云”和“服务器”两个词，按照业务需求希望将其作为一个词，可以将“云服务器”加入新词表中，如下图：

![截屏2024-06-14 14.39.17.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718347170645-deb16717-2794-4054-babd-3ba0874c8b9a.png#clientId=u7edc5c97-eb02-4&from=ui&height=230&id=u70c898f8&originHeight=454&originWidth=972&originalType=binary&ratio=2&rotation=0&showTitle=false&size=109839&status=done&style=none&taskId=u87bb17ef-3c7d-45b7-b928-775534539c7&title=&width=493)
停用词表的准备同理。

准备好自定义的分词表和停用词表后，需要将词表上传到指定位置：

1. 登录[阿里云Elasticsearch控制台](https://elasticsearch.console.aliyun.com/#/home)
2. 在左侧导航栏，单击**Elasticsearch实例**
3. 进入目标实例
   1. 在顶部菜单栏处，选择资源组和地域。
   2. 在**Elasticsearch实例**中单击目标实例ID
4. 在左侧导航栏，选择**配置与管理 > 插件配置**，如下图所示：

![截屏2024-06-14 11.43.51.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718336782889-3ed1a504-d658-4565-83f0-cf15658c1532.png#clientId=u7edc5c97-eb02-4&from=ui&id=u90b6bf5d&originHeight=578&originWidth=3078&originalType=binary&ratio=2&rotation=0&showTitle=false&size=487164&status=done&style=none&taskId=uccb832f7-c125-4ff7-aae5-4570f7c806a&title=)
说明：Elasticsearch 7.16及以上版本的实例和部分地域的基于云原生管控的实例不支持IK词典冷更新。
热更新生效方式：第一次上传词典文件时，会对整个集群的词典进行更新，需要重启集群才能生效；二次上传同名文件不会触发集群重启，在运行过程中直接加载词库。

5. 在**热更新**页面，单击右下方的**配置**。
6. 在**IK主分词词库**下方，选择词典的更新方式，并按照以下说明上传词典文件。

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718336772676-95154846-1f73-4279-83d4-2c61b6c534c5.png#clientId=u7edc5c97-eb02-4&from=paste&id=uf6df4831&originHeight=351&originWidth=573&originalType=url&ratio=2&rotation=0&showTitle=false&size=12325&status=done&style=none&taskId=u53a1d954-a4c0-4947-bbbd-aa0f911125c&title=)
**说明：**IK热更新不支持修改系统自带的主词典，如果您需要修改系统主词典请使用IK冷更新的方式。
阿里云Elasticsearch支持**上传DIC文件**和**添加OSS文件**两种词典更新方式：

- **上传DIC文件**：单击**上传DIC文件**，选择一个本地文件进行上传。
- **添加OSS文件**：输入Bucket名称和文件名称，单击**添加**。请确保Bucket与当前Elasticsearch实例在同一地域下，且文件为DIC文件（以下步骤以**new_word.dic**文件进行说明）。且源端（OSS）的文件内容发生变化后，需要重新手动配置上传才能生效，不支持自动同步更新。

上传成功后，会显示如下：（停用词的上传步骤同上）
![截屏2024-06-14 14.45.18.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718347548381-35e724a1-3feb-4a19-b4dc-e234908a18cf.png#clientId=u7edc5c97-eb02-4&from=ui&height=684&id=uf00ec559&originHeight=1526&originWidth=1212&originalType=binary&ratio=2&rotation=0&showTitle=false&size=393145&status=done&style=none&taskId=u58d50bca-c317-4531-be6e-87f6292c530&title=&width=543)
**警告 **以下操作会重启实例，为保证您的业务不受影响，请确认后在业务低峰期进行操作。

7. 滑动到页面底端，勾选**该操作会重启实例，请确认后操作**（第一次上传词典文件，需要重启），单击**保存**。保存后，集群会进行滚动重启，等待滚动重启结束后，词典会自动生效。

词典使用一段时间后，如果需要扩充或者减少词典中的内容，请继续执行以下步骤修改上传的**new_word.dic**文件。

8. 进入词典热更新页面，先删除之前上传的同名词典文件，重新上传修改过的**new_word.dic**同名词典文件。

因为修改的是已存在的同名词典文件的内容，所以本次上传修改过的同名词典文件不需要滚动重启整个集群。

9. 单击**保存**。

由于阿里云Elasticsearch节点上的插件具有自动加载词典文件的功能，所以每个节点获取词典文件的可能时间不同，请耐心等待词典生效。大概两分钟后再使用更新之后的词典，为了保证准确性，可登录Kibana控制台进行验证，具体可参考：[https://help.aliyun.com/zh/es/user-guide/use-the-analysis-ik-plug-in?spm=a2cba.elasticsearch_plugin.c_plugin.2.4bd6a68cQZ3Jo6](https://help.aliyun.com/zh/es/user-guide/use-the-analysis-ik-plug-in?spm=a2cba.elasticsearch_plugin.c_plugin.2.4bd6a68cQZ3Jo6)

10. 配置完成后，重新拉起PAI-RAG服务，参考2和3中配置（其中，检索模式选择keyword或hybrid），则可以使用更新词表后的ES全文检索

# 索引管理

1. 进入目标实例，点击左侧菜单选项中的“可视化控制”，开通Kibana

![截屏2024-06-25 14.15.32.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1719296187038-ebb7f3ad-2128-46ce-b8d3-d03f20534a4a.png#clientId=ud5bb8e6c-e200-4&from=ui&id=u2465dd69&originHeight=1044&originWidth=1560&originalType=binary&ratio=2&rotation=0&showTitle=false&size=942500&status=done&style=none&taskId=u290eeb2f-8a53-40e4-b77b-129c30b8f3b&title=)

2. 以公网访问为例，点击上图的右下角的修改配置，可以看到如下界面，然后点击红框中的修改，添加需要的ip白名单，以逗号分隔

![截屏2024-06-25 14.17.02.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1719296344933-4500d630-99b4-407d-b7b8-8c10cac2f469.png#clientId=ud5bb8e6c-e200-4&from=ui&id=u237a8148&originHeight=534&originWidth=2166&originalType=binary&ratio=2&rotation=0&showTitle=false&size=582171&status=done&style=none&taskId=ub133ee5b-964d-49f7-a6ae-60089a71389&title=)

3. 添加完成后，回退上层实例界面，点击右下角的公网入口，如下图：

![截屏2024-06-25 14.20.48.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1719296464294-c1fb6e3d-ba37-4149-9056-bd1af56f6409.png#clientId=ud5bb8e6c-e200-4&from=ui&height=492&id=ub57f802b&originHeight=984&originWidth=952&originalType=binary&ratio=2&rotation=0&showTitle=false&size=544405&status=done&style=none&taskId=u0ceb5f7a-a179-4f41-8a58-12c13f1ed7f&title=&width=476)

4. 输入创建es实例的用户名和密码，点击登录，进入界面如下图所示：

![截屏2024-06-25 14.23.46.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1719296686010-64a5d4d6-b5b7-4c47-92a6-cb488ce816a5.png#clientId=ud5bb8e6c-e200-4&from=ui&height=351&id=u59ad63e1&originHeight=1286&originWidth=2042&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1608704&status=done&style=none&taskId=ub8af7d0a-6930-431f-b361-6bfd8ba8ba1&title=&width=558)

5. 点击上图左上角的隐藏菜单，下拉找到Management，可以进行索引管理，如下图所示：

![截屏2024-06-18 17.19.38.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1718702452892-4a802b88-c7b4-48be-aff9-e349b65d4c90.png#clientId=ucd45316e-a74b-4&from=ui&height=335&id=u8bf320a0&originHeight=1382&originWidth=2458&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1916831&status=done&style=none&taskId=uc365f6ce-8786-4fff-a0bb-22217d3e730&title=&width=596)
按照上图红框中操作，在Management中“索引”页面，可以看到2中创建的Index Name (es_test)，点击具体索引名称，则可以对相应的索引进行各种管理操作，如查看设置、清空、关闭等
![截屏2024-07-09 10.04.14.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1720490735797-cb549235-7dc8-4805-aa10-bb702fbfc569.png#clientId=u980b499d-d8dc-4&from=ui&height=182&id=u530e1a71&originHeight=840&originWidth=2752&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1245323&status=done&style=none&taskId=u62f6eebb-5add-43b1-acbf-17f88837cc2&title=&width=596)
点击上图右半部分索引名称（es_test）旁红框图标，可进入到该索引的详情页面，查看切片信息、各类字段如content、embedding等，如下图所示：
![截屏2024-07-09 10.05.10.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/311784/1720490785800-7b20cdf5-5d95-4e28-be4c-8967cd029f30.png#clientId=u980b499d-d8dc-4&from=ui&height=320&id=u891bcee3&originHeight=1108&originWidth=2052&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1550501&status=done&style=none&taskId=ub91d5e87-ecb2-4849-9253-2c268da65ff&title=&width=593)
