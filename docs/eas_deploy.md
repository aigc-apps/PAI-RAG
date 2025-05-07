# EAS自定义部署RAG服务

模型在线服务EAS（Elastic Algorithm Service）是阿里云PAI产品为实现一站式模型开发部署应用，针对在线推理场景提供的模型在线服务，支持将模型服务部署在公共资源组或专属资源组，实现基于异构硬件（CPU和GPU）的模型加载和数据请求的实时响应。

我们支持通过`场景化部署`和`自定义部署`两种方式来一键部署RAG服务，其中，

- `场景化部署`: 更加方便，只需要配置几个参数即可完成。可参考[场景化部署文档](https://help.aliyun.com/zh/pai/user-guide/deploy-a-rag-based-dialogue-system)。
- `自定义部署`可以更灵活地配置服务，比如，部署GPU版本镜像，配置链路追踪服务等等。可参考[自定义部署文档](https://help.aliyun.com/zh/pai/use-cases/custom-deployment-of-rag-service#47e8104831b4f)
