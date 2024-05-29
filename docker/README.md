# 镜像打包与使用说明

## 1. 镜像构建

### 1.1 EasyRAG镜像

```bash
cd docker/
docker build -t rag:test -f Dockerfile .
```

- 环境建议使用新加坡ECS或本机电脑。由于需要下载基础镜像nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04及安装一系列依赖，弹内开发机网络受限。

- 其中，由于新加坡无法连接Gitlab Repo，因此项目代码部分采用先打包上传至OSS，再下载代码包的方式。

打包命令可参考：

```bash
sh scripts/package_and_upload_to_oss.sh
```

- 已经写好Dockerfile，可酌情根据需要调整。【注意替换其中的代码包名称: EasyRAG_yyyymmdd-hhmmss-N.tar.gz】

### 1.2 （可选）Arize-phoenix镜像

- 为了在EAS上部署Arize-phoenix-tracing服务，故采取镜像方式。

- 采用Arize-phoenix官方镜像，地址：**arizephoenix/phoenix:latest**。

## 2. 本地测试

```bash
docker run -td --network host rag:test
docker exec -it [id] bash
# 进入容器后执行 (按需export环境变量)
pai_rag run
```

## 3. 镜像上传

测试阶段先使用组内个人镜像 mybigpai:aigc_apps
[mybigpai:aigc_apps地址](https://cr.console.aliyun.com/repository/cn-beijing/mybigpai/aigc_apps/details)

```bash
$ docker login --username=xxxx@xxx registry.cn-beijing.aliyuncs.com
$ docker tag [ImageId] registry.cn-beijing.aliyuncs.com/mybigpai/aigc_apps:[镜像版本号]
$ docker push registry.cn-beijing.aliyuncs.com/mybigpai/aigc_apps:[镜像版本号]
```
