# Docker build

## Server

### CPU

```bash
docker build -f Dockerfile -t rag_serve:0.1 .
```

### GPU

```bash
docker build -f Dockerfile_gpu -t rag_serve:0.1_gpu .
```

## UI

```bash
docker build -f Dockerfile_ui -t rag_ui:0.1 .
```

## Nginx

```bash
docker build -f Dockerfile_nginx -t rag_nginx:0.1 .
```

# 常见问题

## docker pull timeout

建议更换docker镜像源为阿里云镜像，在阿里云在容器镜像服务 -> 镜像工具 -> 镜像加速器 中可以找到阿里云的专属镜像加速器，按照指示说明修改daemon配置文件来使用加速器即可。
