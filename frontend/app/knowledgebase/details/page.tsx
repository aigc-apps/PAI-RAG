"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Loader2, CheckCircle } from "lucide-react";
import { PreviewButton } from "@/app/knowledgebase/details/preview-button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Slider } from "@/components/ui/slider";
import { ScanSearch, TextSearch, SearchCode } from "lucide-react";
import * as Toast from "@radix-ui/react-toast";

interface KnowledgeBaseFile {
  id: string;
  name: string;
  type: string;
  size: string;
  status: string;
  uploadedAt: string;
}

interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  doc_num: number;
  chunk_num: number;
  chunk_config: {
    parser_type: string; // 切片类型
    separator: string; // 切片标识符
    chunk_size: string; // 切片大小
    chunk_overlap: string; // 切片重叠大小
  };
  embedding_model: string; //向量模型名称
  retrieval_config: {
    retrieval_mode: string; // 索引类型：vector, fulltext, hybrid
    top_k: number; // Top-K 值
    similarity_threshold: string; // 相似度分数阈值
    rerank_model: string; // rerank模型名称
    vector_weight?: string; // 向量检索权重（仅 hybrid 时使用）
  };
  files?: KnowledgeBaseFile[];
}

interface EmbeddingModel {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
}
export default function KnowledgeBaseDetailPage({
  knowledgebase_id,
  setActiveTab,
}: {
  knowledgebase_id: string;
  setActiveTab: (tab: string) => void;
}) {
  const [knowledgebase, setKnowledgeBase] = useState<KnowledgeBase>(); // 知识库列表
  const [knowledgebasesloading, setKnowledgeBasesLoading] = useState(true); // 加载状态
  const [knowledgebasesrror, setKnowledgeBasesError] = useState(""); // 错误信息
  const [embeddingmodels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(""); // 错误信息
  const [editknowledgebase, setEditKnowledgeBase] = useState<KnowledgeBase>(); // 编辑的知识库
  const [toastState, setToastState] = useState({
    open: false,
    title: "",
    description: "",
    variant: "default" as "default" | "destructive",
  });
  // 递归更新嵌套对象
  const updateNestedObject = (
    obj: Record<string, any>,
    keys: string[],
    val: any,
  ): any => {
    const [currentKey, ...rest] = keys;
    const value = Array.isArray(val) ? val[0] : val;

    if (rest.length === 0) {
      return {
        ...obj,
        [currentKey]: value,
      };
    }

    return {
      ...obj,
      [currentKey]: updateNestedObject(obj[currentKey], rest, value),
    };
  };

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const [embRes] = await Promise.all([
          fetch(`http://localhost:${port}/v1/config/embeddings`),
        ]);

        const embData = (await embRes.json())?.data || [];
        console.log("embData", embData);
        setEmbeddingModels([...embData]);
      } catch (err: any) {
        setModelError(err || "加载失败");
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, []);
  const handleEditInputChange = (
    e:
      | React.ChangeEvent<HTMLInputElement>
      | React.ChangeEvent<HTMLTextAreaElement>,
  ) => {
    const { id, value } = e.target;
    setEditKnowledgeBase((prev) => {
      if (!prev) return prev;
      const path = id.split(".");
      return {
        ...prev,
        ...updateNestedObject(prev, path, value),
      };
    });
    console.log("编辑输入变化:", id, value);
    console.log("当前编辑的知识库状态:", editknowledgebase);
  };

  const handleEditFieldChange =
    (fieldPath: string) => (value: string | boolean | number[]) => {
      setEditKnowledgeBase((prev) => {
        if (!prev) return prev;

        const path = fieldPath.split(".");

        return updateNestedObject(prev, path, value);
      });
      console.log("当前编辑的知识库状态:", editknowledgebase);
    };

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const res = await fetch(
          `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}`,
        );
        if (!res.ok) throw new Error("获取知识库列表失败");
        const json_data = await res.json();
        const kb_data = json_data.data;

        // TODO：获取每个知识库的文件列表
        // // 并行获取每个知识库的文件列表
        // const knowledgeBasesWithFiles = await Promise.all(
        //   data.map(async (kb) => {
        //     try {
        //       // 模拟 API 请求：/v1/knowledgebases/${kb.id}/files
        //       const files = await fetch(`http://localhost:${port}/v1/knowledgebases/${kb.id}/files`);
        //       return { ...kb, files }; // 合并文件列表
        //     } catch (err) {
        //       console.error(`获取知识库 ${kb.id} 文件失败`, err);
        //       return { ...kb, files: [] }; // 失败时返回空数组
        //     }
        //   })
        // );

        // const data = [
        //     { id: '1', name: '产品文档库', description: '包含所有产品技术规格与使用指南' },
        //     { id: '2', name: '技术白皮书', description: '深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计' },
        //     { id: '3', name: '用户指南', description: '从入门到精通的全流程操作手册' },
        //     { id: '4', name: 'API 文档', description: 'RESTful 接口规范与示例' }
        // ]

        setKnowledgeBase(kb_data); // 更新状态
        console.log("知识库详情数据:", kb_data);
        setEditKnowledgeBase(kb_data || undefined); // 设置编辑的知识库
      } catch (err: any) {
        setKnowledgeBasesError(err || "加载失败");
      } finally {
        setKnowledgeBasesLoading(false);
      }
    };
    fetchConfigs();
  }, []);

  if (!knowledgebase) {
    return <div className="p-6">加载中...</div>;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log("更新知识库:", editknowledgebase);
    try {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(
        `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(editknowledgebase), // 包装为数组
        },
      );
      if (!res.ok) throw new Error(`修改 ${knowledgebase_id} 配置失败`);
      setToastState({
        open: true,
        title: `知识库${knowledgebase_id} 配置已修改`,
        description: "修改的模型配置已成功保存",
        variant: "default",
      });
      console.log(`update ${knowledgebase_id}`, editknowledgebase);
    } catch (err: any) {
      setToastState({
        open: true,
        title: "知识库配置修改失败",
        description: err.message || "请检查网络或重试",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="mb-6 flex items-center gap-2">
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() => setActiveTab(`/knowledgebase`)}
                    >
                      知识库
                    </Button>
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem>
                  <BreadcrumbPage>{knowledgebase.name}</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-4">
        <Tabs defaultValue="details">
          <TabsList className="py-4 bg-muted rounded-lg flex-none">
            <TabsTrigger value="details" className="p-4">
              文件列表
            </TabsTrigger>
            <TabsTrigger value="settings" className="p-4">
              知识库设置
            </TabsTrigger>
            <TabsTrigger value="retrieval_test" className="p-4">
              检索测试
            </TabsTrigger>
          </TabsList>
          <TabsContent value="details" className="py-4">
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>知识库文件列表</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-4">名称：{knowledgebase.name}</p>
                <p className="text-muted-foreground mb-4">
                  描述：{knowledgebase.description}
                </p>

                {knowledgebase.files && knowledgebase.files.length > 0 ? (
                  <>
                    <h3 className="text-lg font-semibold mt-6 mb-3">
                      文件列表
                    </h3>
                    <ScrollArea className="h-[400px] rounded-md border">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>文件名</TableHead>
                            <TableHead>文件格式</TableHead>
                            <TableHead>文件大小</TableHead>
                            <TableHead>状态</TableHead>
                            <TableHead>上传时间</TableHead>
                            <TableHead>操作</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {knowledgebase.files.map((file) => (
                            <TableRow key={file.id}>
                              <TableCell>
                                <Button
                                  variant="link"
                                  className="font-medium text-blue-600"
                                >
                                  {file.name}
                                </Button>
                              </TableCell>
                              <TableCell>{file.type}</TableCell>
                              <TableCell>{file.size}</TableCell>
                              <TableCell>
                                {file.status === "pending" ? (
                                  <div className="flex items-center text-yellow-500">
                                    <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                                    解析中
                                  </div>
                                ) : file.status === "done" ? (
                                  <div className="flex items-center text-green-500">
                                    <CheckCircle className="mr-1 h-4 w-4" />
                                    解析完成
                                  </div>
                                ) : (
                                  <span>{file.status}</span> // 兜底显示原始状态
                                )}
                              </TableCell>
                              <TableCell>{file.uploadedAt}</TableCell>
                              <TableCell>
                                <PreviewButton
                                  kbId={knowledgebase_id}
                                  fileId={file.id}
                                />
                                <Button
                                  variant="link"
                                  className="text-sm text-blue-600"
                                >
                                  查看切片
                                </Button>
                                <Button
                                  variant="link"
                                  className="text-sm text-blue-600"
                                >
                                  删除
                                </Button>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  </>
                ) : (
                  <p className="text-muted-foreground">暂无文件</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="settings" className="py-4">
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>知识库配置编辑</CardTitle>
              </CardHeader>
              <CardContent>
                {/* 基础信息 */}
                <div className="space-y-2 p-4 pt-0 border-b">
                  <h3 className="font-semibold pb-2">基础信息</h3>
                  <div className="space-y-2">
                    <Label htmlFor="name">
                      知识库名称 <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      id="name"
                      value={editknowledgebase?.name}
                      onChange={handleEditInputChange}
                      placeholder="请输入知识库名称"
                      required
                    />
                    <p className="text-sm text-muted-foreground">
                      例如："产品文档库"、"技术白皮书"
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="description">描述</Label>
                    <Textarea
                      id="description"
                      value={editknowledgebase?.description}
                      onChange={handleEditInputChange}
                      placeholder="描述知识库内容（可选）"
                      rows={3}
                    />
                  </div>
                </div>

                {/* 切片配置 */}
                <div className="space-y-2 p-4 border-b">
                  <h3 className="font-semibold pb-2">文档切片配置</h3>
                  <div className="grid grid-cols-4 gap-8">
                    <div className="space-y-2 col-span-1">
                      <Label htmlFor="parserType">
                        切片类型 (parser_type){" "}
                        <span className="text-destructive">*</span>
                      </Label>
                      <Select
                        value={
                          editknowledgebase?.chunk_config.parser_type || "Token"
                        }
                        onValueChange={handleEditFieldChange(
                          "chunk_config.parser_type",
                        )}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="请选择切片类型" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectGroup>
                            <SelectItem value="Token">Token</SelectItem>
                            <SelectItem value="Sentence">Sentence</SelectItem>
                            <SelectItem value="Paragraph">Paragraph</SelectItem>
                            <SelectItem value="Semantic">Semantic</SelectItem>
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                      <p className="text-sm text-muted-foreground">
                        推荐值：Sentence
                      </p>
                    </div>

                    <div className="space-y-2 col-span-1">
                      <Label htmlFor="separator">
                        切片标识符 (separator){" "}
                        <span className="text-destructive">*</span>
                      </Label>
                      <Input
                        type="text"
                        id="chunk_config.separator"
                        value={editknowledgebase?.chunk_config.separator}
                        onChange={handleEditInputChange}
                        required
                      />
                      <p className="text-sm text-muted-foreground">
                        推荐值：\n\n
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-4 gap-8">
                    <div className="space-y-2">
                      <Label htmlFor="chunkSize">
                        切片大小 (chunk_size){" "}
                        <span className="text-destructive">*</span>
                      </Label>
                      <Input
                        type="number"
                        id="chunk_config.chunk_size"
                        value={editknowledgebase?.chunk_config.chunk_size}
                        onChange={handleEditInputChange}
                        min="100"
                        max="1000"
                        required
                      />
                      <p className="text-sm text-muted-foreground">
                        推荐值：512
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="chunkOverlap">
                        切片重叠大小 (chunk_overlap)
                      </Label>
                      <Input
                        type="number"
                        id="chunk_config.chunk_overlap"
                        value={editknowledgebase?.chunk_config.chunk_overlap}
                        onChange={handleEditInputChange}
                        min="0"
                        max="200"
                      />
                      <p className="text-sm text-muted-foreground">
                        推荐值：50
                      </p>
                    </div>
                  </div>
                </div>

                {/* 索引及检索配置 */}
                <div className="space-y-2 p-4 border-b">
                  <h3 className="font-semibold pb-2">索引及检索设置</h3>
                  <div className="grid grid-cols-6 space-y-2">
                    <Label htmlFor="embeddingModel" className="col-span-1">
                      向量模型 (embedding_model){" "}
                      <span className="text-destructive">*</span>
                    </Label>
                    <div className="col-span-2">
                      <Select
                        value={
                          editknowledgebase?.embedding_model || "BAAI/bge-m3"
                        }
                        onValueChange={handleEditFieldChange("embedding_model")}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="请选择向量类型" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectGroup>
                            {embeddingmodels.map((model) => (
                              <SelectItem
                                key={model.id}
                                value={model.model_name}
                              >
                                {model.model_name}
                              </SelectItem>
                            ))}
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid grid-cols-6 space-y-2">
                    <Label className="col-span-1">检索设置</Label>
                    <div className="col-span-3 ">
                      <div className="space-y-2 pt-4 pb-2">
                        <ToggleGroup
                          type="single"
                          value={
                            editknowledgebase?.retrieval_config.retrieval_mode
                          }
                          onValueChange={handleEditFieldChange(
                            "retrieval_config.retrieval_mode",
                          )}
                          variant="outline"
                          className="flex gap-x-4 overflow-visible"
                        >
                          <ToggleGroupItem
                            value="vector"
                            aria-label="向量检索"
                            className="!rounded-full px-6 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
                          >
                            <ScanSearch />
                            向量检索
                          </ToggleGroupItem>
                          <ToggleGroupItem
                            value="fulltext"
                            aria-label="全文检索"
                            className="!rounded-full px-6 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
                          >
                            <TextSearch />
                            全文检索
                          </ToggleGroupItem>
                          <ToggleGroupItem
                            value="hybrid"
                            aria-label="混合检索"
                            className="!rounded-full px-6 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
                          >
                            <SearchCode />
                            混合检索
                          </ToggleGroupItem>
                        </ToggleGroup>
                      </div>
                      {/* 动态参数配置区域 */}
                      <div className="space-y-4 pt-2">
                        {editknowledgebase?.retrieval_config.retrieval_mode ===
                          "vector" && (
                          <div>
                            <div className="grid grid-cols-5 space-y-4 ">
                              <Label htmlFor="topk" className="col-span-1">
                                Top-K 值
                              </Label>
                              <Input
                                type="number"
                                id="retrieval_config.top_k"
                                value={
                                  editknowledgebase?.retrieval_config.top_k
                                }
                                onChange={handleEditInputChange}
                                min="1"
                                max="100"
                                required
                                className="col-span-2 w-full"
                              />
                              <p className="col-span-2 text-sm text-muted-foreground pt-2 pl-6">
                                推荐值：5-20，最大支持100条结果
                              </p>
                            </div>
                            <div className="grid grid-cols-5 space-y-4 ">
                              <Label htmlFor="threshold" className="col-span-1">
                                向量相似度分数阈值
                              </Label>
                              <Input
                                type="number"
                                id="retrieval_config.similarity_threshold"
                                value={
                                  editknowledgebase?.retrieval_config
                                    .similarity_threshold
                                }
                                onChange={handleEditInputChange}
                                step="0.05"
                                min="0"
                                max="1"
                                required
                                className="col-span-2 w-full"
                              />
                              <p className="col-span-2 text-sm text-muted-foreground pt-2 pl-6">
                                推荐值：0.8
                              </p>
                            </div>
                            <Label
                              htmlFor="embeddingModel"
                              className="col-span-1"
                            >
                              重排序模型 (rerank_model){" "}
                              <span className="text-destructive">*</span>
                            </Label>
                            <div className="col-span-2">
                              <Select
                                defaultValue={
                                  editknowledgebase.retrieval_config
                                    .rerank_model
                                }
                                onValueChange={handleEditFieldChange(
                                  "retrieval_config.rerank_model",
                                )}
                              >
                                <SelectTrigger className="w-full">
                                  <SelectValue placeholder="请选择重排序模型" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectGroup>
                                    <SelectItem value="none">
                                      NO RERANK
                                    </SelectItem>
                                    <SelectItem value="BAAI/bge-reranker-base">
                                      BAAI/bge-reranker-base
                                    </SelectItem>
                                    <SelectItem value="BAAI/bge-reranker-large">
                                      BAAI/bge-reranker-large
                                    </SelectItem>
                                    <SelectItem value="qwen3">qwen3</SelectItem>
                                  </SelectGroup>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="col-span-1 text-sm text-muted-foreground">
                              <p className="pt-2 pl-6">
                                推荐值： BAAI/bge-reranker-base
                              </p>
                            </div>
                          </div>
                        )}
                        {editknowledgebase?.retrieval_config.retrieval_mode ===
                          "fulltext" && (
                          <div>
                            <div className="grid grid-cols-5 space-y-4 ">
                              <Label htmlFor="topk" className="col-span-1">
                                Top-K 值
                              </Label>
                              <Input
                                type="number"
                                id="retrieval_config.top_k"
                                value={
                                  editknowledgebase?.retrieval_config.top_k
                                }
                                onChange={handleEditInputChange}
                                min="1"
                                max="100"
                                required
                                className="col-span-2 w-full"
                              />
                              <p className="col-span-2 text-sm text-muted-foreground pt-2 pl-6">
                                推荐值：5-20，最大支持100条结果
                              </p>
                            </div>
                            <div className="grid grid-cols-5 space-y-4 ">
                              <Label htmlFor="threshold" className="col-span-1">
                                全文相似度分数阈值
                              </Label>
                              <Input
                                type="number"
                                id="retrieval_config.similarity_threshold"
                                value={
                                  editknowledgebase?.retrieval_config
                                    .similarity_threshold
                                }
                                onChange={handleEditInputChange}
                                step="0.05"
                                min="0"
                                max="1"
                                required
                                className="col-span-2 w-full"
                              />
                              <p className="col-span-2 text-sm text-muted-foreground pt-2 pl-6">
                                推荐值：0.8
                              </p>
                            </div>
                            <div className="flex items-start gap-3 pt-2">
                              <Label
                                htmlFor="embeddingModel"
                                className="col-span-1"
                              >
                                重排序模型 (rerank_model){" "}
                                <span className="text-destructive">*</span>
                              </Label>
                              <div className="col-span-2">
                                <Select
                                  defaultValue={
                                    editknowledgebase.retrieval_config
                                      .rerank_model
                                  }
                                  onValueChange={handleEditFieldChange(
                                    "retrieval_config.rerank_model",
                                  )}
                                >
                                  <SelectTrigger className="w-full">
                                    <SelectValue placeholder="请选择重排序模型" />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectGroup>
                                      <SelectItem value="none">
                                        NO RERANK
                                      </SelectItem>
                                      <SelectItem value="BAAI/bge-reranker-base">
                                        BAAI/bge-reranker-base
                                      </SelectItem>
                                      <SelectItem value="BAAI/bge-reranker-large">
                                        BAAI/bge-reranker-large
                                      </SelectItem>
                                      <SelectItem value="qwen3">
                                        qwen3
                                      </SelectItem>
                                    </SelectGroup>
                                  </SelectContent>
                                </Select>
                              </div>

                              <div className="col-span-1 text-sm text-muted-foreground">
                                <p className="pt-2 pl-6">
                                  推荐值： BAAI/bge-reranker-base
                                </p>
                              </div>
                            </div>
                          </div>
                        )}
                        {editknowledgebase?.retrieval_config.retrieval_mode ===
                          "hybrid" && (
                          <div className="space-y-6">
                            <div className="grid grid-cols-5 gap-4 pt-2">
                              <Label
                                htmlFor="embeddingWeight"
                                className="col-span-1"
                              >
                                向量检索权重
                              </Label>
                              <Slider
                                id="retrieval_config.vector_weight"
                                min={0}
                                max={1}
                                step={0.1}
                                value={[
                                  parseFloat(
                                    String(
                                      editknowledgebase?.retrieval_config
                                        .vector_weight,
                                    ),
                                  ),
                                ]}
                                onValueChange={handleEditFieldChange(
                                  "retrieval_config.vector_weight",
                                )}
                                className="col-span-2"
                              />
                              <span className="w-12 text-right text-sm font-medium col-span-1">
                                {
                                  editknowledgebase?.retrieval_config
                                    .vector_weight
                                }
                              </span>
                            </div>
                            <div className="grid grid-cols-5 space-y-4 ">
                              <Label htmlFor="topk" className="col-span-1">
                                Top-K 值
                              </Label>
                              <Input
                                type="number"
                                id="retrieval_config.top_k"
                                value={
                                  editknowledgebase?.retrieval_config.top_k
                                }
                                onChange={handleEditInputChange}
                                min="1"
                                max="100"
                                required
                                className="col-span-2 w-full"
                              />
                              <p className="col-span-2 text-sm text-muted-foreground pt-2 pl-6">
                                推荐值：5-20，最大支持100条结果
                              </p>
                            </div>
                            <div className="grid grid-cols-5 space-y-4 ">
                              <Label htmlFor="threshold" className="col-span-1">
                                混合相似度分数阈值
                              </Label>
                              <Input
                                type="number"
                                id="retrieval_config.similarity_threshold"
                                value={
                                  editknowledgebase?.retrieval_config
                                    .similarity_threshold
                                }
                                onChange={handleEditInputChange}
                                step="0.05"
                                min="0"
                                max="1"
                                required
                                className="col-span-2 w-full"
                              />
                              <p className="col-span-2 text-sm text-muted-foreground pt-2 pl-6">
                                推荐值：0.8
                              </p>
                            </div>
                            <div className="flex items-start gap-3 pt-2">
                              <Label
                                htmlFor="embeddingModel"
                                className="col-span-1"
                              >
                                重排序模型 (rerank_model){" "}
                                <span className="text-destructive">*</span>
                              </Label>
                              <div className="col-span-2">
                                <Select
                                  defaultValue={
                                    editknowledgebase.retrieval_config
                                      .rerank_model
                                  }
                                  onValueChange={handleEditFieldChange(
                                    "retrieval_config.rerank_model",
                                  )}
                                >
                                  <SelectTrigger className="w-full">
                                    <SelectValue placeholder="请选择重排序模型" />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectGroup>
                                      <SelectItem value="none">
                                        NO RERANK
                                      </SelectItem>
                                      <SelectItem value="BAAI/bge-reranker-base">
                                        BAAI/bge-reranker-base
                                      </SelectItem>
                                      <SelectItem value="BAAI/bge-reranker-large">
                                        BAAI/bge-reranker-large
                                      </SelectItem>
                                      <SelectItem value="qwen3">
                                        qwen3
                                      </SelectItem>
                                    </SelectGroup>
                                  </SelectContent>
                                </Select>
                              </div>

                              <div className="col-span-1 text-sm text-muted-foreground">
                                <p className="pt-2 pl-6">
                                  推荐值： BAAI/bge-reranker-base
                                </p>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-center space-x-6">
                <Button
                  type="submit"
                  className="px-10 py-3 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
                  onClick={handleSubmit}
                >
                  保存设置
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
        <Toast.Root
          open={toastState.open}
          onOpenChange={(open) => setToastState((prev) => ({ ...prev, open }))}
          className={`grid grid-cols-[auto_1fr] items-center gap-x-4 rounded-md border px-4 py-6 shadow-lg transition-all data-[state=open]:animate-slideIn data-[state=closed]:animate-fadeOut ${
            toastState.variant === "destructive"
              ? "border-red-500 bg-red-50 text-red-900"
              : "border-gray-200 bg-white text-gray-900"
          }`}
        >
          <Toast.Description className="pl-4 text-sm font-medium">
            {toastState.description}
          </Toast.Description>
          <Toast.Action
            altText="关闭"
            onClick={() => setToastState((prev) => ({ ...prev, open: false }))}
          >
            ×
          </Toast.Action>
        </Toast.Root>

        {/* 触发 Toast 的隐藏容器 */}
        <Toast.Viewport className="fixed bottom-0 right-0 z-[100] m-0 flex w-96 flex-col gap-2 p-6" />
      </div>
    </div>
  );
}
