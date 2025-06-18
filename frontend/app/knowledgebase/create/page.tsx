"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  ArrowLeft,
  SearchCode,
  Save,
  TextSearch,
  ScanSearch,
  SkipBack,
} from "lucide-react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Slider } from "@/components/ui/slider";

export default function KnowledgeBaseCreatePage({
  setActiveTab,
}: {
  setActiveTab: (tab: string) => void;
}) {
  const [formData, setFormData] = useState({
    kb_name: "",
    kb_description: "",
    chunk_config: {
      parser_type: "Sentence",
      separator: "\n\n",
      chunk_size: "512",
      chunk_overlap: "50",
    },
    embedding_config: {
      model_name: "bge-m3",
    },
    retrieval_config: {
      index_type: "vector",
      top_k: "5",
      similarity_threshold: "0.8",
      enable_rerank: false,
      vector_weight: "0.7",
    },
  });

  const [indexType, setIndexType] = useState("vector");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // 模拟创建知识库请求
    console.log("创建知识库:", formData);
    // const formData = {
    //     "kb_name": "pairag_QA_v1",
    //     "kb_description": "eqwewq",
    //     "chunk_config": {
    //         "parser_type": "Sentence",
    //         "separator": "\\n\\n",
    //         "chunk_size": "512",
    //         "chunk_overlap": "50"
    //     },
    //     "embedding_config": {
    //         "model_name": "bge-m3"
    //     },
    //     "retrieval_config": {
    //         "index_type": "hybrid",
    //         "top_k": "5",
    //         "similarity_threshold": "0.8",
    //         "enable_rerank": true,
    //         "vector_weight": "0.5"
    //     }
    // }
    // 实际应调用 API: POST /v1/knowledgebase/create
    // params: formData
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="mb-2 flex items-center gap-2">
            {/* 面包屑导航 */}
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
                  <BreadcrumbPage>新建知识库</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
          <div className="mb-2 flex items-center gap-2">
            <Button
              variant="outline"
              className="h-8 w-8"
              onClick={() => setActiveTab("/knowledgebase")}
            >
              <ArrowLeft />
            </Button>
            <h1 className="text-2xl font-bold">新建知识库</h1>
          </div>
        </div>
      </div>

      {/* 可滚动内容区域 */}
      <div className="flex-1 overflow-y-auto">
        <Card className="border-none px-4">
          <CardHeader>
            <CardTitle className="text-xl font-semibold">知识库配置</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* 基础信息 */}
              <div className="space-y-2 p-4 border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-semibold pb-2">基础信息</h3>
                <div className="space-y-2">
                  <Label htmlFor="name">
                    知识库名称 <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="name"
                    value={formData.kb_name}
                    onChange={(e) =>
                      setFormData({ ...formData, kb_name: e.target.value })
                    }
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
                    value={formData.kb_description}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        kb_description: e.target.value,
                      })
                    }
                    placeholder="描述知识库内容（可选）"
                    rows={3}
                  />
                </div>
              </div>

              {/* 切片配置 */}
              <div className="space-y-2 p-4 border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-semibold pb-2">文档切片配置</h3>
                <div className="grid grid-cols-4 gap-8">
                  <div className="space-y-2">
                    <Label htmlFor="parserType">
                      切片类型 (parser_type){" "}
                      <span className="text-destructive">*</span>
                    </Label>
                    <Select>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="请选择切片类型" />
                      </SelectTrigger>
                      <SelectContent
                        id="parserType"
                        defaultValue={formData.chunk_config.parser_type}
                      >
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

                  <div className="space-y-2">
                    <Label htmlFor="separator">
                      切片标识符 (separator){" "}
                      <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      type="text"
                      id="separator"
                      value={formData.chunk_config.separator}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          chunk_config: {
                            ...formData.chunk_config,
                            separator: e.target.value,
                          },
                        })
                      }
                      required
                    />
                    <p className="text-sm text-muted-foreground">
                      推荐值：\n\n
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="chunkSize">
                      切片大小 (chunk_size){" "}
                      <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      type="number"
                      id="chunkSize"
                      value={formData.chunk_config.chunk_size}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          chunk_config: {
                            ...formData.chunk_config,
                            chunk_size: e.target.value,
                          },
                        })
                      }
                      min="100"
                      max="1000"
                      required
                    />
                    <p className="text-sm text-muted-foreground">推荐值：512</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="chunkOverlap">
                      切片重叠大小 (chunk_overlap)
                    </Label>
                    <Input
                      type="number"
                      id="chunkOverlap"
                      value={formData.chunk_config.chunk_overlap}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          chunk_config: {
                            ...formData.chunk_config,
                            chunk_overlap: e.target.value,
                          },
                        })
                      }
                      min="0"
                      max="200"
                    />
                    <p className="text-sm text-muted-foreground">推荐值：50</p>
                  </div>
                </div>
              </div>

              {/* 索引及检索配置 */}
              <div className="space-y-2 p-4 border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-semibold pb-2">索引及检索设置</h3>
                <div className="grid grid-cols-6 space-y-2">
                  <Label htmlFor="embeddingModel" className="col-span-1">
                    向量模型 (embedding_model){" "}
                    <span className="text-destructive">*</span>
                  </Label>
                  <div className="col-span-2">
                    <Select>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="请选择向量类型" />
                      </SelectTrigger>
                      <SelectContent
                        id="embeddingModel"
                        defaultValue={formData.embedding_config.model_name}
                      >
                        <SelectGroup>
                          <SelectItem value="bge-m3">bge-m3</SelectItem>
                          <SelectItem value="text-embedding-v1">
                            text-embedding-v1
                          </SelectItem>
                          <SelectItem value="qwen3">qwen3</SelectItem>
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="col-span-1 text-sm text-muted-foreground">
                    <p className="pt-2 pl-6">推荐值：bge-m3</p>
                  </div>
                </div>
                <div className="grid grid-cols-6 space-y-2">
                  <Label className="col-span-1">检索设置</Label>
                  <div className="col-span-3 ">
                    <div className="space-y-2 pt-4 pb-2">
                      <ToggleGroup
                        type="single"
                        value={formData.retrieval_config.index_type}
                        onValueChange={(value) => {
                          // 同时更新 indexType 和 formData.retrieval_config.index_type
                          setIndexType(value);
                          setFormData({
                            ...formData,
                            retrieval_config: {
                              ...formData.retrieval_config,
                              index_type: value,
                            },
                          });
                        }}
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
                      {indexType === "vector" && (
                        <div>
                          <div className="grid grid-cols-5 space-y-4 ">
                            <Label htmlFor="topk" className="col-span-1">
                              Top-K 值
                            </Label>
                            <Input
                              type="number"
                              id="topk"
                              value={formData.retrieval_config.top_k}
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    top_k: e.target.value,
                                  },
                                })
                              }
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
                              id="threshold"
                              value={
                                formData.retrieval_config.similarity_threshold
                              }
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    similarity_threshold: e.target.value,
                                  },
                                })
                              }
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
                            <Checkbox
                              id="terms-2"
                              checked={formData.retrieval_config.enable_rerank}
                              onCheckedChange={(checkedState) => {
                                // 将 CheckedState 转换为 boolean
                                const isChecked = checkedState === true;
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    enable_rerank: isChecked,
                                  },
                                });
                              }}
                            />
                            <div className="grid gap-2">
                              <Label htmlFor="terms-2">启用 Rerank 模型</Label>
                              <p className="text-muted-foreground text-sm">
                                默认使用bge-ranker模型进行重排序
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      {indexType === "fulltext" && (
                        <div>
                          <div className="grid grid-cols-5 space-y-4 ">
                            <Label htmlFor="topk" className="col-span-1">
                              Top-K 值
                            </Label>
                            <Input
                              type="number"
                              id="topk"
                              value={formData.retrieval_config.top_k}
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    top_k: e.target.value,
                                  },
                                })
                              }
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
                              id="threshold"
                              value={
                                formData.retrieval_config.similarity_threshold
                              }
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    similarity_threshold: e.target.value,
                                  },
                                })
                              }
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
                            <Checkbox
                              id="terms-2"
                              checked={formData.retrieval_config.enable_rerank}
                              onCheckedChange={(checkedState) => {
                                // 将 CheckedState 转换为 boolean
                                const isChecked = checkedState === true;
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    enable_rerank: isChecked,
                                  },
                                });
                              }}
                            />
                            <div className="grid gap-2">
                              <Label htmlFor="terms-2">启用 Rerank 模型</Label>
                              <p className="text-muted-foreground text-sm">
                                默认使用bge-ranker模型进行重排序
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      {indexType === "hybrid" && (
                        <div className="space-y-6">
                          <div className="grid grid-cols-5 gap-4 pt-2">
                            <Label
                              htmlFor="embeddingWeight"
                              className="col-span-1"
                            >
                              向量检索权重
                            </Label>
                            <Slider
                              id="embeddingWeight"
                              min={0}
                              max={1}
                              step={0.1}
                              value={[
                                parseFloat(
                                  formData.retrieval_config.vector_weight,
                                ),
                              ]}
                              onValueChange={(value) =>
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    vector_weight: value[0].toString(),
                                  },
                                })
                              }
                              className="col-span-2"
                            />
                            <span className="w-12 text-right text-sm font-medium col-span-1">
                              {formData.retrieval_config.vector_weight}
                            </span>
                          </div>
                          <div className="grid grid-cols-5 space-y-4 ">
                            <Label htmlFor="topk" className="col-span-1">
                              Top-K 值
                            </Label>
                            <Input
                              type="number"
                              id="topk"
                              value={formData.retrieval_config.top_k}
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    top_k: e.target.value,
                                  },
                                })
                              }
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
                              id="threshold"
                              value={
                                formData.retrieval_config.similarity_threshold
                              }
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    similarity_threshold: e.target.value,
                                  },
                                })
                              }
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
                            <Checkbox
                              id="terms-2"
                              checked={formData.retrieval_config.enable_rerank}
                              onCheckedChange={(checkedState) => {
                                // 将 CheckedState 转换为 boolean
                                const isChecked = checkedState === true;
                                setFormData({
                                  ...formData,
                                  retrieval_config: {
                                    ...formData.retrieval_config,
                                    enable_rerank: isChecked,
                                  },
                                });
                              }}
                            />
                            <div className="grid gap-2">
                              <Label htmlFor="terms-2">启用 Rerank 模型</Label>
                              <p className="text-muted-foreground text-sm">
                                默认使用bge-ranker模型进行重排序
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex justify-center gap-4 pb-4">
                <Button
                  type="button"
                  variant="outline"
                  className="w-40"
                  onClick={() => setActiveTab("/knowledgebase")}
                >
                  <SkipBack />
                  取消
                </Button>
                <Button type="submit" className="w-40">
                  {" "}
                  <Save />
                  创建
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
