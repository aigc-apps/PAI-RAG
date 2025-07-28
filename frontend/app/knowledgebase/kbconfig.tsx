"use client";
import React, { useState, useEffect, FC } from "react";
import {
  Card,
  CardContent,
  CardFooter,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
  AlertCircleIcon,
  CirclePlus,
  Trash2Icon,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface EmbeddingModel {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
}

// 元数据配置
export interface MetadataConfig {
  id: string;
  name: string;
  value_type: string;
  description: string;
}

export interface KbConfig {
  id: string;
  name: string;
  description: string;
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
    similarity_threshold: number; // 相似度分数阈值
    rerank_model: string; // rerank模型名称
    vector_weight?: number; // 向量检索权重（仅 hybrid 时使用）
  };
}

interface KbConfigProps {
  kbConfig: KbConfig;
  metadataConfigs: MetadataConfig[];
  isCreate: boolean;
  onSaveSuccess: (kb: KbConfig) => void;
  onCancel: () => void;
}

// 知识库配置卡片
export const KbConfigCard: FC<KbConfigProps> = ({
  kbConfig,
  metadataConfigs,
  isCreate,
  onSaveSuccess,
  onCancel,
}) => {
  const [kb, setKb] = useState<KbConfig>(kbConfig);
  const [indexType, setIndexType] = useState("vector");
  const [metadataOpen, setMetadataOpen] = useState(false);
  const [metadataName, setmetadataName] = useState("");
  const [metadataValueType, setMetadataValueType] = useState("string");
  const [metadataDesc, setMetadataDesc] = useState("");
  const [metadataError, setMetadataError] = useState("");
  const [embeddingmodels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(""); // 错误信息
  const [saveErrorMsg, setSaveErrorMsg] = useState(""); // 保存KB错误信息
  const [metadata_configs, setMetadataConfigs] =
    useState<MetadataConfig[]>(metadataConfigs);

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
        const [embRes] = await Promise.all([
          fetch(`${API_BASE}/v1/config/embeddings`),
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log("保存知识库结果:", kb);
    const API_BASE =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
    const submit_url = isCreate
      ? `${API_BASE}/v1/config/knowledgebases`
      : `${API_BASE}/v1/config/knowledgebases/${kb.id}`;
    const updateMethod = isCreate ? "POST" : "PATCH";
    try {
      const res = await fetch(submit_url, {
        method: updateMethod,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(kb), // 包装为数组
      });

      if (!res.ok) throw new Error(`保存知识库失败: ${await res.text()}`);
      const jsondata = await res.json();
      setSaveErrorMsg("");
      onSaveSuccess(jsondata.data as KbConfig);
    } catch (err: any) {
      console.log("保存知识库失败", err.message);
      setSaveErrorMsg(err.message);
    }
  };

  const handleRemoveMetadataEntry = async (id: string) => {
    if (metadata_configs != null) {
      const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
      const metadata_url = `${API_BASE}/v1/config/knowledgebases/${kb.id}/metadata/${id}`;
      try {
        const res = await fetch(metadata_url, {
          method: "DELETE",
        });
        if (!res.ok) throw new Error(`删除metadata失败: ${await res.text()}`);

        const updated_metadata_configs = metadata_configs.filter(
          (config: any) => config.id !== id,
        );
        setMetadataConfigs(updated_metadata_configs);

        console.log("删除的元数据：", id);
      } catch (err: any) {
        console.log("删除元数据失败。", err.message);
      }
    }
  };

  const handleAddMetadataConfig = async () => {
    if (!metadataName) {
      setMetadataError("必须填入元数据名称。");
      return;
    }
    let updated_metadata_configs = metadata_configs || [];

    if (
      updated_metadata_configs.some((config) => config.name === metadataName)
    ) {
      setMetadataError(`元数据名称 '${metadataName}' 已经存在.`);
      return;
    }

    const API_BASE =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
    const metadata_url = `${API_BASE}/v1/config/knowledgebases/${kb.id}/metadata`;
    try {
      const res = await fetch(metadata_url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kb_id: kb.id,
          name: metadataName,
          value_type: metadataValueType,
          description: metadataDesc,
        }), // 包装为数组
      });
      if (!res.ok) throw new Error(`保存metadata失败: ${await res.text()}`);
      const new_metadata_json = await res.json();
      const new_metadata = new_metadata_json.data as MetadataConfig;
      updated_metadata_configs.push(new_metadata);
      setMetadataConfigs(updated_metadata_configs);
      console.log("添加元数据成功.");
    } catch (err: any) {
      console.log("保存知识库失败", err.message);
      setSaveErrorMsg(err.message);
    } finally {
      setmetadataName("");
      setMetadataError("");
      setMetadataValueType("string");
      setMetadataDesc("");
      setMetadataOpen(false);
    }
  };

  function handleCancelMetadataConfig() {
    setmetadataName("");
    setMetadataError("");
    setMetadataValueType("string");
    setMetadataDesc("");
    console.log("清空metadata信息");
  }

  return (
    <div>
      <div>
        <Tabs defaultValue="basic_info">
          <TabsList>
            <TabsTrigger value="basic_info">基本信息</TabsTrigger>
            <TabsTrigger value="chunk_info">切片设置</TabsTrigger>
            <TabsTrigger value="retrieval_info">检索设置</TabsTrigger>
            <TabsTrigger value="metadata_info">元数据</TabsTrigger>
          </TabsList>
          <TabsContent value="basic_info">
            <Card>
              <CardHeader>
                <CardTitle>基本信息</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-6">
                <div className="space-y-2">
                  <Label htmlFor="name">
                    知识库名称 <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="name"
                    value={kb.name}
                    onChange={(e) =>
                      setKb((prev) => ({ ...prev, name: e.target.value }))
                    }
                    placeholder="请输入知识库名称"
                    required
                  />
                  <p className="text-sm text-muted-foreground">
                    例如："XX产品用户手册"、"IT操作说明"
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">描述</Label>
                  <Textarea
                    id="description"
                    value={kb.description}
                    onChange={(e) =>
                      setKb((prev) => ({
                        ...prev,
                        description: e.target.value,
                      }))
                    }
                    placeholder="描述知识库内容（可选）"
                    rows={3}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="chunk_info">
            <Card>
              <CardHeader>
                <CardTitle>切片设置</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-6">
                <div className="grid grid-cols-4 gap-8">
                  <div className="space-y-2">
                    <Label htmlFor="separator">
                      切片标识符 (separator){" "}
                      <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      type="text"
                      id="separator"
                      value={kb.chunk_config.separator}
                      onChange={(e) =>
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            separator: e.target.value,
                          },
                        }))
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
                      value={kb.chunk_config.chunk_size}
                      onChange={(e) =>
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            chunk_size: e.target.value,
                          },
                        }))
                      }
                      min="100"
                      max="2000"
                      required
                    />
                    <p className="text-sm text-muted-foreground">
                      推荐值: 1000
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="chunkOverlap">
                      切片重叠大小 (chunk_overlap)
                    </Label>
                    <Input
                      type="number"
                      id="chunkOverlap"
                      value={kb.chunk_config.chunk_overlap}
                      onChange={(e) =>
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            chunk_overlap: e.target.value,
                          },
                        }))
                      }
                      min="0"
                      max="200"
                    />
                    <p className="text-sm text-muted-foreground">推荐值: 50</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="retrieval_info">
            <Card>
              <CardHeader>
                <CardTitle>检索设置</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-6">
                <div className="grid grid-cols-6 space-y-2">
                  <Label htmlFor="embeddingModel" className="col-span-1">
                    向量模型 <span className="text-destructive">*</span>
                  </Label>
                  <div className="col-span-2">
                    <Select
                      defaultValue={kb.embedding_model}
                      onValueChange={(value) => {
                        setKb((prev) => ({ ...prev, embedding_model: value }));
                      }}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="请选择向量类型" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          {embeddingmodels.map((model) => (
                            <SelectItem key={model.id} value={model.model_id}>
                              {model.model_id}
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
                        value={kb.retrieval_config.retrieval_mode}
                        onValueChange={(value) => {
                          // 同时更新 indexType 和 formData.retrieval_config.index_type
                          setIndexType(value);
                          setKb((prev) => ({
                            ...prev,
                            retrieval_config: {
                              ...prev.retrieval_config,
                              retrieval_mode: value,
                            },
                          }));
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
                      <div>
                        <div className="grid grid-cols-5 space-y-4 ">
                          <Label htmlFor="topk" className="col-span-1">
                            Top-K 值
                          </Label>
                          <Input
                            type="number"
                            id="topk"
                            value={kb.retrieval_config.top_k}
                            onChange={(e) =>
                              setKb((prev) => ({
                                ...prev,
                                retrieval_config: {
                                  ...prev.retrieval_config,
                                  top_k: parseInt(e.target.value),
                                },
                              }))
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
                            相似度分数阈值
                          </Label>
                          <Input
                            type="number"
                            id="threshold"
                            value={kb.retrieval_config.similarity_threshold}
                            onChange={(e) =>
                              setKb((prev) => ({
                                ...prev,
                                retrieval_config: {
                                  ...prev.retrieval_config,
                                  similarity_threshold: parseFloat(
                                    e.target.value,
                                  ),
                                },
                              }))
                            }
                            step="0.05"
                            min="0"
                            max="1"
                            required
                            className="col-span-2 w-full"
                          />
                          <p className="col-span-2 text-sm text-muted-foreground pt-2 pl-6">
                            推荐值：0.4
                          </p>
                        </div>
                        <div className="grid grid-cols-6 space-y-2">
                          <Label
                            htmlFor="embeddingModel"
                            className="col-span-1"
                          >
                            重排序模型 (rerank_model){" "}
                            <span className="text-destructive">*</span>
                          </Label>
                          <div className="col-span-2">
                            <Select
                              defaultValue={kb.retrieval_config.rerank_model}
                              onValueChange={(value) => {
                                setKb((prev) => ({
                                  ...prev,
                                  retrieval_config: {
                                    ...prev.retrieval_config,
                                    rerank_model: value,
                                  },
                                }));
                              }}
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
                      </div>
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
                              value={[kb.retrieval_config.vector_weight || 0.7]}
                              onValueChange={(value) =>
                                setKb((prev) => ({
                                  ...prev,
                                  retrieval_config: {
                                    ...prev.retrieval_config,
                                    vector_weight: value[0],
                                  },
                                }))
                              }
                              className="col-span-2"
                            />
                            <span className="w-12 text-right text-sm font-medium col-span-1">
                              {kb.retrieval_config.vector_weight}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="metadata_info">
            <Card>
              <CardHeader>
                <CardTitle>元数据</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-6">
                <div className="grid space-y-2">
                  <div className="w-full">
                    <Table className="w-full">
                      {metadata_configs === null ||
                      metadata_configs.length == 0 ? (
                        <TableCaption>尚未配置元数据信息</TableCaption>
                      ) : (
                        <TableCaption>
                          已添加{metadata_configs.length}条元数据信息。{" "}
                        </TableCaption>
                      )}
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[100px]">
                            元数据名称(Key)
                          </TableHead>
                          <TableHead className="w-[100px]">
                            元数据类型
                          </TableHead>
                          <TableHead>元数据描述</TableHead>
                          <TableHead className="w-[100px] text-right">
                            <Dialog
                              open={metadataOpen}
                              onOpenChange={setMetadataOpen}
                            >
                              <DialogTrigger asChild>
                                <Button variant="outline">
                                  {" "}
                                  <CirclePlus /> 添加元数据
                                </Button>
                              </DialogTrigger>
                              <DialogContent className="sm:max-w-[425px]">
                                <DialogHeader>
                                  <DialogTitle>添加元数据</DialogTitle>
                                  <DialogDescription>
                                    请设定一个元数据名称（英文和数字），如city,
                                    category，用于在知识库内检索。
                                  </DialogDescription>
                                </DialogHeader>
                                <div className="grid gap-4">
                                  <div className="grid gap-3">
                                    <Label htmlFor="metadata_key">
                                      元数据名称
                                    </Label>
                                    <Input
                                      id="metadata_key"
                                      onChange={(e) =>
                                        setmetadataName(e.target.value)
                                      }
                                    />
                                  </div>
                                  <div className="grid gap-3">
                                    <Label htmlFor="metadata_value_type">
                                      值类型
                                    </Label>
                                    <Select
                                      defaultValue="string"
                                      onValueChange={(value) =>
                                        setMetadataValueType(value)
                                      }
                                    >
                                      <SelectTrigger className="w-[180px]">
                                        <SelectValue
                                          placeholder="选择值类型"
                                          defaultValue="string"
                                        />
                                      </SelectTrigger>
                                      <SelectContent>
                                        <SelectGroup>
                                          <SelectLabel>值类型</SelectLabel>
                                          <SelectItem value="string">
                                            String
                                          </SelectItem>
                                          <SelectItem value="number">
                                            Number
                                          </SelectItem>
                                          <SelectItem value="datetime">
                                            DateTime
                                          </SelectItem>
                                        </SelectGroup>
                                      </SelectContent>
                                    </Select>
                                  </div>
                                  <div className="grid gap-3">
                                    <Label htmlFor="metadata_desc">
                                      元数据描述
                                    </Label>
                                    <Input
                                      id="metadata_desc"
                                      placeholder="输入元数据相关描述。"
                                      onChange={(e) =>
                                        setMetadataDesc(e.target.value)
                                      }
                                    />
                                  </div>
                                </div>
                                {metadataError ? (
                                  <Alert variant="destructive">
                                    <AlertCircleIcon />
                                    <AlertTitle>无法保存metadata.</AlertTitle>
                                    <AlertDescription>
                                      <p>{metadataError}</p>
                                    </AlertDescription>
                                  </Alert>
                                ) : null}
                                <DialogFooter>
                                  <DialogClose asChild>
                                    <Button
                                      variant="outline"
                                      onClick={handleCancelMetadataConfig}
                                    >
                                      取消
                                    </Button>
                                  </DialogClose>
                                  <Button
                                    type="button"
                                    onClick={handleAddMetadataConfig}
                                  >
                                    保存
                                  </Button>
                                </DialogFooter>
                              </DialogContent>
                            </Dialog>
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {metadata_configs?.map((item, index) => (
                          <TableRow key={index}>
                            <TableCell className="font-medium">
                              {item.name}
                            </TableCell>
                            <TableCell>{item.value_type}</TableCell>
                            <TableCell>{item.description}</TableCell>
                            <TableCell className="text-right">
                              <Button
                                variant="secondary"
                                size="icon"
                                className="size-8"
                                onClick={() =>
                                  handleRemoveMetadataEntry(item.id)
                                }
                              >
                                <Trash2Icon />
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        <div className="block w-full">
          {saveErrorMsg !== "" && (
            <Alert variant="destructive">
              <AlertCircleIcon />
              <AlertDescription>
                <p>{saveErrorMsg}</p>
              </AlertDescription>
            </Alert>
          )}
        </div>
      </div>
      <div className="fixed bottom-0 inset-x-0 h-20 bg-white border-t left-64 flex justify-around items-center z-50 ">
        <div>
          {isCreate && (
            <div className="flex justify-center gap-4 pb-4">
              <Button
                type="button"
                variant="outline"
                className="w-40"
                onClick={() => onCancel()}
              >
                <SkipBack />
                取消
              </Button>
              <Button type="button" className="w-40" onClick={handleSubmit}>
                {" "}
                <Save />
                创建
              </Button>
            </div>
          )}
          {!isCreate && (
            <div className="flex justify-center gap-4 pb-4">
              <Button type="button" className="w-40" onClick={handleSubmit}>
                {" "}
                <Save />
                保存设置
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
