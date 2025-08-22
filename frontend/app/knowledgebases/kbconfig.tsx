'use client';
import React, { useState, useEffect, FC } from 'react';
import {
  Card,
  CardContent,
  CardFooter,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
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
} from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Slider } from '@/components/ui/slider';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogClose,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Checkbox } from '@/components/ui/checkbox';

interface EmbeddingModel {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
}

interface RerankerModel {
  id: string;
  model_id: string;
  model_name: string;
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
    enable_rerank: boolean;
    rerank_model?: string; // rerank模型名称
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
  const [indexType, setIndexType] = useState('vector');
  const [metadataOpen, setMetadataOpen] = useState(false);
  const [metadataName, setmetadataName] = useState('');
  const [metadataValueType, setMetadataValueType] = useState('string');
  const [metadataDesc, setMetadataDesc] = useState('');
  const [metadataError, setMetadataError] = useState('');
  const [embeddingmodels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [rerankermodels, setRerankerModels] = useState<RerankerModel[]>([]);
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(''); // 错误信息
  const [saveErrorMsg, setSaveErrorMsg] = useState(''); // 保存KB错误信息
  const [metadata_configs, setMetadataConfigs] =
    useState<MetadataConfig[]>(metadataConfigs);

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const [embRes, rerankerRes] = await Promise.all([
          fetch(`/api/config/embeddings`),
          fetch(`/api/config/rerankers`),
        ]);

        const embData = (await embRes.json())?.data.items || [];
        console.log('embData', embData);
        setEmbeddingModels([...embData]);

        const rerankerData = (await rerankerRes.json())?.data.items || [];
        console.log('rerankerData', rerankerData);
        setRerankerModels([...rerankerData]);
      } catch (err: any) {
        setModelError(err || '加载失败');
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log('保存知识库结果:', kb);
    const submit_url = isCreate
      ? `/api/config/knowledgebases`
      : `/api/config/knowledgebases/${kb.id}`;
    const updateMethod = isCreate ? 'POST' : 'PUT';
    try {
      const res = await fetch(submit_url, {
        method: updateMethod,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(kb), // 包装为数组
      });

      if (!res.ok) throw new Error(`保存知识库失败: ${await res.text()}`);
      const jsondata = await res.json();
      setSaveErrorMsg('');
      onSaveSuccess(jsondata.data as KbConfig);
    } catch (err: any) {
      console.log('保存知识库失败', err.message);
      setSaveErrorMsg(err.message);
    }
  };

  const handleRemoveMetadataEntry = async (id: string) => {
    if (metadata_configs != null) {
      const metadata_url = `/api/config/knowledgebases/${kb.id}/metadata/${id}`;
      try {
        const res = await fetch(metadata_url, {
          method: 'DELETE',
        });
        if (!res.ok) throw new Error(`删除metadata失败: ${await res.text()}`);

        const updated_metadata_configs = metadata_configs.filter(
          (config: any) => config.id !== id,
        );
        setMetadataConfigs(updated_metadata_configs);

        console.log('删除的元数据：', id);
      } catch (err: any) {
        console.log('删除元数据失败。', err.message);
      }
    }
  };

  const handleAddMetadataConfig = async () => {
    if (!metadataName) {
      setMetadataError('必须填入元数据名称。');
      return;
    }
    const updated_metadata_configs = metadata_configs || [];

    if (
      updated_metadata_configs.some((config) => config.name === metadataName)
    ) {
      setMetadataError(`元数据名称 '${metadataName}' 已经存在.`);
      return;
    }

    const metadata_url = `/api/config/knowledgebases/${kb.id}/metadata`;
    try {
      const res = await fetch(metadata_url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      console.log('添加元数据成功.');
    } catch (err: any) {
      console.log('保存知识库失败', err.message);
      setSaveErrorMsg(err.message);
    } finally {
      setmetadataName('');
      setMetadataError('');
      setMetadataValueType('string');
      setMetadataDesc('');
      setMetadataOpen(false);
    }
  };

  function handleCancelMetadataConfig() {
    setmetadataName('');
    setMetadataError('');
    setMetadataValueType('string');
    setMetadataDesc('');
    console.log('清空metadata信息');
  }

  return (
    <div className="h-200 overflow-y-auto">
      <div>
        <div className="flex space-y-2 gap-3 px-4 items-center">
          <Label htmlFor="name" className="w-[100px]">
            知识库名称 <span className="text-destructive">*</span>
          </Label>
          <Input
            id="name"
            className="w-60"
            value={kb.name}
            onChange={(e) =>
              setKb((prev) => ({ ...prev, name: e.target.value }))
            }
            placeholder="请输入知识库名称"
            required
          />
          <p className="text-sm text-muted-foreground">
            例如：&quot;XX产品用户手册&quot;、&quot;IT操作说明&quot;
          </p>
        </div>

        <div className="flex gap-3 px-4 items-center pt-6">
          <Label htmlFor="description" className="w-[100px]">
            知识库描述
          </Label>
          <Textarea
            id="description"
            className="w-120"
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

        <div className="flex gap-3 px-4 items-center pt-6">
          <Label htmlFor="chunkSize" className="w-[100px]">
            切片大小
            <span className="text-destructive">*</span>
          </Label>
          <Input
            type="number"
            className="w-60"
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
          <p className="text-sm text-muted-foreground">推荐值: 1000</p>

          <Label htmlFor="chunkOverlap" className="w-[100px] ml-20">
            切片重叠
            <span className="text-destructive">*</span>
          </Label>
          <Input
            type="number"
            className="w-60"
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

        <div className="flex gap-3 px-4 items-center pt-6">
          <Label htmlFor="embeddingModel" className="w-[100px]">
            向量模型 <span className="text-destructive">*</span>
          </Label>
          <Select
            value={kb.embedding_model}
            onValueChange={(value) => {
              setKb((prev) => ({ ...prev, embedding_model: value }));
            }}
          >
            <SelectTrigger className="w-40">
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
          <Label htmlFor="topk" className="w-[50px] ml-20">
            Top-K:
          </Label>
          <Slider
            className="w-60"
            defaultValue={[1]}
            max={100}
            min={1}
            step={1}
            value={[kb.retrieval_config.top_k]}
            onValueChange={(value: number[]) => {
              setKb((prev) => ({
                ...prev,
                retrieval_config: {
                  ...prev.retrieval_config,
                  top_k: value[0],
                },
              }));
            }}
          />
          <span className="font-medium"> {kb.retrieval_config.top_k} </span>

          <Label htmlFor="topk" className="w-[80px] ml-20">
            相似度阈值:
          </Label>
          <Slider
            className="w-60"
            defaultValue={[0]}
            max={1}
            step={0.01}
            value={[kb.retrieval_config.similarity_threshold]}
            onValueChange={(value: number[]) => {
              setKb((prev) => ({
                ...prev,
                retrieval_config: {
                  ...prev.retrieval_config,
                  similarity_threshold: value[0],
                },
              }));
            }}
          />
          <span className="font-medium">
            {' '}
            {kb.retrieval_config.similarity_threshold}{' '}
          </span>
        </div>

        <div className="flex gap-3 px-4 items-center pt-6">
          <Label className="w-[100px]">检索策略</Label>
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

          {indexType === 'hybrid' && (
            <div className="ml-10 flex">
              <Label htmlFor="embeddingWeight" className="w-[100px]">
                向量检索权重
              </Label>
              <Slider
                id="embeddingWeight"
                className="w-60"
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
              />
              <span className="w-12 text-right text-md font-medium">
                {kb.retrieval_config.vector_weight}
              </span>
            </div>
          )}
        </div>

        <div className="flex gap-3 px-4 items-center pt-6 h-14">
          <Label className="w-[100px]">开启重排序</Label>
          <Checkbox
            id="enable_reranker"
            checked={kb.retrieval_config.enable_rerank ?? false}
            onCheckedChange={(checked) => {
              setKb((prev) => ({
                ...prev,
                retrieval_config: {
                  ...prev.retrieval_config,
                  enable_rerank: Boolean(checked),
                },
              }));
            }}
          />
          {kb.retrieval_config.enable_rerank && (
            <div className="flex ml-20">
              <Label htmlFor="rerank_model" className="w-[100px]">
                重排序模型
                <span className="text-destructive">*</span>
              </Label>
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
                <SelectTrigger>
                  <SelectValue placeholder="请选择重排序模型" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {rerankermodels.map((model) => (
                      <SelectItem key={model.id} value={model.model_id}>
                        {model.model_id}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
        {!isCreate && (
          <div className="pt-6 px-4 gap-4">
            <div className="w-full">
              <Label htmlFor="metadata" className="w-[100px]">
                元数据配置
              </Label>
              <Table className="w-full">
                {metadata_configs === null || metadata_configs.length == 0 ? (
                  <TableCaption>尚未配置元数据信息</TableCaption>
                ) : (
                  <TableCaption>
                    已添加{metadata_configs.length}条元数据信息。{' '}
                  </TableCaption>
                )}
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">元数据名称(Key)</TableHead>
                    <TableHead className="w-[100px]">元数据类型</TableHead>
                    <TableHead>元数据描述</TableHead>
                    <TableHead className="w-[100px] text-right">
                      <Dialog
                        open={metadataOpen}
                        onOpenChange={setMetadataOpen}
                      >
                        <DialogTrigger asChild>
                          <Button variant="outline">
                            {' '}
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
                          <div className="grid gap-3">
                            <div className="grid gap-3">
                              <Label htmlFor="metadata_key">元数据名称</Label>
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
                              <Label htmlFor="metadata_desc">元数据描述</Label>
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
                      <TableCell className="font-medium">{item.name}</TableCell>
                      <TableCell>{item.value_type}</TableCell>
                      <TableCell>{item.description}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="secondary"
                          size="icon"
                          className="size-8"
                          onClick={() => handleRemoveMetadataEntry(item.id)}
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
        )}
        <div className="block w-full">
          {saveErrorMsg !== '' && (
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
            <div className="flex justify-center gap-3 pb-4">
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
                {' '}
                <Save />
                创建
              </Button>
            </div>
          )}
          {!isCreate && (
            <div className="flex justify-center gap-3 pb-4">
              <Button type="button" className="w-40" onClick={handleSubmit}>
                {' '}
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
