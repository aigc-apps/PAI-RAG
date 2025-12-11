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
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
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
  count?: number; // 引用该元数据的文件数量
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
    image_caption_model?: string; // 图片理解模型ID
  };
  embedding_model: string; //向量模型名称
  retrieval_config: {
    retrieval_mode: string; // 索引类型：vector, fulltext, hybrid
    top_k: number; // Top-K 值
    similarity_threshold: number; // 相似度分数阈值
    enable_rerank: boolean;
    rerank_model?: string; // rerank模型名称
    rerank_top_k?: number; // Rerank-Top-K 值
    vector_weight?: number; // 向量检索权重（仅 hybrid 时使用）
  };
}

interface KbConfigProps {
  kbConfig: KbConfig;
  isCreate: boolean;
  onSaveSuccess: (kb: KbConfig) => void;
  onCancel: () => void;
}

// 知识库配置卡片
export const KbConfigCard: FC<KbConfigProps> = ({
  kbConfig,
  isCreate,
  onSaveSuccess,
  onCancel,
}) => {
  const [kb, setKb] = useState<KbConfig>(kbConfig);
  const [indexType, setIndexType] = useState('vector');
  const [embeddingmodels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [rerankermodels, setRerankerModels] = useState<RerankerModel[]>([]);
  const [visionModels, setVisionModels] = useState<Array<{ id: string; model_id: string; model: string }>>([]);
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(''); // 错误信息

  const [saveErrorMsg, setSaveErrorMsg] = useState(''); // 保存KB错误信息
  const [vectorDbType, setVectorDbType] = useState<string>('local');
  const { tenantFetch } = useTenantFetch();
  // 不支持全文检索和混合检索的向量数据库类型列表
  const VECTOR_DB_TYPES_WITHOUT_FULLTEXT = ['local', 'opensearch', 'hologres'];
  
  const isFulltextSupported = !VECTOR_DB_TYPES_WITHOUT_FULLTEXT.includes(vectorDbType);


  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const [embRes, rerankerRes, vectordbRes, visionRes] = await Promise.all([
          tenantFetch(`/api/config/embeddings`),
          tenantFetch(`/api/config/rerankers`),
          tenantFetch(`/api/config/vectordb`),
          tenantFetch(`/api/config/llms?vision_support=true&size=1000`),
        ]);

        const embData = (await embRes.json())?.data.items || [];
        console.log('embData', embData);
        setEmbeddingModels([...embData]);

        const rerankerData = (await rerankerRes.json())?.data.items || [];
        console.log('rerankerData', rerankerData);
        setRerankerModels([...rerankerData]);
        
        // 获取向量数据库类型
        if (vectordbRes.ok) {
          const vectordbData = (await vectordbRes.json())?.data;
          if (vectordbData?.type) {
            setVectorDbType(vectordbData.type);
            // 如果当前检索模式不支持，回退到向量检索
            const currentIsFulltextSupported = !VECTOR_DB_TYPES_WITHOUT_FULLTEXT.includes(vectordbData.type);
            const currentRetrievalMode = kb.retrieval_config?.retrieval_mode || 'hybrid';
            if (!currentIsFulltextSupported && (currentRetrievalMode === 'fulltext' || currentRetrievalMode === 'hybrid')) {
              setKb((prev) => ({
                ...prev,
                retrieval_config: {
                  ...prev.retrieval_config,
                  retrieval_mode: 'vector',
                },
              }));
              setIndexType('vector');
            }
          }
        }

        const visionData = (await visionRes.json())?.data.items || [];
        console.log('visionData', visionData);
        setVisionModels(visionData.map((m: any) => ({ id: m.id, model_id: m.model_id, model: m.model })));
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
    kb.retrieval_config.enable_rerank = kb.retrieval_config.rerank_model && kb.retrieval_config.rerank_model.length > 0  ? true : false;
    try {
      const res = await tenantFetch(submit_url, {
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



  return (
    <div className="h-200 overflow-y-auto">
      <div>
        <div className="flex space-y-2 gap-3 px-4 items-center">
          <Label htmlFor="name" className="w-[100px] text-xs">
            知识库名称 <span className="text-destructive">*</span>
          </Label>
          <Input
            id="name"
            className="w-60 h-6 text-[0.2rem] font-normal"
            value={kb.name}
            onChange={(e) =>
              setKb((prev) => ({ ...prev, name: e.target.value }))
            }
            placeholder="请输入知识库名称"
            required
          />
          <p className="text-xs text-muted-foreground">
            例如：&quot;XX产品用户手册&quot;、&quot;IT操作说明&quot;
          </p>
        </div>

        <div className="flex gap-3 px-4 items-center pt-3">
          <Label htmlFor="description" className="w-[100px] text-xs">
            知识库描述
          </Label>
          <Textarea
            id="description"
            className="w-120 text-xs"
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

        <div className="flex gap-3 px-4 items-center pt-3">
          <Label htmlFor="chunkSize" className="w-[100px] text-xs">
            切片大小
            <span className="text-destructive">*</span>
          </Label>
          <Input
            type="number"
            className="w-60 h-6 text-xs"
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
          <p className="text-xs text-muted-foreground">推荐值: 1000</p>

          <Label htmlFor="chunkOverlap" className="w-[100px] ml-20 text-xs">
            切片重叠
            <span className="text-destructive">*</span>
          </Label>
          <Input
            type="number"
            className="w-60 h-6 text-xs"
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
          <p className="text-xs text-muted-foreground">推荐值: 50</p>
        </div>

        <div className="flex gap-3 px-4 items-center pt-3">
          <Label htmlFor="imageCaptionModel" className="w-[100px] text-xs">
            图片理解模型
          </Label>
          <Select
            value={kb.chunk_config.image_caption_model || 'DISABLED'}
            onValueChange={(value) => {
              setKb((prev) => ({
                ...prev,
                chunk_config: {
                  ...prev.chunk_config,
                  image_caption_model: value !== "DISABLED" ? value: undefined,
                },
              }));
            }}
          >
            <SelectTrigger className="w-60 h-6 text-xs">
              <SelectValue placeholder="请选择图片理解模型（可选）" />
            </SelectTrigger>
            <SelectContent className="text-xs">
              <SelectGroup>
                <SelectItem value="DISABLED" className="text-xs h-5">
                  不使用图片理解模型
                </SelectItem>
                {visionModels.map((model) => (
                  <SelectItem key={model.id} value={model.model_id} className="text-xs h-5">
                    {model.model_id} ({model.model})
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">用于理解图片内容</p>
        </div>

        <div className="flex gap-3 px-4 items-center pt-3">
          <Label htmlFor="embeddingModel" className="w-[100px] text-xs">
            向量模型 <span className="text-destructive">*</span>
          </Label>
          <Select
            value={kb.embedding_model}
            onValueChange={(value) => {
              setKb((prev) => ({ ...prev, embedding_model: value }));
            }}
          >
            <SelectTrigger className="w-40 h-6 text-xs">
              <SelectValue placeholder="请选择向量类型" />
            </SelectTrigger>
            <SelectContent className="text-xs">
              <SelectGroup>
                {embeddingmodels.map((model) => (
                  <SelectItem key={model.id} value={model.model_id} className="text-xs h-5">
                    {model.model_id}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
          <Label htmlFor="topk" className="w-[50px] ml-20 text-xs">
            Top-K:
          </Label>
          <Slider
            className="w-60"
            defaultValue={[5]}
            max={100}
            min={0}
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
          <span className="font-medium text-xs"> {kb.retrieval_config.top_k} </span>

          <Label htmlFor="topk" className="w-[80px] ml-20 text-xs">
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
          <span className="font-medium text-xs">
            {' '}
            {kb.retrieval_config.similarity_threshold?.toFixed(2) ?? '0.00'}{' '}
          </span>
        </div>

        <div className="flex gap-3 px-4 items-center pt-3">
          <Label className="w-[100px] text-xs">检索策略</Label>
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
            className="flex gap-x-1 overflow-visible"
          >
            <ToggleGroupItem
              value="vector"
              aria-label="向量检索"
              className="!rounded-full px-1.5 py-0.5 text-xs data-[state=on]:bg-black data-[state=on]:text-white"
            >
              <ScanSearch className="w-2 h-2 mr-0.5" />
              向量检索
            </ToggleGroupItem>
            {isFulltextSupported && (
              <ToggleGroupItem
                value="fulltext"
                aria-label="全文检索"
                className="!rounded-full px-6 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
              >
                <TextSearch />
                全文检索
              </ToggleGroupItem>
            )}
            {isFulltextSupported && (
              <ToggleGroupItem
                value="hybrid"
                aria-label="混合检索"
                className="!rounded-full px-6 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
              >
                <SearchCode />
                混合检索
              </ToggleGroupItem>
            )}
          </ToggleGroup>

          {indexType === 'hybrid' && (
            <div className="ml-10 flex items-center">
              <Label htmlFor="embeddingWeight" className="w-[100px] text-xs">
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
              <span className="w-12 text-right text-xs font-medium">
                {kb.retrieval_config.vector_weight?.toFixed(1) ?? '0.7'}
              </span>
            </div>
          )}
        </div>

        <div className="flex gap-3 px-4 items-center pt-3">
          <Label className="w-[100px] text-xs">开启重排序</Label>
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
            className="h-3.5 w-3.5"
          />
          {kb.retrieval_config.enable_rerank && (
            <>
              <div className="flex ml-20 items-center">
                <Label htmlFor="rerank_model" className="w-[100px] text-xs">
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
                  <SelectTrigger className="h-6 text-xs">
                    <SelectValue placeholder="请选择重排序模型" />
                  </SelectTrigger>
                  <SelectContent className="text-xs">
                    <SelectGroup>
                      {rerankermodels.map((model) => (
                        <SelectItem key={model.id} value={model.model_id} className="text-xs h-5">
                          {model.model_id}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex ml-20 items-center">
                <Label htmlFor="rerank_top_k" className="w-[100px] text-xs">
                  Rerank-Top-K
                </Label>
                <Slider
                  className="w-60"
                  defaultValue={[5]}
                  max={10}
                  min={0}
                  step={1}
                  value={[kb.retrieval_config.rerank_top_k ?? 5]}
                  onValueChange={(value: number[]) => {
                    setKb((prev) => ({
                      ...prev,
                      retrieval_config: {
                        ...prev.retrieval_config,
                        rerank_top_k: value[0],
                      },
                    }));
                  }}
                />
                <span className="font-medium ml-2 text-xs">
                  {kb.retrieval_config.rerank_top_k ?? 5}
                </span>
              </div>
            </>
          )}
        </div>
        <div className="block w-full">
          {saveErrorMsg !== '' && (
            <Alert variant="destructive" className="text-xs py-2">
              <AlertCircleIcon className="h-3 w-3" />
              <AlertDescription className="text-xs">
                <p>{saveErrorMsg}</p>
              </AlertDescription>
            </Alert>
          )}
        </div>
      </div>

      <div className="fixed bottom-0 inset-x-0 h-16 bg-white border-t left-64 flex justify-around items-center z-50 ">
        <div>
          {isCreate && (
            <div className="flex justify-center gap-2 pb-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="w-32 text-xs h-7"
                onClick={() => onCancel()}
              >
                <SkipBack className="h-3 w-3" />
                取消
              </Button>
              <Button type="button" size="sm" className="w-32 text-xs h-7" onClick={handleSubmit}>
                {' '}
                <Save className="h-3 w-3" />
                创建
              </Button>
            </div>
          )}
          {!isCreate && (
            <div className="flex justify-center gap-2 pb-2">
              <Button type="button" size="sm" className="w-32 text-xs h-7" onClick={handleSubmit}>
                {' '}
                <Save className="h-3 w-3" />
                保存设置
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
