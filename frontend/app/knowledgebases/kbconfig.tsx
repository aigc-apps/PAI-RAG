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
  HelpCircle,
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface EmbeddingModel {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
  provider_name?: string;
}

interface RerankerModel {
  id: string;
  model_id: string;
  model_name: string;
  provider_name?: string;
}

interface VisionModel {
  id: string;
  model_id: string;
  model: string;
  provider_name?: string;
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
    separator?: string; // 切片标识符
    chunk_size?: string; // 切片大小
    chunk_overlap?: string; // 切片重叠大小
    image_caption_model?: string; // 图片理解模型ID
    image_caption_provider_name?: string; // 图片理解模型服务商
    table_config?: {
      concat_rows?: boolean;
      row_joiner?: string;
      header_index_max?: number;
      format_sheet_data_to_json?: boolean;
      sheet_column_filters?: string[];
    };
  };
  embedding_model: string; //向量模型名称
  embedding_provider_name?: string; // 向量模型服务商
  retrieval_config: {
    retrieval_mode: string; // 索引类型：vector, fulltext, hybrid
    top_k: number; // Top-K 值
    similarity_threshold: number; // 相似度分数阈值
    enable_rerank: boolean;
    rerank_model?: string; // rerank模型名称
    rerank_provider_name?: string; // rerank模型服务商
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
  const [visionModels, setVisionModels] = useState<VisionModel[]>([]);
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
          tenantFetch(`/api/config/embeddings?size=1000`),
          tenantFetch(`/api/config/rerankers?size=1000`),
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
        const mappedVisionModels = visionData.map((m: any) => ({ id: m.id, model_id: m.model_id, model: m.model, provider_name: m.provider_name }));
        setVisionModels(mappedVisionModels);

        // 初始化 provider_name：如果为空，从模型列表中填充
        setKb((prev) => {
          const updates: Partial<KbConfig> = {};
          
          // embedding_provider_name
          if (!prev.embedding_provider_name && prev.embedding_model) {
            const embModel = embData.find((m: EmbeddingModel) => m.model_id === prev.embedding_model);
            if (embModel?.provider_name) {
              updates.embedding_provider_name = embModel.provider_name;
            }
          }
          
          // chunk_config.image_caption_provider_name
          if (prev.chunk_config?.image_caption_model && !prev.chunk_config?.image_caption_provider_name) {
            const visionModel = mappedVisionModels.find((m: VisionModel) => m.model_id === prev.chunk_config.image_caption_model);
            if (visionModel?.provider_name) {
              updates.chunk_config = {
                ...prev.chunk_config,
                image_caption_provider_name: visionModel.provider_name,
              };
            }
          }
          
          // retrieval_config.rerank_provider_name
          if (prev.retrieval_config?.rerank_model && !prev.retrieval_config?.rerank_provider_name) {
            const rerankerModel = rerankerData.find((m: RerankerModel) => m.model_id === prev.retrieval_config.rerank_model);
            if (rerankerModel?.provider_name) {
              updates.retrieval_config = {
                ...prev.retrieval_config,
                rerank_provider_name: rerankerModel.provider_name,
              };
            }
          }
          
          if (Object.keys(updates).length > 0) {
            return { ...prev, ...updates };
          }
          return prev;
        });
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
      <div className="space-y-4 px-4">
        {/* 基本信息 */}
        <div className="flex gap-3 items-center">
          <Label htmlFor="name" className="w-[100px] text-xs">
            知识库名称 <span className="text-destructive">*</span>
          </Label>
          <Input
            id="name"
            className="w-80 h-8 text-xs font-normal"
            value={kb.name}
            onChange={(e) =>
              setKb((prev) => ({ ...prev, name: e.target.value }))
            }
            placeholder="请输入知识库名称"
            required
          />
        </div>

        <div className="flex gap-3 items-start pt-3">
          <Label htmlFor="description" className="w-[100px] text-xs pt-2">
            知识库描述
          </Label>
          <Textarea
            id="description"
            className="w-[600px] text-xs"
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

        {/* 分段设置卡片 */}
        <div className="flex gap-3 items-start pt-3">
          <div className="flex items-center gap-1 w-[100px] pt-2">
            <Label className="text-xs">
              分段设置
            </Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="right" className="max-w-xs">
                  <p className="text-xs">
                    分段设置参数对以下文件类型无效：.csv, .xlsx, .xls, .jsonl
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Card className="flex-1">
            <CardContent className="space-y-4 pt-6">
            <div className="flex gap-3 items-center">
              <Label htmlFor="parserType" className="w-[100px] text-xs">
                切片类型
                <span className="text-destructive">*</span>
              </Label>
              <Select
                value={kb.chunk_config.parser_type || 'structure'}
                onValueChange={(value) => {
                  setKb((prev) => {
                    const newConfig: any = {
                      ...prev.chunk_config,
                      parser_type: value,
                    };
                    
                    // 根据新的 parser_type 初始化相应的配置
                    if (value === 'table') {
                      newConfig.table_config = prev.chunk_config.table_config || {
                        concat_rows: false,
                        row_joiner: '\n',
                        header_index_max: 0,
                        format_sheet_data_to_json: false,
                      };
                      // 清除其他类型的配置
                      delete newConfig.chunk_size;
                      delete newConfig.chunk_overlap;
                      delete newConfig.separator;
                    } else if (value === 'paragraph') {
                      newConfig.separator = prev.chunk_config.separator || '\n\n';
                      newConfig.chunk_size = prev.chunk_config.chunk_size || '1000';
                      newConfig.chunk_overlap = prev.chunk_config.chunk_overlap || '50';
                      // 清除 table_config
                      delete newConfig.table_config;
                    } else {
                      newConfig.separator = prev.chunk_config.separator || '\n\n';
                      newConfig.chunk_size = prev.chunk_config.chunk_size || '1000';
                      newConfig.chunk_overlap = prev.chunk_config.chunk_overlap || '50';
                      // 清除 table_config
                      delete newConfig.table_config;
                    }
                    
                    return {
                      ...prev,
                      chunk_config: newConfig,
                    };
                  });
                }}
              >
                <SelectTrigger className="w-60 h-6 text-xs">
                  <SelectValue placeholder="请选择切片类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    <SelectItem value="structure" className="text-xs h-5">
                      结构化(structure)
                    </SelectItem>
                    <SelectItem value="token" className="text-xs h-5">
                      按token
                    </SelectItem>
                    <SelectItem value="table" className="text-xs h-5">
                      表格(table)
                    </SelectItem>
                    <SelectItem value="paragraph" className="text-xs h-5">
                      段落(paragraph)
                    </SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">选择文档切片方式</p>
            </div>

            {/* Table Config - 只在 parser_type === 'table' 时显示 */}
            {kb.chunk_config.parser_type === 'table' && (
              <div className="space-y-3">
                <div className="flex gap-3 items-center">
                  <div className="flex gap-3 items-center flex-1">
                    <Label htmlFor="table-header-index-max" className="w-[100px] text-xs">
                      最大表头行index
                    </Label>
                    <Input
                      type="number"
                      className="w-60 h-6 text-xs"
                      id="table-header-index-max"
                      value={kb.chunk_config.table_config?.header_index_max ?? 0}
                      onChange={(e) =>
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            table_config: {
                              ...prev.chunk_config.table_config,
                              header_index_max: e.target.value ? parseInt(e.target.value) : 0,
                            },
                          },
                        }))
                      }
                      min="0"
                    />
                  </div>
                  <div className="flex gap-3 items-center flex-1">
                    <Label htmlFor="table-format-json" className="w-[100px] text-xs">
                      格式化为Json
                    </Label>
                    <Checkbox
                      id="table-format-json"
                      checked={kb.chunk_config.table_config?.format_sheet_data_to_json ?? false}
                      onCheckedChange={(checked) =>
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            table_config: {
                              ...prev.chunk_config.table_config,
                              format_sheet_data_to_json: checked === true,
                            },
                          },
                        }))
                      }
                    />
                  </div>
                </div>
                <div className="flex gap-3 items-center">
                  <div className="flex gap-3 items-center flex-1">
                    <Label htmlFor="table-concat-rows" className="w-[100px] text-xs">
                      合并行
                    </Label>
                    <Checkbox
                      id="table-concat-rows"
                      checked={kb.chunk_config.table_config?.concat_rows ?? false}
                      onCheckedChange={(checked) =>
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            table_config: {
                              ...prev.chunk_config.table_config,
                              concat_rows: checked === true,
                            },
                          },
                        }))
                      }
                    />
                  </div>
                  <div className="flex gap-3 items-center flex-1">
                    <Label htmlFor="table-row-joiner" className="w-[100px] text-xs">
                      行分隔符
                    </Label>
                    <Input
                      type="text"
                      className="w-60 h-6 text-xs"
                      id="table-row-joiner"
                      value={kb.chunk_config.table_config?.row_joiner || '\n'}
                      onChange={(e) =>
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            table_config: {
                              ...prev.chunk_config.table_config,
                              row_joiner: e.target.value,
                            },
                          },
                        }))
                      }
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Paragraph Config - 只在 parser_type === 'paragraph' 时显示 */}
            {kb.chunk_config.parser_type === 'paragraph' && (
              <div className="space-y-3">
                <div className="flex gap-3 items-center">
                  <Label htmlFor="paragraph-separator" className="w-[100px] text-xs">
                    分隔符
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    type="text"
                    className="w-60 h-6 text-xs"
                    id="paragraph-separator"
                    value={kb.chunk_config.separator || '\n\n'}
                    onChange={(e) =>
                      setKb((prev) => ({
                        ...prev,
                        chunk_config: {
                          ...prev.chunk_config,
                          separator: e.target.value,
                        },
                      }))
                    }
                  />
                </div>
                <div className="flex gap-3 items-center">
                  <Label htmlFor="chunkSize" className="w-[100px] text-xs">
                    切片大小
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    className="w-60 h-6 text-xs"
                    id="chunkSize"
                    value={kb.chunk_config.chunk_size ?? ''}
                    placeholder="1000"
                    onChange={(e) => {
                      const value = e.target.value;
                      // 只允许数字和空字符串
                      if (value === '' || /^\d+$/.test(value)) {
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            chunk_size: value,
                          },
                        }));
                      }
                    }}
                    required
                  />
                  <p className="text-xs text-muted-foreground">推荐值: 1000</p>

                  <Label htmlFor="chunkOverlap" className="w-[100px] ml-20 text-xs">
                    切片重叠
                    <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    className="w-60 h-6 text-xs"
                    id="chunkOverlap"
                    value={kb.chunk_config.chunk_overlap ?? ''}
                    placeholder="50"
                    onChange={(e) => {
                      const value = e.target.value;
                      // 只允许数字和空字符串
                      if (value === '' || /^\d+$/.test(value)) {
                        setKb((prev) => ({
                          ...prev,
                          chunk_config: {
                            ...prev.chunk_config,
                            chunk_overlap: value,
                          },
                        }));
                      }
                    }}
                  />
                  <p className="text-xs text-muted-foreground">推荐值: 50</p>
                </div>
              </div>
            )}

            {/* Default Config - 只在 parser_type 为 'structure' 或 'token' 时显示 */}
            {(kb.chunk_config.parser_type === 'structure' || kb.chunk_config.parser_type === 'token') && (
              <div className="flex gap-3 items-center">
                <Label htmlFor="chunkSize" className="w-[100px] text-xs">
                  切片大小
                  <span className="text-destructive">*</span>
                </Label>
                <Input
                  type="text"
                  inputMode="numeric"
                  className="w-60 h-6 text-xs"
                  id="chunkSize"
                  value={kb.chunk_config.chunk_size ?? ''}
                  placeholder="1000"
                  onChange={(e) => {
                    const value = e.target.value;
                    // 只允许数字和空字符串
                    if (value === '' || /^\d+$/.test(value)) {
                      setKb((prev) => ({
                        ...prev,
                        chunk_config: {
                          ...prev.chunk_config,
                          chunk_size: value,
                        },
                      }));
                    }
                  }}
                  required
                />
                <p className="text-xs text-muted-foreground">推荐值: 1000</p>

                <Label htmlFor="chunkOverlap" className="w-[100px] ml-20 text-xs">
                  切片重叠
                  <span className="text-destructive">*</span>
                </Label>
                <Input
                  type="text"
                  inputMode="numeric"
                  className="w-60 h-6 text-xs"
                  id="chunkOverlap"
                  value={kb.chunk_config.chunk_overlap ?? ''}
                  placeholder="50"
                  onChange={(e) => {
                    const value = e.target.value;
                    // 只允许数字和空字符串
                    if (value === '' || /^\d+$/.test(value)) {
                      setKb((prev) => ({
                        ...prev,
                        chunk_config: {
                          ...prev.chunk_config,
                          chunk_overlap: value,
                        },
                      }));
                    }
                  }}
                />
                <p className="text-xs text-muted-foreground">推荐值: 50</p>
              </div>
            )}

            <div className="flex gap-3 items-center">
              <Label htmlFor="imageCaptionModel" className="w-[100px] text-xs">
                图片理解模型
              </Label>
              <Select
                value={kb.chunk_config.image_caption_model || 'DISABLED'}
                onValueChange={(value) => {
                  const selectedModel = visionModels.find(m => m.model_id === value);
                  setKb((prev) => ({
                    ...prev,
                    chunk_config: {
                      ...prev.chunk_config,
                      image_caption_model: value !== "DISABLED" ? value : undefined,
                      image_caption_provider_name: selectedModel?.provider_name || prev.chunk_config.image_caption_provider_name,
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

            <div className="flex gap-3 items-center">
              <Label htmlFor="embeddingModel" className="w-[100px] text-xs">
                向量模型 <span className="text-destructive">*</span>
              </Label>
              <Select
                value={kb.embedding_model}
                onValueChange={(value) => {
                  const selectedModel = embeddingmodels.find(m => m.model_id === value);
                  setKb((prev) => ({ 
                    ...prev, 
                    embedding_model: value,
                    embedding_provider_name: selectedModel?.provider_name || prev.embedding_provider_name,
                  }));
                }}
              >
                <SelectTrigger className="w-60 h-6 text-xs">
                  <SelectValue placeholder="请选择向量模型" />
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
            </div>
            </CardContent>
          </Card>
        </div>

        {/* 检索设置卡片 */}
        <div className="flex gap-3 items-start pt-3">
          <Label className="w-[100px] text-xs pt-2">
            检索设置
          </Label>
          <Card className="flex-1">
            <CardContent className="space-y-4 pt-6">
            <div className="flex gap-3 items-center">
              <Label className="w-[100px] text-xs">检索策略</Label>
              <ToggleGroup
                type="single"
                value={kb.retrieval_config.retrieval_mode}
                onValueChange={(value) => {
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
                  <span className="w-12 text-right text-xs font-medium ml-2">
                    {kb.retrieval_config.vector_weight?.toFixed(1) ?? '0.7'}
                  </span>
                </div>
              )}
            </div>

            <div className="flex gap-3 items-center">
              <Label htmlFor="topk" className="w-[100px] text-xs">
                Top-K
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
              <span className="font-medium text-xs ml-2"> {kb.retrieval_config.top_k} </span>
              <p className="text-xs text-muted-foreground ml-4">检索返回的最相似结果数量</p>
            </div>

            <div className="flex gap-3 items-center">
              <Label htmlFor="similarityThreshold" className="w-[100px] text-xs">
                相似度阈值
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
              <span className="font-medium text-xs ml-2">
                {kb.retrieval_config.similarity_threshold?.toFixed(2) ?? '0.00'}
              </span>
              <p className="text-xs text-muted-foreground ml-4">仅返回相似度大于等于该值的结果</p>
            </div>

            <div className="flex gap-3 items-center">
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
                        const selectedModel = rerankermodels.find(m => m.model_id === value);
                        setKb((prev) => ({
                          ...prev,
                          retrieval_config: {
                            ...prev.retrieval_config,
                            rerank_model: value,
                            rerank_provider_name: selectedModel?.provider_name || 'openai_like',
                          },
                        }));
                      }}
                    >
                      <SelectTrigger className="h-6 text-xs w-60">
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
            </CardContent>
          </Card>
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
