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
import { useI18n } from '@/app/providers/i18n';
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
  FileText,
  Scissors,
  SearchCheck,
  Info,
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
  const { t } = useI18n();

  const [kb, setKb] = useState<KbConfig>(kbConfig);
  const [indexType, setIndexType] = useState('vector');
  const [embeddingmodels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [rerankermodels, setRerankerModels] = useState<RerankerModel[]>([]);
  const [visionModels, setVisionModels] = useState<VisionModel[]>([]);
  const [modelloading, setModelLoading] = useState(true);
  const [modelerror, setModelError] = useState('');

  const [saveErrorMsg, setSaveErrorMsg] = useState('');
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
        setModelError(err || t('knowledgebase.loadModelError'));
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

      if (!res.ok) throw new Error(`${t('knowledgebase.saveKbError')}: ${await res.text()}`);
      const jsondata = await res.json();
      setSaveErrorMsg('');
      onSaveSuccess(jsondata.data as KbConfig);
    } catch (err: any) {
      console.log('保存知识库失败', err.message);
      setSaveErrorMsg(err.message);
    }
  };



  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 overflow-y-auto px-6 pb-2">
        {/* Section 1: 基本信息 */}
        <section className="py-3 border-b border-border">
          <div className="flex items-center gap-2 mb-3">
            <div className="flex items-center justify-center w-6 h-6 rounded-md bg-primary/10 text-primary shrink-0">
              <FileText className="w-3.5 h-3.5" />
            </div>
            <h3 className="text-sm font-semibold leading-tight">{t('knowledgebase.nameLabel').replace('*', '').trim() || '基本信息'}</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-start pl-8">
            <div className="space-y-1.5 md:col-span-4">
              <Label htmlFor="name" className="text-xs font-medium">
                {t('knowledgebase.nameLabel')} <span className="text-destructive">*</span>
              </Label>
              <Input
                id="name"
                className="h-8 text-xs"
                value={kb.name}
                onChange={(e) =>
                  setKb((prev) => ({ ...prev, name: e.target.value }))
                }
                placeholder={t('knowledgebase.namePlaceholder')}
                required
              />
            </div>
            <div className="space-y-1.5 md:col-span-8">
              <Label htmlFor="description" className="text-xs font-medium">
                {t('knowledgebase.descriptionLabel')}
              </Label>
              <Input
                id="description"
                className="h-8 text-xs"
                value={kb.description}
                onChange={(e) =>
                  setKb((prev) => ({
                    ...prev,
                    description: e.target.value,
                  }))
                }
                placeholder={t('knowledgebase.descriptionPlaceholder')}
              />
            </div>
          </div>
        </section>

        {/* Section 2: 分段设置 */}
        <section className="py-3 border-b border-border">
          <div className="flex items-center gap-2 mb-3">
            <div className="flex items-center justify-center w-6 h-6 rounded-md bg-primary/10 text-primary shrink-0">
              <Scissors className="w-3.5 h-3.5" />
            </div>
            <h3 className="text-sm font-semibold leading-tight">{t('knowledgebase.chunkSettings')}</h3>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="right" className="max-w-xs">
                  <p className="text-xs">{t('knowledgebase.chunkSettingsHint')}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="space-y-3 pl-8">
            <div className="space-y-1.5 w-full max-w-md">
              <Label htmlFor="parserType" className="text-xs font-medium">
                {t('knowledgebase.parserType')}
                <span className="text-destructive ml-1">*</span>
              </Label>
              <Select
                value={kb.chunk_config.parser_type || 'structure'}
                onValueChange={(value) => {
                  setKb((prev) => {
                    const newConfig: any = {
                      ...prev.chunk_config,
                      parser_type: value,
                    };

                    if (value === 'table') {
                      newConfig.table_config = prev.chunk_config.table_config || {
                        concat_rows: false,
                        row_joiner: '\n',
                        header_index_max: 0,
                        format_sheet_data_to_json: false,
                      };
                      newConfig.chunk_size = prev.chunk_config.chunk_size || '1000';
                      delete newConfig.chunk_overlap;
                      delete newConfig.separator;
                    } else if (value === 'paragraph') {
                      newConfig.separator = prev.chunk_config.separator || '\n\n';
                      newConfig.chunk_size = prev.chunk_config.chunk_size || '1000';
                      newConfig.chunk_overlap = prev.chunk_config.chunk_overlap || '50';
                      delete newConfig.table_config;
                    } else {
                      newConfig.separator = prev.chunk_config.separator || '\n\n';
                      newConfig.chunk_size = prev.chunk_config.chunk_size || '1000';
                      newConfig.chunk_overlap = prev.chunk_config.chunk_overlap || '50';
                      delete newConfig.table_config;
                    }

                    return {
                      ...prev,
                      chunk_config: newConfig,
                    };
                  });
                }}
              >
                <SelectTrigger className="h-8 text-xs w-full">
                  <SelectValue placeholder={t('knowledgebase.selectParserType')} />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    <SelectItem value="structure" className="text-xs">
                      {t('knowledgebase.structure')}
                    </SelectItem>
                    <SelectItem value="token" className="text-xs">
                      {t('knowledgebase.token')}
                    </SelectItem>
                    <SelectItem value="table" className="text-xs">
                      {t('knowledgebase.table')}
                    </SelectItem>
                    <SelectItem value="paragraph" className="text-xs">
                      {t('knowledgebase.paragraph')}
                    </SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
              <p className="text-[11px] text-muted-foreground">{t('knowledgebase.selectChunkMode')}</p>
            </div>

            {/* Table Config */}
            {kb.chunk_config.parser_type === 'table' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 rounded-md bg-muted/30 p-3">
                <div className="space-y-1.5">
                  <Label htmlFor="table-header-index-max" className="text-xs font-medium">
                    {t('knowledgebase.maxHeaderIndex')}
                  </Label>
                  <Input
                    type="number"
                    className="h-8 text-xs"
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
                <div className="space-y-1.5">
                  <Label htmlFor="table-row-joiner" className="text-xs font-medium">
                    {t('knowledgebase.rowJoiner')}
                  </Label>
                  <Input
                    type="text"
                    className="h-8 text-xs"
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
                <div className="space-y-1.5">
                  <Label htmlFor="table-chunkSize" className="text-xs font-medium">
                    {t('knowledgebase.chunkSize')}
                    <span className="text-destructive ml-1">*</span>
                  </Label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    className="h-8 text-xs"
                    id="table-chunkSize"
                    value={kb.chunk_config.chunk_size ?? ''}
                    placeholder="1000"
                    onChange={(e) => {
                      const value = e.target.value;
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
                  <p className="text-[11px] text-muted-foreground">{t('knowledgebase.recommendedValue', { value: '1000' })}</p>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs font-medium block">{t('knowledgebase.mergeRows')} / {t('knowledgebase.formatAsJson')}</Label>
                  <div className="flex gap-4 items-center h-8">
                    <label htmlFor="table-concat-rows" className="flex items-center gap-1.5 text-xs cursor-pointer">
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
                      {t('knowledgebase.mergeRows')}
                    </label>
                    <label htmlFor="table-format-json" className="flex items-center gap-1.5 text-xs cursor-pointer">
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
                      {t('knowledgebase.formatAsJson')}
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* Paragraph Config */}
            {kb.chunk_config.parser_type === 'paragraph' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 rounded-md bg-muted/30 p-3">
                <div className="space-y-1.5">
                  <Label htmlFor="paragraph-separator" className="text-xs font-medium">
                    {t('knowledgebase.separator')}
                    <span className="text-destructive ml-1">*</span>
                  </Label>
                  <Input
                    type="text"
                    className="h-8 text-xs"
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
                <div className="space-y-1.5">
                  <Label htmlFor="chunkSize" className="text-xs font-medium">
                    {t('knowledgebase.chunkSize')}
                    <span className="text-destructive ml-1">*</span>
                  </Label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    className="h-8 text-xs"
                    id="chunkSize"
                    value={kb.chunk_config.chunk_size ?? ''}
                    placeholder="1000"
                    onChange={(e) => {
                      const value = e.target.value;
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
                  <p className="text-[11px] text-muted-foreground">{t('knowledgebase.recommendedValue', { value: '1000' })}</p>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="chunkOverlap" className="text-xs font-medium">
                    {t('knowledgebase.chunkOverlap')}
                    <span className="text-destructive ml-1">*</span>
                  </Label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    className="h-8 text-xs"
                    id="chunkOverlap"
                    value={kb.chunk_config.chunk_overlap ?? ''}
                    placeholder="50"
                    onChange={(e) => {
                      const value = e.target.value;
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
                  <p className="text-[11px] text-muted-foreground">{t('knowledgebase.recommendedValue', { value: '50' })}</p>
                </div>
              </div>
            )}

            {/* Default Config - structure or token */}
            {(kb.chunk_config.parser_type === 'structure' || kb.chunk_config.parser_type === 'token') && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 rounded-md bg-muted/30 p-3">
                <div className="space-y-1.5">
                  <Label htmlFor="chunkSize" className="text-xs font-medium">
                    {t('knowledgebase.chunkSize')}
                    <span className="text-destructive ml-1">*</span>
                  </Label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    className="h-8 text-xs"
                    id="chunkSize"
                    value={kb.chunk_config.chunk_size ?? ''}
                    placeholder="1000"
                    onChange={(e) => {
                      const value = e.target.value;
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
                  <p className="text-[11px] text-muted-foreground">{t('knowledgebase.recommendedValue', { value: '1000' })}</p>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="chunkOverlap" className="text-xs font-medium">
                    {t('knowledgebase.chunkOverlap')}
                    <span className="text-destructive ml-1">*</span>
                  </Label>
                  <Input
                    type="text"
                    inputMode="numeric"
                    className="h-8 text-xs"
                    id="chunkOverlap"
                    value={kb.chunk_config.chunk_overlap ?? ''}
                    placeholder="50"
                    onChange={(e) => {
                      const value = e.target.value;
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
                  <p className="text-[11px] text-muted-foreground">{t('knowledgebase.recommendedValue', { value: '50' })}</p>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label htmlFor="embeddingModel" className="text-xs font-medium">
                  {t('knowledgebase.embeddingModelLabel')} <span className="text-destructive ml-1">*</span>
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
                  <SelectTrigger className="h-8 text-xs w-full">
                    <SelectValue placeholder={t('knowledgebase.selectEmbeddingModel')} />
                  </SelectTrigger>
                  <SelectContent className="text-xs">
                    <SelectGroup>
                      {embeddingmodels.map((model) => (
                        <SelectItem key={model.id} value={model.model_id} className="text-xs">
                          {model.model_id}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="imageCaptionModel" className="text-xs font-medium">
                  {t('knowledgebase.imageCaptionModelLabel')}
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
                  <SelectTrigger className="h-8 text-xs w-full">
                    <SelectValue placeholder={t('knowledgebase.selectImageModel')} />
                  </SelectTrigger>
                  <SelectContent className="text-xs">
                    <SelectGroup>
                      <SelectItem value="DISABLED" className="text-xs">
                        {t('knowledgebase.disableImageModel')}
                      </SelectItem>
                      {visionModels.map((model) => (
                        <SelectItem key={model.id} value={model.model_id} className="text-xs">
                          {model.model_id} ({model.model})
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <p className="text-[11px] text-muted-foreground">{t('knowledgebase.imageModelHint')}</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: 检索设置 */}
        <section className="py-3 border-b border-border">
          <div className="flex items-center gap-2 mb-3">
            <div className="flex items-center justify-center w-6 h-6 rounded-md bg-primary/10 text-primary shrink-0">
              <SearchCheck className="w-3.5 h-3.5" />
            </div>
            <h3 className="text-sm font-semibold leading-tight">{t('knowledgebase.retrievalSettings')}</h3>
          </div>
          <div className="space-y-3 pl-8">
            {/* 检索策略 */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">{t('knowledgebase.retrievalStrategy')}</Label>
              <div className="flex items-center gap-3 flex-wrap">
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
                  className="gap-x-1"
                >
                  <ToggleGroupItem
                    value="vector"
                    aria-label={t('knowledgebase.vectorSearch')}
                    className="!rounded-md px-3 text-xs data-[state=on]:bg-primary data-[state=on]:text-primary-foreground h-7"
                  >
                    <ScanSearch className="w-3 h-3 mr-1" />
                    {t('knowledgebase.vectorSearch')}
                  </ToggleGroupItem>
                  {isFulltextSupported && (
                    <ToggleGroupItem
                      value="fulltext"
                      aria-label={t('knowledgebase.fulltextSearch')}
                      className="!rounded-md px-3 text-xs data-[state=on]:bg-primary data-[state=on]:text-primary-foreground h-7"
                    >
                      <TextSearch className="w-3 h-3 mr-1" />
                      {t('knowledgebase.fulltextSearch')}
                    </ToggleGroupItem>
                  )}
                  {isFulltextSupported && (
                    <ToggleGroupItem
                      value="hybrid"
                      aria-label={t('knowledgebase.hybridSearch')}
                      className="!rounded-md px-3 text-xs data-[state=on]:bg-primary data-[state=on]:text-primary-foreground h-7"
                    >
                      <SearchCode className="w-3 h-3 mr-1" />
                      {t('knowledgebase.hybridSearch')}
                    </ToggleGroupItem>
                  )}
                </ToggleGroup>
                {indexType === 'hybrid' && (
                  <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2 py-1">
                    <Label htmlFor="embeddingWeight" className="text-xs whitespace-nowrap">
                      {t('knowledgebase.vectorWeight')}
                    </Label>
                    <Slider
                      id="embeddingWeight"
                      className="w-40"
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
                    <span className="w-8 text-right text-xs font-medium tabular-nums">
                      {kb.retrieval_config.vector_weight?.toFixed(1) ?? '0.7'}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Top-K + Similarity threshold in one grid row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label htmlFor="topk" className="text-xs font-medium">Top-K</Label>
                  <span className="text-xs font-medium tabular-nums">{kb.retrieval_config.top_k}</span>
                </div>
                <Slider
                  id="topk"
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
                <p className="text-[11px] text-muted-foreground">{t('knowledgebase.topKHint')}</p>
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label htmlFor="similarityThreshold" className="text-xs font-medium">
                    {t('knowledgebase.similarityThreshold')}
                  </Label>
                  <span className="text-xs font-medium tabular-nums">
                    {kb.retrieval_config.similarity_threshold?.toFixed(2) ?? '0.00'}
                  </span>
                </div>
                <Slider
                  id="similarityThreshold"
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
                <p className="text-[11px] text-muted-foreground">{t('knowledgebase.similarityHint')}</p>
              </div>
            </div>

            {/* Rerank toggle + options */}
            <div className="rounded-md bg-muted/30 p-3 space-y-3">
              <label htmlFor="enable_reranker" className="flex items-center gap-2 cursor-pointer">
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
                <span className="text-xs font-medium">{t('knowledgebase.enableRerank')}</span>
              </label>
              {kb.retrieval_config.enable_rerank && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="rerank_model" className="text-xs font-medium">
                      {t('knowledgebase.rerankModelLabel')}
                      <span className="text-destructive ml-1">*</span>
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
                      <SelectTrigger className="h-8 text-xs w-full">
                        <SelectValue placeholder={t('knowledgebase.selectRerankModel')} />
                      </SelectTrigger>
                      <SelectContent className="text-xs">
                        <SelectGroup>
                          {rerankermodels.map((model) => (
                            <SelectItem key={model.id} value={model.model_id} className="text-xs">
                              {model.model_id}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="rerank_top_k" className="text-xs font-medium">
                        {t('knowledgebase.rerankTopK')}
                      </Label>
                      <span className="text-xs font-medium tabular-nums">
                        {kb.retrieval_config.rerank_top_k ?? 5}
                      </span>
                    </div>
                    <Slider
                      id="rerank_top_k"
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
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {saveErrorMsg !== '' && (
          <Alert variant="destructive" className="text-xs py-2 mt-3">
            <AlertCircleIcon className="h-3 w-3" />
            <AlertDescription className="text-xs">
              <p>{saveErrorMsg}</p>
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Sticky save bar (in-flow, no fixed positioning) */}
      <div className="flex-none border-t border-border bg-background/90 backdrop-blur-sm px-6 py-2.5 flex justify-end items-center gap-2">
        {isCreate && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onCancel()}
          >
            <SkipBack className="h-3 w-3 mr-1" />
            {t('common.cancel')}
          </Button>
        )}
        <Button type="button" size="sm" className="min-w-28" onClick={handleSubmit}>
          <Save className="h-3 w-3 mr-1" />
          {isCreate ? t('common.create') : t('knowledgebase.saveSettings')}
        </Button>
      </div>
    </div>
  );
};
