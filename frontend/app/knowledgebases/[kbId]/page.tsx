'use client';
import React, { useState, useEffect, useCallback, useRef, use } from 'react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardFooter,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Label } from '@/components/ui/label';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  SheetFooter,
  SheetClose,
} from '@/components/ui/sheet';

import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

import {
  Loader2,
  CheckCircle,
  XCircle,
  Trash2Icon,
  AlertCircleIcon,
  SearchIcon,
  ChevronDownIcon,
  RefreshCcwIcon,
  CirclePlayIcon,
  Search,
  MoreVertical,
  Upload,
  InfoIcon,
  Database,
  Edit,
  Pencil,
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogTrigger,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { MarkdownViewer } from '@/app/knowledgebases/[kbId]/viewer/markdown-viewer';
import { JsonlViewer } from '@/app/knowledgebases/[kbId]/viewer/jsonl-viewer';
import { HtmlViewer } from '@/app/knowledgebases/[kbId]/viewer/html-viewer';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { PlusIcon, FilterIcon } from 'lucide-react';
import * as Toast from '@radix-ui/react-toast';
import { KbConfig, KbConfigCard, MetadataConfig } from '../kbconfig';
import { formatFileSize, formatBeijingTime } from '../utils/utils';
import { useI18n } from '@/app/providers/i18n';
import { FileStatusFilter } from '@/components/customized/file-status-filter';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuItem,
} from '@/components/ui/dropdown-menu';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { PhotoProvider, PhotoView } from 'react-photo-view';
import 'react-photo-view/dist/react-photo-view.css';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Skeleton } from '@/components/ui/skeleton';
import { DatetimeInput } from '../datetime';
import { Role } from '@/app/config/role/role';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import { Slider } from '@/components/ui/slider';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { SearchCode, TextSearch, ScanSearch, ChevronDownIcon as ChevronDown, ChevronUpIcon as ChevronUp, Save } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { HeaderPortal } from '@/components/header-portal';
import { Settings as SettingsIcon } from 'lucide-react';
interface KnowledgeBaseFile {
  id: string;
  file_name: string;
  file_size: string;
  status: string;
  file_source: string;
  created_at: string;
  updated_at: string;
  failed_reason: string;
  file_extension?: string;
  chunk_config?: {
    parser_type?: string;
    [key: string]: any;
  };
  file_metadata: {
    [key: string]: any;
    file_url?: string;
    is_local?: boolean;
  };
}

interface ImageInfo {
  url: string;
  desc: string;
}

interface SearchRecord {
  content: string;
  title: string;
  score: number;
  metadata: {
    file_path: string;
    file_name: string;
    file_size: number;
    file_extension: string;
    images: string[];
    images_info: Array<ImageInfo>;
    rerank: boolean;
  };
}

import {
  ConditionGroup,
  ConditionGroupEditor,
  createEmptyConditionGroup,
  conditionGroupToPayload,
} from './condition-group-editor';

export default function KnowledgeBaseDetailPage(
  { params } : { params: Promise<{ kbId: string }> }
) {
  const { t } = useI18n();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [knowledgebase, setKnowledgeBase] = useState<KbConfig>();
  const [kbfiles, setKbFiles] = useState(Array<KnowledgeBaseFile>); // 知识库列表
  const [page, setPage] = useState(1);
  const pageRef = useRef(page);
  const [totalPages, setTotalPages] = useState(1);
  const fileSizePerPage = 10;
  const [kbquery, setKbQuery] = useState(''); //查询
  const [fileQuery, setFileQuery] = useState('');
  const fileQueryRef = useRef(fileQuery);
  const [statusFilter, setStatusFilter] = useState('all');
  const statusRef = useRef(statusFilter);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [showBatchDeleteDialog, setShowBatchDeleteDialog] = useState(false);
  const [showBatchReprocessDialog, setShowBatchReprocessDialog] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewFile, setPreviewFile] = useState<KnowledgeBaseFile | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState<Record<string, boolean>>({});
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [chunkConfigDialogOpen, setChunkConfigDialogOpen] = useState(false);
  const [viewingChunkConfig, setViewingChunkConfig] = useState<{file_name: string; chunk_config: any} | null>(null);
  const [searchrecords, setSearchRecords] = useState(Array<SearchRecord>); // 搜索结果
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null); // 搜索错误信息
  const [expandedCards, setExpandedCards] = useState<Record<number, boolean>>({}); // 展开的卡片索引
  const [loadingMsg, setLoadingMsg] = useState(t('knowledgebase.loadingKbConfig'));
  const [conditionGroup, setConditionGroup] = useState<ConditionGroup>(createEmptyConditionGroup());
  const [fileSource, setFileSource] = useState('');
  const [fileSourceOpen, setFileSourceOpen] = useState(false);
  const [currentFileId, setCurrentFileId] = useState<string>('');
  const { kbId } = use(params);

  let isRefreshing = false;
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0); // 上传进度 0-100
  const [uploadStep, setUploadStep] = useState<'idle' | 'uploading' | 'uploaded' | 'parsing'>('idle'); // 上传步骤
  const [isDragging, setIsDragging] = useState(false); // 拖拽状态
  const [uploadedFiles, setUploadedFiles] = useState<Array<{id: string; file_name: string; file_path: string; chunk_config?: any}>>([]);  // 已上传待解析的文件
  const [uploadChunkConfig, setUploadChunkConfig] = useState<{
    parser_type: string;
    separator?: string;
    chunk_size?: string;
    chunk_overlap?: string;
    image_caption_model?: string;
    image_caption_provider_name?: string;
    table_config?: {
      concat_rows?: boolean;
      row_joiner?: string;
      header_index_max?: number | null;
      format_sheet_data_to_json?: boolean;
      sheet_column_filters?: string[];
    };
  } | null>(null); // 上传时使用的 chunk_config
  const [deleting, setDeleting] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [reprocessChunkConfigDialogOpen, setReprocessChunkConfigDialogOpen] = useState(false);
  const [reprocessChunkConfig, setReprocessChunkConfig] = useState<{
    parser_type: string;
    separator?: string;
    chunk_size?: string;
    chunk_overlap?: string;
    image_caption_model?: string;
    image_caption_provider_name?: string;
    table_config?: {
      concat_rows?: boolean;
      row_joiner?: string;
      header_index_max?: number | null;
      format_sheet_data_to_json?: boolean;
      sheet_column_filters?: string[];
    };
  } | null>(null);
  const [pendingReprocessFileId, setPendingReprocessFileId] = useState<string | null>(null);
  const [pendingReprocessFileIds, setPendingReprocessFileIds] = useState<string[]>([]);
  const [isBatchReprocess, setIsBatchReprocess] = useState(false);
  const [isEditingMetadata, setIsEditingMetadata] = useState(false);
  const [editingMetadata, setEditingMetadata] = useState<{ [k: string]: any }>(
    {},
  );
  const [metadataDialogOpen, setMetadataDialogOpen] = useState(false);
  const [currentMetadataFileId, setCurrentMetadataFileId] = useState<string>('');
  const [metadataConfigs, setMetadataConfigs] = useState<MetadataConfig[]>([]);
  const [metadataValueTypes, setMetadataValueTypes] = useState<{
    [k: string]: any;
  }>({});
  const [metadataEditError, setMetadataEditError] = useState<string>('');
  const [availableMetadataKeys, setAvailableMetadataKeys] = useState<string[]>(
    [],
  );
  const [metadataConfigDialogOpen, setMetadataConfigDialogOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [view, setView] = useState<'details' | 'retrieval_test'>('details');
  const [metadataEditDialogOpen, setMetadataEditDialogOpen] = useState(false);
  const [editingMetadataConfig, setEditingMetadataConfig] = useState<MetadataConfig | null>(null);
  const [newMetadataName, setNewMetadataName] = useState('');
  const [newMetadataValueType, setNewMetadataValueType] = useState('string');
  const [newMetadataDesc, setNewMetadataDesc] = useState('');
  const [metadataError, setMetadataError] = useState('');

  const [roles, setRoles] = useState<Role[]>([]);
  const [roleDialogOpen, setRoleDialogOpen] = useState(false);
  const [editRoleFileId, setEditRoleFileId] = useState('');
  const [activeRoleIds, setActiveRoleIds] = useState<string[]>([]);
  const [activeRoleNames, setActiveRoleNames] = useState<string[]>([]);
  const [user, setUser] = useState('');
  const router = useRouter();
  const abortControllerRef = useRef<AbortController | null>(null);
  
  // 检索设置状态
  const [retrievalSetting, setRetrievalSetting] = useState<{
    retrieval_mode?: string;
    rerank_provider_name?: string;
    vector_weight?: number;
    enable_rerank?: boolean;
    rerank_model?: string;
    top_k?: number;
    similarity_threshold?: number;
    rerank_top_k?: number;
  }>({});
  const [rerankerModels, setRerankerModels] = useState<Array<{id: string; model_id: string; model_name: string}>>([]);
  const [visionModels, setVisionModels] = useState<Array<{id: string; model_id: string; model: string; provider_name?: string}>>([]);
  const [retrievalSettingOpen, setRetrievalSettingOpen] = useState(true);
  const [vectorDbType, setVectorDbType] = useState<string>('local');
  const { tenantFetch, tenantId } = useTenantFetch();

  // 不支持全文检索和混合检索的向量数据库类型列表
  const VECTOR_DB_TYPES_WITHOUT_FULLTEXT = ['local', 'opensearch', 'hologres'];
  
  const isFulltextSupported = !VECTOR_DB_TYPES_WITHOUT_FULLTEXT.includes(vectorDbType);

  const default_metadata_keys = [
    'file_name',
    'file_path',
    'file_size',
    'file_extension',
    'file_url',
    'doc_id',
  ];

  const handleQueryInputChange = (
    e:
      | React.ChangeEvent<HTMLInputElement>
      | React.ChangeEvent<HTMLTextAreaElement>,
  ) => {
    const { id, value } = e.target;
    setKbQuery(value);
  };

  const handleSearchSubmit = async () => {
    setSearching(true);
    setSearchError(null);
    setSearchRecords([]);
    console.log('handleSearchSubmit');

    try {
      const search_result = await tenantFetch(`/api/retrieval`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: kbquery,
          user_id: user,
          knowledge_id: kbId,
          retrieval_setting: retrievalSetting,
          metadata_condition: conditionGroupToPayload(conditionGroup),
        }),
      });

      const search_json = await search_result.json();
      console.log('搜索知识库结果:', search_json);

      // 检查响应中的 status_code 或 code 字段
      const statusCode = search_json.status_code || search_json.code;
      if (statusCode && statusCode !== 200) {
        const errorMessage = search_json.message || search_json.error || t('knowledgebase.searchFailed');
        setSearchError(`错误 ${statusCode}: ${errorMessage}`);
        setSearchRecords([]);
        setSearching(false);
        return;
      }

      // 检查 HTTP 状态码
      if (!search_result.ok) {
        const errorMessage = search_json.message || search_json.error || `HTTP ${search_result.status}: 搜索知识库失败`;
        setSearchError(errorMessage);
        setSearchRecords([]);
        setSearching(false);
        return;
      }

      const records = search_json.records || search_json.data?.records || [];
      setSearchRecords(records);
      setSearchError(null);
    } catch (err: any) {
      const errorMessage = err.message || t('knowledgebase.searchKbFailed');
      setSearchError(errorMessage);
      setSearchRecords([]);
    } finally {
      setSearching(false);
    }
  };

  useEffect(() => {
    pageRef.current = page;
  }, [page]);

  useEffect(() => {
    statusRef.current = statusFilter;
  }, [statusFilter]);

  useEffect(() => {
    fileQueryRef.current = fileQuery;
  }, [fileQuery]);


  const fetchKbFiles = useCallback(async () => {
    // 取消上一次请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // 创建新的 AbortController
    const controller = new AbortController();
    abortControllerRef.current = controller;

    console.log("Refreshing file list... query:", fileQueryRef.current, statusRef.current);

    const filter = statusRef.current;
    const url = `/api/config/knowledgebases/${kbId}/files?page=${pageRef.current}&size=${fileSizePerPage}&query=${fileQueryRef.current || ''}&status=${filter === 'all' ? '': filter}`;

    try {
      const files_res = await tenantFetch(url, { signal: controller.signal, });
      if (!files_res.ok) throw new Error(t('knowledgebase.fetchFilesFailed'));

      const file_json_data = await files_res.json();
      console.log('获取知识库文件reponse:', file_json_data);
      const data = file_json_data.data.items;
      setKbFiles(data || []);
      setTotalPages(file_json_data.data.pages);

    } catch (err: any) {
      if (err instanceof Error && err.name !== 'AbortError') {
        toast.error(err.message);
      }
    }
  }, [kbId]);

  // 页面和状态筛选变化时立即获取数据
  useEffect(() => {
    fetchKbFiles();
  }, [fetchKbFiles, page, statusFilter]);

  // 搜索关键词变化时触发搜索（带防抖）
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setPage(1); // 搜索时重置到第一页
      fetchKbFiles();
    }, 300); // 300ms 防抖

    return () => clearTimeout(timeoutId);
  }, [fileQuery, fetchKbFiles]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const fetchMetadataConfigs = useCallback(async () => {
    try {
      const metaRes = await tenantFetch(`/api/config/knowledgebases/${kbId}/metadata`);
      if (!metaRes.ok) throw new Error(t('knowledgebase.fetchMetaFailed'));
      const metadata_json = await metaRes.json();
      const metadata_data = metadata_json.data.items as MetadataConfig[];
      console.log('metadata_data', metadata_data);
      const valueTypes = Object.fromEntries(
        metadata_data.map((metadata) => [metadata.name, metadata.value_type]),
      ) as { [key: string]: string };

      console.log('知识库元数据: ', metadata_data, valueTypes);

      setMetadataValueTypes({ ...valueTypes, '': 'string' });
      setMetadataConfigs(metadata_data);
    } catch (err: any) {
      toast.error(err.message);
    }
  }, [kbId]);

  const fetchKbConfigs = useCallback(async () => {
    try {
      setLoadingMsg(t('knowledgebase.loadingKbConfig'));
      const [kbRes, metaRes, rerankerRes, vectordbRes, visionRes] = await Promise.all([
        tenantFetch(`/api/config/knowledgebases/${kbId}`),
        tenantFetch(`/api/config/knowledgebases/${kbId}/metadata`),
        tenantFetch(`/api/config/rerankers`),
        tenantFetch(`/api/config/vectordb`),
        tenantFetch(`/api/config/llms?vision_support=true&size=1000`),
      ]);

      if (!kbRes.ok) {
        const errorData = await kbRes.json();
        setLoadingMsg(errorData.message || t('knowledgebase.fetchKbConfigFailed'));
        throw new Error(errorData.message || t('knowledgebase.fetchKbConfigFailed'));
      }
      const json_data = await kbRes.json();
      const kb_data = json_data.data;

      setKnowledgeBase(kb_data); // 更新状态
      // 初始化上传时的 chunk_config 为知识库的默认配置
      if (kb_data?.chunk_config) {
        const config: any = {
          parser_type: kb_data.chunk_config.parser_type || 'structure',
          image_caption_model: kb_data.chunk_config.image_caption_model,
          image_caption_provider_name: kb_data.chunk_config.image_caption_provider_name,
        };
        
                        if (kb_data.chunk_config.parser_type === 'table') {
                          config.table_config = kb_data.chunk_config.table_config || {
                            concat_rows: false,
                            row_joiner: '\n',
                            header_index_max: 0,
                            format_sheet_data_to_json: false,
                          };
                        } else if (kb_data.chunk_config.parser_type === 'paragraph') {
                          config.separator = kb_data.chunk_config.separator || '\n\n';
          config.chunk_size = String(kb_data.chunk_config.chunk_size || 1000);
          config.chunk_overlap = String(kb_data.chunk_config.chunk_overlap || 50);
        } else {
          config.separator = kb_data.chunk_config.separator || '\n\n';
          config.chunk_size = String(kb_data.chunk_config.chunk_size || 1000);
          config.chunk_overlap = String(kb_data.chunk_config.chunk_overlap || 50);
        }
        
        setUploadChunkConfig(config);
      }
      console.log('知识库详情数据:', kb_data);

      // 获取向量数据库类型
      let currentVectorDbType = 'local';
      if (vectordbRes.ok) {
        const vectordbData = (await vectordbRes.json())?.data;
        if (vectordbData?.type) {
          currentVectorDbType = vectordbData.type;
          setVectorDbType(currentVectorDbType);
        }
      }
      
      // 检查是否支持全文检索
      const currentIsFulltextSupported = !VECTOR_DB_TYPES_WITHOUT_FULLTEXT.includes(currentVectorDbType);
      
      // 初始化检索设置，从 knowledgebase.retrieval_config 获取默认值
      if (kb_data?.retrieval_config) {
        let retrievalMode = kb_data.retrieval_config.retrieval_mode || 'hybrid';
        // 如果向量数据库不支持全文检索，且当前模式是全文检索或混合检索，则回退到向量检索
        if (!currentIsFulltextSupported && (retrievalMode === 'fulltext' || retrievalMode === 'hybrid')) {
          retrievalMode = 'vector';
        }
        setRetrievalSetting({
          retrieval_mode: retrievalMode,
          vector_weight: kb_data.retrieval_config.vector_weight ?? 0.5,
          enable_rerank: kb_data.retrieval_config.enable_rerank ?? false,
          rerank_model: kb_data.retrieval_config.rerank_model || '',
          top_k: kb_data.retrieval_config.top_k ?? 5,
          similarity_threshold: kb_data.retrieval_config.similarity_threshold ?? 0.2,
          rerank_top_k: kb_data.retrieval_config.rerank_top_k ?? 5,
          rerank_provider_name: kb_data.retrieval_config.rerank_provider_name || '',
        });
      }

      // 处理元数据
      if (!metaRes.ok) throw new Error(t('knowledgebase.fetchMetaFailed'));
      const metadata_json = await metaRes.json();
      const metadata_data = metadata_json.data.items as MetadataConfig[];
      const valueTypes = Object.fromEntries(
        metadata_data.map((metadata) => [metadata.name, metadata.value_type]),
      ) as { [key: string]: string };

      console.log('知识库元数据: ', metadata_data, valueTypes);

      setMetadataValueTypes({ ...valueTypes, '': 'string' });
      setMetadataConfigs(metadata_data);

      // 获取重排序模型列表
      if (rerankerRes.ok) {
        const rerankerData = (await rerankerRes.json())?.data?.items || [];
        setRerankerModels(rerankerData);
      }

      // 获取图片理解模型列表
      if (visionRes.ok) {
        const visionData = (await visionRes.json())?.data?.items || [];
        const mappedVisionModels = visionData.map((m: any) => ({ 
          id: m.id, 
          model_id: m.model_id, 
          model: m.model, 
          provider_name: m.provider_name 
        }));
        setVisionModels(mappedVisionModels);
      }
    } catch (err: any) {
      toast.error(err.message);
    }
  }, [kbId]);

  useEffect(() => {
    fetchKbConfigs();
  }, [fetchKbConfigs]);

  if (!knowledgebase) {
    return <div className="p-6">{loadingMsg}</div>;
  }

  const handleSaveSuccess = async (kb: KbConfig) => {
    toast.success(t('knowledgebase.saveKbSuccess'));
    // 刷新知识库配置信息
    await fetchKbConfigs();
  };

  const handleSaveRetrievalSetting = async () => {
    try {
      // 构建retrieval_config对象，使用当前检索设置的值，如果没有则使用知识库的默认值
      const retrieval_config = {
        retrieval_mode: retrievalSetting.retrieval_mode || knowledgebase?.retrieval_config?.retrieval_mode || 'hybrid',
        top_k: retrievalSetting.top_k ?? knowledgebase?.retrieval_config?.top_k ?? 5,
        similarity_threshold: retrievalSetting.similarity_threshold ?? knowledgebase?.retrieval_config?.similarity_threshold ?? 0.2,
        vector_weight: retrievalSetting.vector_weight ?? knowledgebase?.retrieval_config?.vector_weight ?? 0.5,
        enable_rerank: retrievalSetting.enable_rerank ?? knowledgebase?.retrieval_config?.enable_rerank ?? false,
        rerank_model: retrievalSetting.rerank_model || knowledgebase?.retrieval_config?.rerank_model || '',
        rerank_top_k: retrievalSetting.rerank_top_k ?? knowledgebase?.retrieval_config?.rerank_top_k ?? 5,
      };

      knowledgebase.retrieval_config = retrieval_config;

      // 调用更新知识库接口，只更新retrieval_config
      const res = await tenantFetch(`/api/config/knowledgebases/${kbId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(knowledgebase),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`保存检索设置失败: ${errorText}`);
      }

      const jsonData = await res.json();
      
      // 更新本地知识库配置
      if (jsonData.data) {
        setKnowledgeBase(jsonData.data);
        // 同时更新检索设置的默认值，使用保存后的值
        setRetrievalSetting({
          retrieval_mode: jsonData.data.retrieval_config?.retrieval_mode || 'hybrid',
          vector_weight: jsonData.data.retrieval_config?.vector_weight ?? 0.5,
          enable_rerank: jsonData.data.retrieval_config?.enable_rerank ?? false,
          rerank_model: jsonData.data.retrieval_config?.rerank_model || '',
          top_k: jsonData.data.retrieval_config?.top_k ?? 5,
          similarity_threshold: jsonData.data.retrieval_config?.similarity_threshold ?? 0.2,
          rerank_top_k: jsonData.data.retrieval_config?.rerank_top_k ?? 5,
        });
      }

      toast.success(t('knowledgebase.saveRetrievalSuccess'));
    } catch (err: any) {
      console.error('保存检索设置失败:', err);
      toast.error(err.message || t('knowledgebase.saveRetrievalFailed'));
    }
  };


  const handleReprocessFile = async (file_id: string) => {
    // 找到文件对象，获取其chunk_config或使用知识库默认配置
    const file = kbfiles.find(f => f.id === file_id);
    const defaultConfig = knowledgebase?.chunk_config || {
      parser_type: 'structure',
      chunk_size: '1000',
      chunk_overlap: '50',
    };
    
    // 初始化切片配置：优先使用文件的chunk_config，否则使用知识库默认配置
    const initialConfig: any = file?.chunk_config || defaultConfig;
    const config: any = {
      parser_type: initialConfig.parser_type || 'structure',
      image_caption_model: initialConfig.image_caption_model,
      image_caption_provider_name: initialConfig.image_caption_provider_name || 'openai_like',
    };
    
    if (initialConfig.parser_type === 'table') {
      config.table_config = initialConfig.table_config || {
        concat_rows: false,
        row_joiner: '\n',
        header_index_max: 0,
        format_sheet_data_to_json: false,
      };
    } else if (initialConfig.parser_type === 'paragraph') {
      config.separator = initialConfig.separator || '\n\n';
      config.chunk_size = String(initialConfig.chunk_size || 1000);
      config.chunk_overlap = String(initialConfig.chunk_overlap || 50);
    } else {
      config.separator = initialConfig.separator || '\n\n';
      config.chunk_size = String(initialConfig.chunk_size || 1000);
      config.chunk_overlap = String(initialConfig.chunk_overlap || 50);
    }
    
    setReprocessChunkConfig(config);
    setPendingReprocessFileId(file_id);
    setIsBatchReprocess(false);
    setReprocessChunkConfigDialogOpen(true);
  };

  const confirmReprocessFile = async () => {
    if (!pendingReprocessFileId) return;
    
    try {
      const body: any = {};
      if (reprocessChunkConfig) {
        // 转换配置格式以匹配API要求
        const chunkConfig: any = {
          parser_type: reprocessChunkConfig.parser_type,
          image_caption_model: reprocessChunkConfig.image_caption_model || null,
          image_caption_provider_name: reprocessChunkConfig.image_caption_provider_name || 'openai_like',
        };
        
        if (reprocessChunkConfig.parser_type === 'table' && reprocessChunkConfig.table_config) {
          chunkConfig.table_config = reprocessChunkConfig.table_config;
        } else {
          chunkConfig.separator = reprocessChunkConfig.separator || '\n\n';
          chunkConfig.chunk_size = reprocessChunkConfig.chunk_size ? parseInt(reprocessChunkConfig.chunk_size) : 1000;
          chunkConfig.chunk_overlap = reprocessChunkConfig.chunk_overlap ? parseInt(reprocessChunkConfig.chunk_overlap) : 50;
        }
        
        body.chunk_config = chunkConfig;
      }
      
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${pendingReprocessFileId}`,
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
        },
      );
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.message || `重新解析失败`);
      }
      toast.success(t('knowledgebase.fileEnqueued'));
      setReprocessChunkConfigDialogOpen(false);
      setPendingReprocessFileId(null);
      setReprocessChunkConfig(null);
    } catch (error: any) {
      toast.error(error.message);
    } finally {
      fetchKbFiles();
    }
  };


  const handleDeleteFile = async (file_id: string) => {
    setDeleting(true);
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${file_id}`,
        {
          method: 'DELETE',
        },
      );
      if (!res.ok) throw new Error(`删除 ${file_id} 失败`);
      toast.success(t('knowledgebase.fileDeleted'));
    } catch (error: any) {
      toast.error(error.message);
    } finally {
      setDeleting(false);
      fetchKbFiles();
    }
  };

  const handleBatchDeleteFiles = async () => {
    if (selectedFiles.size === 0) {
      toast.error(t('knowledgebase.selectAtLeastOne'));
      setShowBatchDeleteDialog(false);
      return;
    }

    setShowBatchDeleteDialog(false);
    setDeleting(true);
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/batch`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            operation: 'delete',
            file_id_list: Array.from(selectedFiles),
          }),
        },
      );
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.message || `批量删除失败`);
      }
      const result = await res.json();
      toast.success(result.message || t('knowledgebase.batchDeleteSuccess', { count: selectedFiles.size }));
      setSelectedFiles(new Set()); // 清空选择
    } catch (error: any) {
      toast.error(error.message || t('knowledgebase.batchDeleteFailed'));
    } finally {
      setDeleting(false);
      fetchKbFiles();
    }
  };

  const handleBatchReprocessFiles = async () => {
    if (selectedFiles.size === 0) {
      toast.error(t('knowledgebase.selectAtLeastOne'));
      setShowBatchReprocessDialog(false);
      return;
    }

    setShowBatchReprocessDialog(false);
    
    // 获取第一个文件的chunk_config或使用知识库默认配置
    const fileIds = Array.from(selectedFiles);
    const firstFile = kbfiles.find(f => fileIds.includes(f.id));
    const defaultConfig = knowledgebase?.chunk_config || {
      parser_type: 'structure',
      chunk_size: '1000',
      chunk_overlap: '50',
    };
    
    // 初始化切片配置：优先使用第一个文件的chunk_config，否则使用知识库默认配置
    const initialConfig: any = firstFile?.chunk_config || defaultConfig;
    const config: any = {
      parser_type: initialConfig.parser_type || 'structure',
      image_caption_model: initialConfig.image_caption_model,
      image_caption_provider_name: initialConfig.image_caption_provider_name || 'openai_like',
    };
    
    if (initialConfig.parser_type === 'table') {
      config.table_config = initialConfig.table_config || {
        concat_rows: false,
        row_joiner: '\n',
        header_index_max: 0,
        format_sheet_data_to_json: false,
      };
    } else if (initialConfig.parser_type === 'paragraph') {
      config.separator = initialConfig.separator || '\n\n';
      config.chunk_size = String(initialConfig.chunk_size || 1000);
      config.chunk_overlap = String(initialConfig.chunk_overlap || 50);
    } else {
      config.separator = initialConfig.separator || '\n\n';
      config.chunk_size = String(initialConfig.chunk_size || 1000);
      config.chunk_overlap = String(initialConfig.chunk_overlap || 50);
    }
    
    setReprocessChunkConfig(config);
    setPendingReprocessFileIds(fileIds);
    setIsBatchReprocess(true);
    setReprocessChunkConfigDialogOpen(true);
  };

  const confirmBatchReprocessFiles = async () => {
    if (pendingReprocessFileIds.length === 0) return;
    
    setReprocessing(true);
    try {
      const body: any = {
        operation: 'reprocess',
        file_id_list: pendingReprocessFileIds,
      };
      
      if (reprocessChunkConfig) {
        // 转换配置格式以匹配API要求
        const chunkConfig: any = {
          parser_type: reprocessChunkConfig.parser_type,
          image_caption_model: reprocessChunkConfig.image_caption_model || null,
          image_caption_provider_name: reprocessChunkConfig.image_caption_provider_name || 'openai_like',
        };
        
        if (reprocessChunkConfig.parser_type === 'table' && reprocessChunkConfig.table_config) {
          chunkConfig.table_config = reprocessChunkConfig.table_config;
        } else {
          chunkConfig.separator = reprocessChunkConfig.separator || '\n\n';
          chunkConfig.chunk_size = reprocessChunkConfig.chunk_size ? parseInt(reprocessChunkConfig.chunk_size) : 1000;
          chunkConfig.chunk_overlap = reprocessChunkConfig.chunk_overlap ? parseInt(reprocessChunkConfig.chunk_overlap) : 50;
        }
        
        body.chunk_config = chunkConfig;
      }
      
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/batch`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
        },
      );
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.message || `批量重新解析失败`);
      }
      const result = await res.json();
      toast.success(result.message || t('knowledgebase.batchReparseSuccess', { count: pendingReprocessFileIds.length }));
      setSelectedFiles(new Set()); // 清空选择
      setReprocessChunkConfigDialogOpen(false);
      setPendingReprocessFileIds([]);
      setReprocessChunkConfig(null);
    } catch (error: any) {
      toast.error(error.message || t('knowledgebase.batchReparseFailed'));
    } finally {
      setReprocessing(false);
      fetchKbFiles();
    }
  };

  const handleSelectFile = (fileId: string, checked: boolean) => {
    setSelectedFiles((prev) => {
      const newSet = new Set(prev);
      if (checked) {
        newSet.add(fileId);
      } else {
        newSet.delete(fileId);
      }
      return newSet;
    });
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedFiles(new Set(kbfiles.map((file) => file.id)));
    } else {
      setSelectedFiles(new Set());
    }
  };

  const isAllSelected = kbfiles.length > 0 && selectedFiles.size === kbfiles.length;

  const loadPreviewContent = async (fileId: string) => {
    setPreviewLoading(true);
    setPreviewError('');
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${fileId}`,
      );
      if (!res.ok) throw new Error(t('knowledgebase.fetchFilesFailed'));
      const json_data = await res.json();
      const kb_file_data = json_data.data;

      // 将相对路径转换为完整的 HTTP 地址
      if (kb_file_data?.file_metadata?.file_url) {
        const fileUrl = kb_file_data.file_metadata.file_url;
        // 如果是相对路径（以 localdata/ 开头），转换为完整 URL
        if (fileUrl.startsWith('localdata/')) {
          const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
          kb_file_data.file_metadata.file_url = `${baseUrl}/api/knowledgebases/${fileUrl}`;
          kb_file_data.file_metadata.is_local = true;
        } else {
          kb_file_data.file_metadata.is_local = false;
        }
      }

      setPreviewFile(kb_file_data);
    } catch (err: any) {
      setPreviewError(err?.message || t('knowledgebase.loadError'));
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSaveFileSource = async () => {
    if (!currentFileId) return;
    
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${currentFileId}/source`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            file_source: fileSource,
          }),
        },
      );
      if (!res.ok) throw new Error(t('knowledgebase.sourceLinkSaveFailed'));

      const fileObj = kbfiles.filter((file) => file.id === currentFileId)[0];
      if (fileObj) {
        fileObj.file_source = fileSource;
      }
      setFileSourceOpen(false);
      setCurrentFileId('');
      toast.success(t('knowledgebase.sourceLinkSaveSuccess'));
    } catch (error: any) {
      toast.error(error.message);
    }
  };

  const selectMetadataKey = async (metadata_key: string) => {
    setMetadataEditError('');
    const emptyKeys = Object.keys(editingMetadata).filter(
      (key) => editingMetadata[key] === '',
    );
    if (emptyKeys.length > 1) throw new Error(t('knowledgebase.multipleNewItemsError'));
    else if (emptyKeys.length === 0) return;
    else {
      if (metadataValueTypes[metadata_key] === "string") {
        editingMetadata[metadata_key] = "";
      }
      else if (metadataValueTypes[metadata_key] === "number") {
        editingMetadata[metadata_key] = 0;
      }
      else {
        editingMetadata[metadata_key] = new Date();
      }
      delete editingMetadata[''];
      const updatedUsableKeys = availableMetadataKeys.filter(
        (name) => name !== metadata_key,
      );
      setAvailableMetadataKeys(updatedUsableKeys);
      console.log('selected keys for metadata: ', editingMetadata);
      setEditingMetadata({ ...editingMetadata });
    }
  };

  const handleOpenMetadata = async (file_id: string) => {
    setMetadataEditError('');
    setIsEditingMetadata(false);
    setCurrentMetadataFileId(file_id);
    try {
      const file_res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${file_id}`,
      );
      if (!file_res.ok) throw new Error(`获取 ${file_id} 失败`);
      const file_json = await file_res.json();
      setEditingMetadata(file_json.data.file_metadata);
      const usable_metadata_keys = metadataConfigs
        .map((metadata) => metadata.name)
        .filter((name) => !(name in file_json.data.file_metadata));
      setAvailableMetadataKeys(usable_metadata_keys);
      console.log('可用的metadata名称：', usable_metadata_keys);
      setMetadataDialogOpen(true);
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  const handAddFileMetadata = () => {
    if (availableMetadataKeys.length === 0) {
      setMetadataEditError(
        t('knowledgebase.noCustomMetadata'),
      );
      return;
    }
    const hasEmptyEntry = Object.keys(editingMetadata).some(
      (key) => editingMetadata[key] === '',
    );
    if (!hasEmptyEntry) {
      editingMetadata[''] = '';
      setEditingMetadata({ ...editingMetadata });
      setMetadataEditError('');
    } else {
      console.log('已经有一个待添加的项目了。');
      setMetadataEditError('');
    }
  };

  const handleDeleteMetadata = (name: string) => {
    console.log('删除metadata:', name, editingMetadata);
    if (name in editingMetadata) {
      delete editingMetadata[name];
      setEditingMetadata(editingMetadata);
      const usable_metadata_keys = metadataConfigs
        .map((metadata) => metadata.name)
        .filter((name) => !(name in editingMetadata));
      setAvailableMetadataKeys(usable_metadata_keys);
      console.log('可用的metadata名称：', usable_metadata_keys);

      setMetadataEditError('');
      console.log('已删除metadata:', name, editingMetadata);
    }
  };

  const handleRoleSelect = (
    role_id: string,
    role_name: string,
    checked: boolean,
  ) => {
    if (checked) {
      if (!activeRoleIds.includes(role_id)) {
        setActiveRoleIds([...activeRoleIds, role_id]);
        setActiveRoleNames([...activeRoleNames, role_name]);
      }
    } else {
      if (activeRoleIds.includes(role_id)) {
        setActiveRoleIds((prev) => prev.filter((id) => id !== role_id));
        setActiveRoleNames((prev) => prev.filter((name) => name !== role_name));
      }
    }
  };

  const clearAllRoles = async () => {
    setActiveRoleIds([]);
    setActiveRoleNames([]);
  };

  const checkFileRole = async (file_id: string) => {
    try {
      setEditRoleFileId(file_id);
      const roleRes = await tenantFetch(`/api/config/roles?size=100`);
      if (!roleRes.ok) {
        toast.error(t('knowledgebase.fetchRoleFailed'));
        return;
      }
      const all_roles = (await roleRes.json()).data.items;
      setRoles(all_roles);

      const permission_name = file_id;
      const res = await tenantFetch(
        `/api/config/roles/permissions?name=${permission_name}&size=100`,
      );
      if (!res.ok) {
        toast.error(t('knowledgebase.fetchPermissionFailed'));
        return;
      }

      const permission_res = await res.json();
      const role_ids = permission_res.data.items.map(
        (item: any) => item.role_id,
      );
      const role_names = all_roles
        .filter((role: any) => role_ids.includes(role.id))
        .map((role: any) => role.name);
      console.log('role_ids:', role_ids);
      console.log('role_names:', role_names);

      setActiveRoleIds(role_ids);
      setActiveRoleNames(role_names);
      setRoleDialogOpen(true);
    } catch (error: any) {
      toast.error(error.message);
    }
  };

  const saveFilePermission = async () => {
    try {
      const roleRes = await tenantFetch(
        `/api/config/roles/permissions/files/${editRoleFileId}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            role_ids: activeRoleIds,
          }),
        },
      );
      if (!roleRes.ok) {
        toast.error(t('knowledgebase.updateRoleFailed'));
        return;
      }
      console.log('更新文件角色成功：', await roleRes.json());
      setRoleDialogOpen(false);
      setEditRoleFileId('');
      toast.success(t('knowledgebase.roleSaveSuccess'));
    } catch (error: any) {
      toast.error(error.message);
    }
  };

  const handleFileUpload = async (files: FileList | null) => {
    console.log('##handleFileUpload', files);
    if (!files) {
      toast.error(t('knowledgebase.fileListEmpty'));
      return;
    }

    // 文件校验 (Demo功能，后续调整优化)
    const validFiles = Array.from(files).filter((file) => {
      // const isValidType = ['application/pdf', 'application/msword'].includes(file.type);
      const isValidSize = file.size <= 1000 * 1024 * 1024;
      // return isValidType && isValidSize;
      return isValidSize;
    });

    if (validFiles.length === 0) {
      toast.error(t('knowledgebase.selectValidFiles'));
      return;
    }

    // 上传文件
    const formData = new FormData();
    validFiles.forEach((file) => {
      formData.append('files', file);
    });

    setUploading(true);
    setUploadStep('uploading');
    setUploadProgress(0);

    try {
      // 生产环境上传大文件直连
      const API_PREFIX = process.env.NEXT_PUBLIC_DEVELOP_MODE === "true" ? "/api" : "/v1";
      console.log("上传后端地址前缀: ", API_PREFIX);

      // 使用 XMLHttpRequest 来获取上传进度
      const upload_result = await new Promise<any>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(progress);
          }
        };

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const result = JSON.parse(xhr.responseText);
              resolve(result);
            } catch (e) {
              reject(new Error(t('knowledgebase.parseResponseFailed')));
            }
          } else {
            try {
              const result = JSON.parse(xhr.responseText);
              reject(new Error(result.message || t('knowledgebase.uploadFailed')));
            } catch (e) {
              reject(new Error(t('knowledgebase.uploadFailed')));
            }
          }
        };

        xhr.onerror = () => {
          reject(new Error(t('knowledgebase.networkError')));
        };

        // 添加 auto_parse=false 参数，只上传不解析
        // 如果提供了 chunk_config，添加到 FormData
        if (uploadChunkConfig) {
          const chunkConfig: any = {
            parser_type: uploadChunkConfig.parser_type,
            image_caption_model: uploadChunkConfig.image_caption_model || null,
            image_caption_provider_name: uploadChunkConfig.image_caption_provider_name || 'openai_like',
          };
          
          if (uploadChunkConfig.parser_type === 'table' && uploadChunkConfig.table_config) {
            chunkConfig.table_config = uploadChunkConfig.table_config;
            // 添加 chunk_size
            chunkConfig.chunk_size = uploadChunkConfig.chunk_size ? parseInt(uploadChunkConfig.chunk_size) : 1000;
          } else if (uploadChunkConfig.parser_type === 'paragraph') {
            chunkConfig.separator = uploadChunkConfig.separator || '\n\n';
            chunkConfig.chunk_size = uploadChunkConfig.chunk_size ? parseInt(uploadChunkConfig.chunk_size) : 1000;
            chunkConfig.chunk_overlap = uploadChunkConfig.chunk_overlap ? parseInt(uploadChunkConfig.chunk_overlap) : 50;
          } else {
            chunkConfig.separator = uploadChunkConfig.separator || '\n\n';
            chunkConfig.chunk_size = uploadChunkConfig.chunk_size ? parseInt(uploadChunkConfig.chunk_size) : 1000;
            chunkConfig.chunk_overlap = uploadChunkConfig.chunk_overlap ? parseInt(uploadChunkConfig.chunk_overlap) : 50;
          }
          
          formData.append('chunk_config', JSON.stringify(chunkConfig));
        }
        xhr.open('POST', `${API_PREFIX}/config/knowledgebases/${kbId}/files?auto_parse=false`);
        xhr.setRequestHeader('X-TENANT-ID', tenantId);
        xhr.send(formData);
      });

      if (upload_result.code !== 200) {
        throw new Error(upload_result.message);
      }

      console.log('上传成功:', upload_result);
      
      // 保存上传的文件信息，用于后续解析
      const uploadedFileList = upload_result.data.map((file: any) => ({
        id: file.id,
        file_name: file.file_name,
        file_path: file.file_path,
        chunk_config: file.chunk_config, // 保存文件的 chunk_config
      }));
      setUploadedFiles(uploadedFileList);
      // 更新显示的 chunk_config 为文件的 chunk_config（如果文件有的话）
      if (uploadedFileList.length > 0 && uploadedFileList[0].chunk_config) {
        const fileChunkConfig = uploadedFileList[0].chunk_config;
        const config: any = {
          parser_type: fileChunkConfig.parser_type || 'structure',
          image_caption_model: fileChunkConfig.image_caption_model,
          image_caption_provider_name: fileChunkConfig.image_caption_provider_name,
        };
        
                        if (fileChunkConfig.parser_type === 'table') {
                          config.table_config = fileChunkConfig.table_config || {
                            concat_rows: false,
                            row_joiner: '\n',
                            header_index_max: 0,
                            format_sheet_data_to_json: false,
                          };
                          // 添加 chunk_size
                          config.chunk_size = String(fileChunkConfig.chunk_size || 1000);
                        } else if (fileChunkConfig.parser_type === 'paragraph') {
                          config.separator = fileChunkConfig.separator || '\n\n';
          config.chunk_size = String(fileChunkConfig.chunk_size || 1000);
          config.chunk_overlap = String(fileChunkConfig.chunk_overlap || 50);
        } else {
          config.separator = fileChunkConfig.separator || '\n\n';
          config.chunk_size = String(fileChunkConfig.chunk_size || 1000);
          config.chunk_overlap = String(fileChunkConfig.chunk_overlap || 50);
        }
        
        setUploadChunkConfig(config);
      }
      setUploadStep('uploaded');
      setUploadProgress(100);
      toast.success(t('knowledgebase.uploadSuccessToast'));
      
      // 清空文件选择框
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      fetchKbFiles();
    } catch (error: any) {
      console.error('上传失败:', error.message);
      toast.error(t('knowledgebase.uploadFailed') + ': ' + error.message);
      setUploadStep('idle');
      setUploadProgress(0);
      setUploading(false);
    }
  };

  // 构建 chunk_config 对象
  const buildChunkConfig = () => {
    if (!uploadChunkConfig) {
      return undefined;
    }

    const chunkConfig: any = {
      parser_type: uploadChunkConfig.parser_type,
      image_caption_model: uploadChunkConfig.image_caption_model || null,
      image_caption_provider_name: uploadChunkConfig.image_caption_provider_name || 'openai_like',
    };
    
    if (uploadChunkConfig.parser_type === 'table' && uploadChunkConfig.table_config) {
      chunkConfig.table_config = uploadChunkConfig.table_config;
      // 添加 chunk_size
      chunkConfig.chunk_size = (uploadChunkConfig.chunk_size === '' || !uploadChunkConfig.chunk_size) 
        ? 1000 
        : parseInt(uploadChunkConfig.chunk_size);
    } else if (uploadChunkConfig.parser_type === 'paragraph') {
      chunkConfig.separator = uploadChunkConfig.separator || '\n\n';
      // 如果为空字符串或undefined/null，使用默认值；否则转换为数字
      chunkConfig.chunk_size = (uploadChunkConfig.chunk_size === '' || !uploadChunkConfig.chunk_size) 
        ? 1000 
        : parseInt(uploadChunkConfig.chunk_size);
      chunkConfig.chunk_overlap = (uploadChunkConfig.chunk_overlap === '' || !uploadChunkConfig.chunk_overlap) 
        ? 50 
        : parseInt(uploadChunkConfig.chunk_overlap);
    } else {
      chunkConfig.separator = uploadChunkConfig.separator || '\n\n';
      // 如果为空字符串或undefined/null，使用默认值；否则转换为数字
      chunkConfig.chunk_size = (uploadChunkConfig.chunk_size === '' || !uploadChunkConfig.chunk_size) 
        ? 1000 
        : parseInt(uploadChunkConfig.chunk_size);
      chunkConfig.chunk_overlap = (uploadChunkConfig.chunk_overlap === '' || !uploadChunkConfig.chunk_overlap) 
        ? 50 
        : parseInt(uploadChunkConfig.chunk_overlap);
    }
    
    return chunkConfig;
  };

  const handleStartParse = async () => {
    if (uploadedFiles.length === 0) {
      toast.error(t('knowledgebase.noFileToParse'));
      return;
    }

    setUploadStep('parsing');

    try {
      const API_PREFIX = process.env.NEXT_PUBLIC_DEVELOP_MODE === "true" ? "/api" : "/v1";
      
      // 构建请求体，包含 files 和可选的 chunk_config
      const requestBody: any = {
        files: uploadedFiles.map(f => ({
          file_name: f.file_name,
          file_path: f.file_path,
        })),
      };
      
      // 如果有配置的 chunk_config，添加到请求体中
      const chunkConfig = buildChunkConfig();
      if (chunkConfig) {
        requestBody.chunk_config = chunkConfig;
      }
      
      // 开始解析（chunk_config 会在解析时一起更新）
      const res = await tenantFetch(
        `${API_PREFIX}/config/knowledgebases/${kbId}/files/parse`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        },
      );

      const result = await res.json();
      if (result.code !== 200) {
        throw new Error(result.message);
      }

      console.log('解析任务提交成功:', result);
      toast.success(t('knowledgebase.parseSubmitSuccess'));
      
      // 重置状态
      setUploadedFiles([]);
      setUploadStep('idle');
      setUploadProgress(0);
      setUploading(false);
      setUploadDialogOpen(false);
      setPage(1);
      fetchKbFiles();
    } catch (error: any) {
      console.error('提交解析任务失败:', error.message);
      toast.error(t('knowledgebase.parseSubmitFailed') + ': ' + error.message);
      setUploadStep('uploaded'); // 回到上传完成状态，可以重试
    }
  };

  const get_metadata_id = (name: string) => {
    console.log('get id', metadataConfigs, name);
    return metadataConfigs.filter((metadata) => metadata.name === name)[0].id;
  };

  // 格式化 datetime 类型的 metadata 值为可读的日期时间字符串
  const formatDatetimeMetadata = (value: any): string => {
    if (value === null || value === undefined || value === '') {
      return '';
    }
    
    // 将秒级时间戳转换为 Date 对象
    let timestamp: number;
    if (typeof value === 'number') {
      timestamp = value;
    } else {
      const parsed = parseFloat(String(value));
      if (isNaN(parsed)) {
        return String(value); // 如果无法解析，返回原始值
      }
      timestamp = parsed;
    }
    
    // 将秒级时间戳转换为 Date 对象
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      return String(value); // 如果日期无效，返回原始值
    }
    
    // 格式化为本地日期时间字符串
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  const validateMetadataValue = (value: any, valueType: string, name: string): { valid: boolean; error?: string; convertedValue?: any } => {
    if (value === '' || value === null || value === undefined) {
      return { valid: false, error: t('knowledgebase.metadataValueEmpty', { name }) };
    }

    if (valueType === 'string') {
      return { valid: true, convertedValue: String(value) };
    } else if (valueType === 'number') {
      const numValue = typeof value === 'number' ? value : parseFloat(String(value));
      if (isNaN(numValue)) {
        return { valid: false, error: t('knowledgebase.metadataValueInvalidNumber', { name, value: String(value) }) };
      }
      return { valid: true, convertedValue: numValue };
    } else if (valueType === 'datetime') {
      // datetime类型需要是timestamp的float value
      let timestamp: number;
      if (typeof value === 'number') {
        timestamp = value;
      } else if (typeof value === 'string') {
        // 尝试解析为数字
        const parsed = parseFloat(value);
        if (!isNaN(parsed)) {
          timestamp = parsed;
        } else {
          // 尝试解析为日期字符串
          const date = new Date(value);
          if (!isNaN(date.getTime())) {
            timestamp = date.getTime(); // 转换为秒级时间戳
          } else {
            return { valid: false, error: t('knowledgebase.metadataValueInvalidDate', { name, value: String(value) }) };
          }
        }
      } else if (value instanceof Date) {
        timestamp = value.getTime(); // 转换为秒级时间戳
      } else {
        return { valid: false, error: t('knowledgebase.metadataValueTypeError', { name }) };
      }
      return { valid: true, convertedValue: timestamp };
    }
    return { valid: true, convertedValue: value };
  };

  const saveEditMetadata = async () => {
    console.log('saveEditMetadata: ', editingMetadata);
    if (!currentMetadataFileId) return;
    
    const hasEmptyEntry = Object.keys(editingMetadata).some(
      (key) => editingMetadata[key] === '',
    );
    if (hasEmptyEntry) {
      setMetadataEditError(t('knowledgebase.metadataEmptyName'));
      return;
    }

    try {
      // 验证所有metadata值是否符合类型要求
      const metadata_entries = [];
      for (const name of Object.keys(editingMetadata).filter((name) => !default_metadata_keys.includes(name))) {
        const valueType = metadataValueTypes[name] || 'string';
        const validation = validateMetadataValue(editingMetadata[name], valueType, name);
        if (!validation.valid) {
          setMetadataEditError(validation.error || t('knowledgebase.metadataValidationFailed'));
          return;
        }
        metadata_entries.push({
          name: name,
          value: validation.convertedValue,
        });
      }
      const bodyData = {
        entries: metadata_entries,
      };
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${currentMetadataFileId}/metadata`,
        {
          method: 'POST',
          body: JSON.stringify(bodyData),
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );
      if (!res.ok) throw Error(t('knowledgebase.metadataSaveFailed'));
      const file_result = (await res.json()).data as KnowledgeBaseFile;
      const updated_kbfiles = kbfiles;
      const target_file_index = updated_kbfiles.findIndex(
        (file) => file.id === currentMetadataFileId,
      );
      updated_kbfiles[target_file_index] = file_result;
      setKbFiles(updated_kbfiles);
      console.log('更新文件成功：', updated_kbfiles);
      // 重新获取metadata列表以获取最新的count信息
      await fetchMetadataConfigs();
      setIsEditingMetadata(false);
      setMetadataDialogOpen(false);
      setCurrentMetadataFileId('');
      toast.success(t('knowledgebase.metadataSaveSuccess'));
    } catch (error: any) {
      console.log('保存metadata失败', error);
      toast.error(error.message || t('knowledgebase.metadataSaveFailed'));
    } finally {
      setMetadataEditError('');
    }
  };

  const handleAddMetadataConfig = async () => {
    if (!newMetadataName) {
      setMetadataError(t('knowledgebase.metadataRequired'));
      return;
    }

    if (metadataConfigs.some((config) => config.name === newMetadataName)) {
      setMetadataError(t('knowledgebase.metadataNameExists', { name: newMetadataName }));
      return;
    }

    const metadata_url = `/api/config/knowledgebases/${kbId}/metadata`;
    try {
      const res = await tenantFetch(metadata_url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kb_id: kbId,
          name: newMetadataName,
          value_type: newMetadataValueType,
          description: newMetadataDesc,
        }),
      });
      if (!res.ok) throw new Error(`保存metadata失败: ${await res.text()}`);
      // 重新获取metadata列表以获取最新的count信息
      await fetchMetadataConfigs();
      setNewMetadataName('');
      setMetadataError('');
      setNewMetadataValueType('string');
      setNewMetadataDesc('');
      setMetadataEditDialogOpen(false);
      toast.success(t('knowledgebase.addMetadataSuccess'));
    } catch (err: any) {
      console.log('保存知识库失败', err.message);
      setMetadataError(err.message);
    }
  };

  const handleRemoveMetadataEntry = async (id: string) => {
    const metadata_url = `/api/config/knowledgebases/${kbId}/metadata/${id}`;
    try {
      const res = await tenantFetch(metadata_url, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`删除metadata失败: ${await res.text()}`);

      // 重新获取metadata列表以获取最新的count信息
      await fetchMetadataConfigs();
      toast.success(t('knowledgebase.deleteMetadataSuccess'));
    } catch (err: any) {
      console.log('删除元数据失败。', err.message);
      toast.error(err.message || t('knowledgebase.deleteMetadataFailed'));
    }
  };

  const handleEditMetadataConfig = (metadata: MetadataConfig) => {
    setEditingMetadataConfig(metadata);
    setNewMetadataName(metadata.name);
    setNewMetadataValueType(metadata.value_type);
    setNewMetadataDesc(metadata.description || '');
    setMetadataError('');
    setMetadataEditDialogOpen(true);
  };

  const handleUpdateMetadataConfig = async () => {
    if (!editingMetadataConfig) return;
    if (!newMetadataName) {
      setMetadataError(t('knowledgebase.metadataRequired'));
      return;
    }

    if (metadataConfigs.some((config) => config.name === newMetadataName && config.id !== editingMetadataConfig.id)) {
      setMetadataError(t('knowledgebase.metadataNameExists', { name: newMetadataName }));
      return;
    }

    const metadata_url = `/api/config/knowledgebases/${kbId}/metadata/${editingMetadataConfig.id}`;
    try {
      const res = await tenantFetch(metadata_url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newMetadataName,
          value_type: newMetadataValueType,
          description: newMetadataDesc,
        }),
      });
      if (!res.ok) throw new Error(`更新metadata失败: ${await res.text()}`);
      // 重新获取metadata列表以获取最新的count信息
      await fetchMetadataConfigs();
      setMetadataEditDialogOpen(false);
      setEditingMetadataConfig(null);
      setNewMetadataName('');
      setMetadataError('');
      setNewMetadataValueType('string');
      setNewMetadataDesc('');
      toast.success(t('knowledgebase.updateMetadataSuccess'));
    } catch (err: any) {
      console.log('更新元数据失败', err.message);
      setMetadataError(err.message);
    }
  };


  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button
                  variant="link"
                  className="px-0 h-auto"
                  onClick={() => router.push('/knowledgebases')}
                >
                  {t('knowledgebase.title')}
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage className="font-semibold">{knowledgebase.name}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        {knowledgebase?.id && (
          <Badge variant="secondary" className="text-[10px] font-mono bg-muted text-muted-foreground">
            ID: {knowledgebase.id}
          </Badge>
        )}
        {knowledgebase.description && (
          <span className="text-xs text-muted-foreground truncate max-w-[240px]">
            {knowledgebase.description}
          </span>
        )}
      </HeaderPortal>
      <div className="flex-1 overflow-y-auto px-2 py-3">
        {/* Small inline view switcher: 文件管理 / 检索测试 */}
        <div className="flex items-center gap-4 px-4 pb-2 border-b border-border mb-3">
          <button
            type="button"
            onClick={() => setView('details')}
            className={`text-xs py-1.5 border-b-2 -mb-[9px] transition-colors ${
              view === 'details'
                ? 'border-primary text-foreground font-medium'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            {t('knowledgebase.fileManagement')}
          </button>
          <button
            type="button"
            onClick={() => setView('retrieval_test')}
            className={`text-xs py-1.5 border-b-2 -mb-[9px] transition-colors ${
              view === 'retrieval_test'
                ? 'border-primary text-foreground font-medium'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            {t('knowledgebase.retrievalTest')}
          </button>
        </div>
        <Tabs value={view} onValueChange={(v) => setView(v as 'details' | 'retrieval_test')}>
          <TabsList className="sr-only" aria-hidden="true">
            <TabsTrigger value="details">{t('knowledgebase.fileManagement')}</TabsTrigger>
            <TabsTrigger value="retrieval_test">{t('knowledgebase.retrievalTest')}</TabsTrigger>
          </TabsList>
          <TabsContent value="details" className="py-2">
            <div className="mb-4 rounded-lg">
              <div className="flex items-center justify-between w-full mb-4">
                <div className="flex gap-2 items-center pl-2">
                 <Search className="h-4 w-4 text-muted-foreground" />
                  <Input
                    value={fileQuery}
                    onChange={(e)=>{setFileQuery(e.target.value)}}
                    type="search_files"
                    placeholder={t('knowledgebase.searchPlaceholderShort')}
                    className="h-6 text-xs w-40"/>

                  {selectedFiles.size > 0 && (
                    <>
                      <Button
                        variant="outline"
                        className="h-6 text-xs"
                        onClick={() => setShowBatchReprocessDialog(true)}
                        disabled={reprocessing}
                      >
                        {reprocessing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            {t('knowledgebase.processing')}
                          </>
                        ) : (
                          <>
                            <CirclePlayIcon className="mr-2 h-4 w-4" />
                            {t('knowledgebase.batchReparse')} ({selectedFiles.size})
                          </>
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        className="h-6 text-xs bg-rose-100 text-rose-700 hover:bg-rose-200 hover:text-rose-800 dark:bg-rose-900/20 dark:text-rose-400 dark:hover:bg-rose-900/40"
                        onClick={() => setShowBatchDeleteDialog(true)}
                        disabled={deleting}
                      >
                        {deleting ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            {t('knowledgebase.deleting')}
                          </>
                        ) : (
                          <>
                            <Trash2Icon className="mr-2 h-4 w-4" />
                            {t('knowledgebase.batchDelete')} ({selectedFiles.size})
                          </>
                        )}
                      </Button>
                      <AlertDialog open={showBatchReprocessDialog} onOpenChange={setShowBatchReprocessDialog}>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>{t('knowledgebase.confirmBatchReparse')}</AlertDialogTitle>
                            <AlertDialogDescription>
                              {t('knowledgebase.confirmBatchReparseDesc', { count: selectedFiles.size })}
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={handleBatchReprocessFiles}
                            >
                              {t('knowledgebase.confirmReparse')}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                      <AlertDialog open={showBatchDeleteDialog} onOpenChange={setShowBatchDeleteDialog}>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>{t('knowledgebase.confirmBatchDelete')}</AlertDialogTitle>
                            <AlertDialogDescription>
                              {t('knowledgebase.confirmBatchDeleteDesc', { count: selectedFiles.size })}
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={handleBatchDeleteFiles}
                              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            >
                              {t('knowledgebase.confirmDelete')}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </>
                  )}
                </div>
                <div className="flex gap-2 items-center">
                <Dialog open={uploadDialogOpen} onOpenChange={async (open) => {
                    // 只有在非上传/解析状态时才允许关闭
                    if (!open && (uploadStep === 'uploading' || uploadStep === 'parsing')) {
                      return;
                    }
                    
                    // 注意：chunk_config 现在只在解析时更新，关闭对话框时不再单独更新
                    
                    setUploadDialogOpen(open);
                    if (!open) {
                      // 关闭时重置状态
                      setUploadStep('idle');
                      setUploadProgress(0);
                      setUploadedFiles([]);
                      setUploading(false);
                      // 重置 chunk_config 为知识库的默认配置
                      if (knowledgebase?.chunk_config) {
                        const config: any = {
                          parser_type: knowledgebase.chunk_config.parser_type || 'structure',
                          image_caption_model: knowledgebase.chunk_config.image_caption_model,
                          image_caption_provider_name: knowledgebase.chunk_config.image_caption_provider_name,
                        };
                        
                        if (knowledgebase.chunk_config.parser_type === 'table') {
                          config.table_config = knowledgebase.chunk_config.table_config || {
                            concat_rows: false,
                            row_joiner: '\n',
                            header_index_max: 0,
                            format_sheet_data_to_json: false,
                          };
                        } else if (knowledgebase.chunk_config.parser_type === 'paragraph') {
                          config.separator = knowledgebase.chunk_config.separator || '\n\n';
                          config.chunk_size = String(knowledgebase.chunk_config.chunk_size || 1000);
                          config.chunk_overlap = String(knowledgebase.chunk_config.chunk_overlap || 50);
                        } else {
                          config.separator = knowledgebase.chunk_config.separator || '\n\n';
                          config.chunk_size = String(knowledgebase.chunk_config.chunk_size || 1000);
                          config.chunk_overlap = String(knowledgebase.chunk_config.chunk_overlap || 50);
                        }
                        
                        setUploadChunkConfig(config);
                      }
                    }
                  }}>
                    <DialogTrigger asChild>
                      <Button
                        variant="default"
                        className="h-6 text-xs"
                        disabled={uploading}
                        onClick={() => setUploadDialogOpen(true)}
                      >
                        <Upload className="h-3 w-3" /> {t('knowledgebase.uploadFile')}
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle>{t('knowledgebase.uploadFile')}</DialogTitle>
                        <DialogDescription>
                          {uploadStep === 'idle' && t('knowledgebase.selectFileToUpload')}
                          {uploadStep === 'uploading' && t('knowledgebase.uploadingFile')}
                          {uploadStep === 'uploaded' && t('knowledgebase.uploadedClickParse')}
                          {uploadStep === 'parsing' && t('knowledgebase.submittingParse')}
                        </DialogDescription>
                      </DialogHeader>
                      
                      {/* 步骤1: 选择文件 */}
                      {uploadStep === 'idle' && (
                        <>
                          <div 
                            className={`flex flex-col items-center justify-center py-8 px-4 cursor-pointer border-2 border-dashed rounded-lg transition-colors ${
                              isDragging 
                                ? 'border-primary bg-primary/10' 
                                : 'hover:bg-muted/50'
                            }`}
                            onClick={() => {
                              document.getElementById('file-upload')?.click();
                            }}
                            onDragOver={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setIsDragging(true);
                            }}
                            onDragEnter={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setIsDragging(true);
                            }}
                            onDragLeave={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setIsDragging(false);
                            }}
                            onDrop={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setIsDragging(false);
                              const files = e.dataTransfer.files;
                              if (files && files.length > 0) {
                                handleFileUpload(files);
                              }
                            }}
                          >
                            <Upload className={`h-12 w-12 mb-4 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
                            <p className="text-sm text-muted-foreground text-center">
                              {t('knowledgebase.supportedFileTypes')}
                            </p>
                            <p className={`text-xs mt-2 ${isDragging ? 'text-primary font-medium' : 'text-muted-foreground'}`}>
                              {isDragging ? t('knowledgebase.releaseToUpload') : t('knowledgebase.dropFileHere')}
                            </p>
                          </div>
                        </>
                      )}

                      {/* 步骤2: 上传中 - 显示进度条 */}
                      {uploadStep === 'uploading' && (
                        <div className="flex flex-col items-center justify-center py-8 px-4">
                          <Loader2 className="h-12 w-12 text-primary mb-4 animate-spin" />
                          <p className="text-sm text-muted-foreground mb-4">{t('knowledgebase.uploadingFile')}</p>
                          <div className="w-full bg-muted rounded-full h-3">
                            <div 
                              className="bg-primary h-3 rounded-full transition-all duration-300"
                              style={{ width: `${uploadProgress}%` }}
                            />
                          </div>
                          <p className="text-sm text-muted-foreground mt-2">{uploadProgress}%</p>
                        </div>
                      )}

                      {/* 步骤3: 上传完成 - 显示文件列表和开始解析按钮 */}
                      {uploadStep === 'uploaded' && (
                        <div className="flex flex-col py-4 px-2">
                          <div className="flex items-center gap-2 mb-4">
                            <CheckCircle className="h-6 w-6 text-green-500" />
                            <span className="text-sm font-medium">{t('knowledgebase.fileUploadSuccess')}</span>
                          </div>
                          <div className="border rounded-lg p-3 mb-4 max-h-40 overflow-y-auto">
                            <p className="text-xs text-muted-foreground mb-2">{t('knowledgebase.uploadedFilesList')}</p>
                            {uploadedFiles.map((file, index) => (
                              <div key={file.id} className="text-sm py-1 border-b last:border-b-0">
                                {index + 1}. {file.file_name}
                              </div>
                            ))}
                          </div>
                          
                          {/* Chunk Config 配置 */}
                          {uploadChunkConfig && (
                            <div className="mb-4 border-t pt-4">
                              <p className="text-sm font-medium mb-3">{t('knowledgebase.chunkConfig')}</p>
                              <div className="space-y-3">
                                <div className="flex gap-3 items-center">
                                  <Label htmlFor="upload-parser-type" className="w-[80px] text-xs">
                                    {t('knowledgebase.parserType')}
                                  </Label>
                                  <Select
                                    value={uploadChunkConfig.parser_type || 'structure'}
                                    onValueChange={(value) => {
                                      setUploadChunkConfig((prev) => {
                                        if (!prev) return null;
                                        const newConfig: any = {
                                          ...prev,
                                          parser_type: value,
                                        };
                                        
                                        // 根据新的 parser_type 初始化相应的配置
                                        if (value === 'table') {
                                          newConfig.table_config = prev.table_config || {
                                            concat_rows: false,
                                            row_joiner: '\n',
                                            header_index_max: 0,
                                            format_sheet_data_to_json: false,
                                          };
                                          // 保留chunk_size，设置默认值
                                          newConfig.chunk_size = prev.chunk_size || '1000';
                                          // 清除其他类型的配置
                                          delete newConfig.chunk_overlap;
                                          delete newConfig.separator;
                                        } else if (value === 'paragraph') {
                                          newConfig.separator = prev.separator || '\n\n';
                                          newConfig.chunk_size = prev.chunk_size || '1000';
                                          newConfig.chunk_overlap = prev.chunk_overlap || '50';
                                          // 清除 table_config
                                          delete newConfig.table_config;
                                        } else {
                                          newConfig.separator = prev.separator || '\n\n';
                                          newConfig.chunk_size = prev.chunk_size || '1000';
                                          newConfig.chunk_overlap = prev.chunk_overlap || '50';
                                          // 清除 table_config
                                          delete newConfig.table_config;
                                        }
                                        
                                        return newConfig;
                                      });
                                    }}
                                  >
                                    <SelectTrigger className="w-[200px] h-7 text-xs">
                                      <SelectValue placeholder={t('knowledgebase.selectParserType')} />
                                    </SelectTrigger>
                                    <SelectContent className="text-xs">
                                      <SelectGroup>
                                        <SelectItem value="structure" className="text-xs h-5">
                                          {t('knowledgebase.structure')}
                                        </SelectItem>
                                        <SelectItem value="token" className="text-xs h-5">
                                          {t('knowledgebase.token')}
                                        </SelectItem>
                                        <SelectItem value="table" className="text-xs h-5">
                                          {t('knowledgebase.table')}
                                        </SelectItem>
                                        <SelectItem value="paragraph" className="text-xs h-5">
                                          {t('knowledgebase.paragraph')}
                                        </SelectItem>
                                      </SelectGroup>
                                    </SelectContent>
                                  </Select>
                                </div>

                                {/* Table Config */}
                                {uploadChunkConfig.parser_type === 'table' && (
                                  <div className="space-y-3">
                                    {/* 第一行：最大表头行index 和 格式化为Json */}
                                    <div className="flex gap-3 items-center">
                                      <div className="flex gap-3 items-center flex-1">
                                        <Label htmlFor="upload-header-index-max" className="w-[80px] text-xs">
                                          {t('knowledgebase.maxHeaderIndex')}
                                        </Label>
                                        <Input
                                          type="number"
                                          className="w-[200px] h-7 text-xs"
                                          id="upload-header-index-max"
                                          value={uploadChunkConfig.table_config?.header_index_max ?? ''}
                                          onChange={(e) => {
                                            setUploadChunkConfig((prev) => prev ? {
                                              ...prev,
                                              table_config: {
                                                ...prev.table_config,
                                                header_index_max: e.target.value === '' ? null : (parseInt(e.target.value) || 0),
                                              },
                                            } : null);
                                          }}
                                          min="0"
                                          placeholder={t('knowledgebase.emptyHeaderRow')}
                                        />
                                      </div>
                                      <div className="flex gap-3 items-center flex-1">
                                        <Label htmlFor="upload-format-json" className="w-[80px] text-xs">
                                          {t('knowledgebase.formatAsJson')}
                                        </Label>
                                        <Checkbox
                                          id="upload-format-json"
                                          checked={uploadChunkConfig.table_config?.format_sheet_data_to_json ?? false}
                                          onCheckedChange={(checked) => {
                                            setUploadChunkConfig((prev) => prev ? {
                                              ...prev,
                                              table_config: {
                                                ...prev.table_config,
                                                format_sheet_data_to_json: checked === true,
                                              },
                                            } : null);
                                          }}
                                        />
                                      </div>
                                    </div>
                                    {/* 第二行：合并行 和 行分隔符 */}
                                    <div className="flex gap-3 items-center">
                                      <div className="flex gap-3 items-center flex-1">
                                        <Label htmlFor="upload-concat-rows" className="w-[80px] text-xs">
                                          {t('knowledgebase.mergeRows')}
                                        </Label>
                                        <Checkbox
                                          id="upload-concat-rows"
                                          checked={uploadChunkConfig.table_config?.concat_rows ?? false}
                                          onCheckedChange={(checked) => {
                                            setUploadChunkConfig((prev) => prev ? {
                                              ...prev,
                                              table_config: {
                                                ...prev.table_config,
                                                concat_rows: checked === true,
                                              },
                                            } : null);
                                          }}
                                        />
                                      </div>
                                      <div className="flex gap-3 items-center flex-1">
                                        <Label htmlFor="upload-row-joiner" className="w-[80px] text-xs">
                                          {t('knowledgebase.rowJoiner')}
                                        </Label>
                                        <Input
                                          type="text"
                                          className="w-[200px] h-7 text-xs"
                                          id="upload-row-joiner"
                                          value={uploadChunkConfig.table_config?.row_joiner || '\n'}
                                          onChange={(e) => {
                                            setUploadChunkConfig((prev) => prev ? {
                                              ...prev,
                                              table_config: {
                                                ...prev.table_config,
                                                row_joiner: e.target.value,
                                              },
                                            } : null);
                                          }}
                                        />
                                      </div>
                                    </div>
                                    {/* 第三行：切片大小 */}
                                    <div className="flex gap-3 items-center">
                                      <Label htmlFor="upload-table-chunk-size" className="w-[80px] text-xs">
                                        {t('knowledgebase.chunkSize')}
                                      </Label>
                                      <Input
                                        type="text"
                                        inputMode="numeric"
                                        className="w-[200px] h-7 text-xs"
                                        id="upload-table-chunk-size"
                                        value={uploadChunkConfig.chunk_size ?? ''}
                                        placeholder="1000"
                                        onChange={(e) => {
                                          const value = e.target.value;
                                          // 只允许数字和空字符串
                                          if (value === '' || /^\d+$/.test(value)) {
                                            setUploadChunkConfig((prev) => prev ? {
                                              ...prev,
                                              chunk_size: value,
                                            } : null);
                                          }
                                        }}
                                      />
                                      <p className="text-xs text-muted-foreground ml-2">{t('knowledgebase.recommendedValue', { value: '1000' })}</p>
                                    </div>
                                  </div>
                                )}

                                {/* Paragraph Config - 只在 parser_type === 'paragraph' 时显示 */}
                                {uploadChunkConfig.parser_type === 'paragraph' && (
                                  <div className="space-y-3">
                                    <div className="flex gap-3 items-center">
                                      <Label htmlFor="upload-separator" className="w-[80px] text-xs">
                                        {t('knowledgebase.separator')}
                                      </Label>
                                      <Input
                                        type="text"
                                        className="w-[200px] h-7 text-xs"
                                        id="upload-separator"
                                        value={uploadChunkConfig.separator || '\n\n'}
                                        onChange={(e) => {
                                          setUploadChunkConfig((prev) => prev ? {
                                            ...prev,
                                            separator: e.target.value,
                                          } : null);
                                        }}
                                      />
                                    </div>
                                    <div className="flex gap-3 items-center">
                                      <Label htmlFor="upload-chunk-size" className="w-[80px] text-xs">
                                        {t('knowledgebase.chunkSize')}
                                      </Label>
                                      <Input
                                        type="text"
                                        inputMode="numeric"
                                        className="w-[200px] h-7 text-xs"
                                        id="upload-chunk-size"
                                        value={uploadChunkConfig.chunk_size ?? ''}
                                        placeholder="1000"
                                        onChange={(e) => {
                                          const value = e.target.value;
                                          // 只允许数字和空字符串
                                          if (value === '' || /^\d+$/.test(value)) {
                                            setUploadChunkConfig((prev) => prev ? {
                                              ...prev,
                                              chunk_size: value,
                                            } : null);
                                          }
                                        }}
                                      />
                                      <Label htmlFor="upload-chunk-overlap" className="w-[80px] text-xs ml-2">
                                        {t('knowledgebase.chunkOverlap')}
                                      </Label>
                                      <Input
                                        type="text"
                                        inputMode="numeric"
                                        className="w-[200px] h-7 text-xs"
                                        id="upload-chunk-overlap"
                                        value={uploadChunkConfig.chunk_overlap ?? ''}
                                        placeholder="50"
                                        onChange={(e) => {
                                          const value = e.target.value;
                                          // 只允许数字和空字符串
                                          if (value === '' || /^\d+$/.test(value)) {
                                            setUploadChunkConfig((prev) => prev ? {
                                              ...prev,
                                              chunk_overlap: value,
                                            } : null);
                                          }
                                        }}
                                      />
                                    </div>
                                  </div>
                                )}

                                {/* Default Config - 只在 parser_type 为 'structure' 或 'token' 时显示 */}
                                {(uploadChunkConfig.parser_type === 'structure' || uploadChunkConfig.parser_type === 'token') && (
                                  <div className="flex gap-3 items-center">
                                    <Label htmlFor="upload-chunk-size" className="w-[80px] text-xs">
                                      {t('knowledgebase.chunkSize')}
                                    </Label>
                                    <Input
                                      type="text"
                                      inputMode="numeric"
                                      className="w-[200px] h-7 text-xs"
                                      id="upload-chunk-size"
                                      value={uploadChunkConfig.chunk_size ?? ''}
                                      placeholder="1000"
                                      onChange={(e) => {
                                        const value = e.target.value;
                                        // 只允许数字和空字符串
                                        if (value === '' || /^\d+$/.test(value)) {
                                          setUploadChunkConfig((prev) => prev ? {
                                            ...prev,
                                            chunk_size: value,
                                          } : null);
                                        }
                                      }}
                                    />
                                    <Label htmlFor="upload-chunk-overlap" className="w-[80px] text-xs ml-2">
                                      {t('knowledgebase.chunkOverlap')}
                                    </Label>
                                    <Input
                                      type="text"
                                      inputMode="numeric"
                                      className="w-[200px] h-7 text-xs"
                                      id="upload-chunk-overlap"
                                      value={uploadChunkConfig.chunk_overlap ?? ''}
                                      placeholder="50"
                                      onChange={(e) => {
                                        const value = e.target.value;
                                        // 只允许数字和空字符串
                                        if (value === '' || /^\d+$/.test(value)) {
                                          setUploadChunkConfig((prev) => prev ? {
                                            ...prev,
                                            chunk_overlap: value,
                                          } : null);
                                        }
                                      }}
                                    />
                                  </div>
                                )}

                                {/* 图片理解模型 */}
                                <div className="flex gap-3 items-center">
                                  <Label htmlFor="upload-image-caption-model" className="w-[80px] text-xs">
                                    {t('knowledgebase.imageCaptionModelLabel')}
                                  </Label>
                                  <Select
                                    value={uploadChunkConfig.image_caption_model || 'DISABLED'}
                                    onValueChange={(value) => {
                                      const selectedModel = visionModels.find(m => m.model_id === value);
                                      setUploadChunkConfig((prev) => prev ? {
                                        ...prev,
                                        image_caption_model: value !== "DISABLED" ? value : undefined,
                                        image_caption_provider_name: selectedModel?.provider_name || prev.image_caption_provider_name,
                                      } : null);
                                    }}
                                  >
                                    <SelectTrigger className="w-[200px] h-7 text-xs">
                                      <SelectValue placeholder={t('knowledgebase.selectImageModel')} />
                                    </SelectTrigger>
                                    <SelectContent className="text-xs">
                                      <SelectGroup>
                                        <SelectItem value="DISABLED" className="text-xs h-5">
                                          {t('knowledgebase.disableImageModel')}
                                        </SelectItem>
                                        {visionModels.map((model) => (
                                          <SelectItem key={model.id} value={model.model_id} className="text-xs h-5">
                                            {model.model_id} ({model.model})
                                          </SelectItem>
                                        ))}
                                      </SelectGroup>
                                    </SelectContent>
                                  </Select>
                                </div>
                              </div>
                            </div>
                          )}
                          
                          <Button 
                            onClick={handleStartParse}
                            className="w-full"
                          >
                            <CirclePlayIcon className="h-4 w-4 mr-2" />
                            {t('knowledgebase.startParse')} ({uploadedFiles.length})
                          </Button>
                        </div>
                      )}

                      {/* 步骤4: 解析中 */}
                      {uploadStep === 'parsing' && (
                        <div className="flex flex-col items-center justify-center py-8 px-4">
                          <Loader2 className="h-12 w-12 text-primary mb-4 animate-spin" />
                          <p className="text-sm text-muted-foreground">{t('knowledgebase.submittingParse')}</p>
                        </div>
                      )}

                      <input
                        id="file-upload"
                        type="file"
                        className="hidden"
                        ref={fileInputRef}
                        onChange={(e) => handleFileUpload(e.target.files)}
                        multiple
                      />
                    </DialogContent>
                  </Dialog>

                  <Button
                    variant="outline"
                    className="h-6 text-xs"
                    onClick={() => {
                      fetchKbFiles();
                      toast.success(t('knowledgebase.refreshSuccess'));
                    }}
                  > 
                    <RefreshCcwIcon className="h-3 w-3"/> {t('knowledgebase.refresh')}
                  </Button>
                  <Button
                    variant="outline"
                    className="h-6 text-xs"
                    onClick={() => setMetadataConfigDialogOpen(true)}
                  >
                    <Database className="h-3 w-3"/> {t('knowledgebase.metadata')}
                  </Button>
                  <Button
                    variant="outline"
                    className="h-6 text-xs"
                    onClick={() => setSettingsDialogOpen(true)}
                  >
                    <SettingsIcon className="h-3 w-3"/> {t('knowledgebase.kbSettings')}
                  </Button>
                </div>
              </div>
              <div>
                    <Table>
                      <TableHeader>
                        <TableRow className='border-border/30 border-y h-8'>
                          <TableHead className="w-12 px-2 py-1">
                            <Checkbox
                              checked={isAllSelected}
                              onCheckedChange={handleSelectAll}
                            />
                          </TableHead>
                          <TableHead className="px-2 py-1">
                            <div className="flex gap-2 items-center max-w-[400px] text-xs text-muted-foreground">
                               {t('knowledgebase.fileName')}
                            </div>
                          </TableHead>
                          <TableHead className="text-xs text-muted-foreground px-2 py-1">{t('knowledgebase.fileSize')}</TableHead>
                          <TableHead className="text-xs text-muted-foreground px-2 py-1">{t('knowledgebase.updatedTime')}</TableHead>
                          <TableHead className="text-xs text-muted-foreground px-2 py-1">{t('knowledgebase.parserTypeCol')}</TableHead>
                          <TableHead className="px-2 py-1">
                            <FileStatusFilter
                              value={statusFilter as 'all' | 'succeeded' | 'failed' | 'pending' | 'parsing' | 'persisting'}
                              onValueChange={setStatusFilter}
                            />
                          </TableHead>
                          <TableHead className="text-xs text-muted-foreground px-2 py-1">{t('knowledgebase.actions')}</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {kbfiles.map((file) => (
                          <TableRow
                            key={file.id}
                            className="cursor-pointer hover:bg-muted/60 transition-colors h-7 border-border/30"
                            onClick={(e) => {
                              // 如果点击的是checkbox或操作按钮，不跳转
                              const target = e.target as HTMLElement;
                              if (target.closest('button') || target.closest('input[type="checkbox"]') || target.closest('[role="menuitem"]')) {
                                return;
                              }
                              router.push(
                                `/knowledgebases/${kbId}/files/${file.id}`,
                              );
                            }}
                            title={t('knowledgebase.clickToViewChunks')}
                          >
                            <TableCell className="px-2 py-0.5" onClick={(e) => e.stopPropagation()}>
                              <Checkbox
                                checked={selectedFiles.has(file.id)}
                                onCheckedChange={(checked) =>
                                  handleSelectFile(file.id, checked as boolean)
                                }
                              />
                            </TableCell>
                            <TableCell className="px-2 py-0.5">
                              <span className="truncate block w-full text-left font-medium text-xs">
                                {file.file_name}
                              </span>
                            </TableCell>
                            <TableCell className="text-xs px-2 py-0.5 text-muted-foreground">
                              {formatFileSize(Number(file.file_size))}
                            </TableCell>
                            <TableCell className="text-xs px-2 py-0.5 text-muted-foreground">
                              {formatBeijingTime(file.updated_at)}
                            </TableCell>
                            <TableCell className="text-xs px-2 py-0.5" onClick={(e) => e.stopPropagation()}>
                              {file.chunk_config?.parser_type && (
                                <Badge
                                  variant="secondary"
                                  className="bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400 cursor-pointer h-5 px-1.5 text-[10px]"
                                  onClick={() => {
                                    setViewingChunkConfig({
                                      file_name: file.file_name,
                                      chunk_config: file.chunk_config
                                    });
                                    setChunkConfigDialogOpen(true);
                                  }}
                                >
                                  <span className="h-2 w-2 rounded-full bg-blue-500 mr-1.5 inline-block"></span>
                                  {file.chunk_config.parser_type === 'structure' ? t('knowledgebase.structureShort') :
                                   file.chunk_config.parser_type === 'token' ? t('knowledgebase.tokenShort') :
                                   file.chunk_config.parser_type === 'table' ? t('knowledgebase.tableShort') :
                                   file.chunk_config.parser_type === 'paragraph' ? t('knowledgebase.paragraphShort') :
                                   file.chunk_config.parser_type}
                                </Badge>
                              )}
                            </TableCell>
                            <TableCell className="text-xs px-2 py-0.5">
                              {file.status === 'pending' ? (
                                <Badge variant="secondary" className="bg-yellow-100 text-yellow-700 hover:bg-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 h-5 px-1.5 text-[10px]">
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  {t('knowledgebase.pendingParse')}
                                </Badge>
                              ) : file.status === 'parsing' ? (
                                <Badge variant="secondary" className="bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400 h-5 px-1.5 text-[10px]">
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  {t('knowledgebase.parsing')}
                                </Badge>
                              ) : file.status === 'persisting' ? (
                                <Badge variant="secondary" className="bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400 h-5 px-1.5 text-[10px]">
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  {t('knowledgebase.persisting')}
                                </Badge>
                              ) : file.status === 'succeeded' ? (
                                <Badge variant="secondary" className="bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/20 dark:text-green-400 h-5 px-1.5 text-[10px]">
                                  <CheckCircle className="mr-1 h-3 w-3" />
                                  {t('knowledgebase.parseSuccess')}
                                </Badge>
                              ) : file.status === 'failed' ? (
                                <HoverCard>
                                  <HoverCardTrigger asChild>
                                    <Badge variant="secondary" className="bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/20 dark:text-red-400 cursor-pointer h-5 px-1.5 text-[10px]">
                                      <XCircle className="mr-1 h-3 w-3" />
                                      {t('knowledgebase.parseFailed')}
                                    </Badge>
                                  </HoverCardTrigger>
                                  <HoverCardContent className="w-80">
                                    {t('knowledgebase.errorReason')}: {file.failed_reason}
                                  </HoverCardContent>
                                </HoverCard>
                              ) : (
                                <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">{file.status}</Badge>
                              )}
                            </TableCell>
                            <TableCell className="gap-1 px-2 py-0.5" onClick={(e) => e.stopPropagation()}>
                              <div className="flex items-center gap-2">
                                <DropdownMenu
                                  open={dropdownOpen[file.id] || false}
                                  onOpenChange={(open) => {
                                    setDropdownOpen(prev => ({
                                      ...prev,
                                      [file.id]: open
                                    }));
                                  }}
                                >
                                  <DropdownMenuTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      className="h-8 w-8 p-0"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                      }}
                                    >
                                      <MoreVertical className="h-3 w-3" />
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                                    <DropdownMenuItem
                                      onSelect={(e) => {
                                        e.preventDefault();
                                        setDropdownOpen(prev => ({
                                          ...prev,
                                          [file.id]: false
                                        }));
                                        setPreviewFile(file);
                                        setPreviewOpen(true);
                                        loadPreviewContent(file.id);
                                      }}
                                    >
                                      <span className="text-xs font-medium">{t('knowledgebase.viewFile')}</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onSelect={(e) => {
                                        e.preventDefault();
                                        setDropdownOpen(prev => ({
                                          ...prev,
                                          [file.id]: false
                                        }));
                                        checkFileRole(file.id);
                                      }}
                                    >
                                      <span className="text-xs font-medium">{t('knowledgebase.permissionSettings')}</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onSelect={(e) => {
                                        e.preventDefault();
                                        setDropdownOpen(prev => ({
                                          ...prev,
                                          [file.id]: false
                                        }));
                                        handleOpenMetadata(file.id);
                                      }}
                                    >
                                      <span className="text-xs font-medium">{t('knowledgebase.metadata')}</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onSelect={(e) => {
                                        e.preventDefault();
                                        setDropdownOpen(prev => ({
                                          ...prev,
                                          [file.id]: false
                                        }));
                                        setCurrentFileId(file.id);
                                        setFileSource(file.file_source || '');
                                        setFileSourceOpen(true);
                                      }}
                                    >
                                      <span className="text-xs font-medium">{t('knowledgebase.sourceLink')}</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onSelect={(e) => {
                                        e.preventDefault();
                                        setDropdownOpen(prev => ({
                                          ...prev,
                                          [file.id]: false
                                        }));
                                        handleReprocessFile(file.id);
                                      }}
                                    >
                                      <span className="text-xs font-medium">{t('knowledgebase.reprocess')}</span> 
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem
                                      onSelect={(e) => {
                                        e.preventDefault();
                                        setDropdownOpen(prev => ({
                                          ...prev,
                                          [file.id]: false
                                        }));
                                        handleDeleteFile(file.id);
                                      }}
                                      className="text-destructive focus:text-destructive"
                                    >
                                      <span className="text-xs font-medium">{t('common.delete')}</span> 
                                    </DropdownMenuItem>
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  
                  {/* 源链接设置对话框 */}
                  <Dialog
                    open={fileSourceOpen}
                    onOpenChange={(open) => {
                      setFileSourceOpen(open);
                      if (!open) {
                        setCurrentFileId('');
                        setFileSource('');
                      }
                    }}
                  >
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle className="text-sm">{t('knowledgebase.setSourceLink')}</DialogTitle>
                        <DialogDescription className="text-xs">
                          {kbfiles.find(f => f.id === currentFileId)?.file_name || ''}
                        </DialogDescription>
                      </DialogHeader>
                      <div className="flex flex-col gap-3 py-2">
                        <div className="flex flex-col gap-2">
                          <Label className="text-xs">{t('knowledgebase.sourceLink')}</Label>
                          <Input
                            type="text"
                            placeholder={t('knowledgebase.sourceLinkPlaceholder')}
                            value={fileSource || ''}
                            onChange={(e) => {
                              setFileSource(e.target.value);
                            }}
                            className="text-xs h-7"
                          />
                        </div>
                      </div>
                      <div className="flex justify-end gap-2">
                        <Button
                          onClick={handleSaveFileSource}
                          size="sm"
                          className="text-xs h-7"
                        >
                          {t('common.save')}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs h-7"
                          onClick={() => {
                            setFileSourceOpen(false);
                            setCurrentFileId('');
                            setFileSource('');
                          }}
                        >
                          {t('common.cancel')}
                        </Button>
                      </div>
                    </DialogContent>
                  </Dialog>

                  {/* 元数据设置对话框 */}
                  <Dialog
                    open={metadataDialogOpen}
                    onOpenChange={(open) => {
                      setMetadataDialogOpen(open);
                      if (!open) {
                        setCurrentMetadataFileId('');
                        setIsEditingMetadata(false);
                        setEditingMetadata({});
                        setMetadataEditError('');
                      }
                    }}
                  >
                    <DialogContent className="sm:max-w-[750px] w-[600px] sm:w-[540px] max-h-[80vh] overflow-y-auto">
                      <DialogHeader>
                        {isEditingMetadata ? (
                          <DialogTitle className="text-sm">{t('knowledgebase.editMetadata')}</DialogTitle>
                        ) : (
                          <DialogTitle className="text-sm">{t('knowledgebase.viewMetadata')}</DialogTitle>
                        )}
                        <DialogDescription className="text-xs">
                          {kbfiles.find(f => f.id === currentMetadataFileId)?.file_name || ''}
                        </DialogDescription>
                      </DialogHeader>
                      <div className="grid flex-1 auto-rows-min gap-3 py-2">
                        <div className="text-xs">
                          {isEditingMetadata ? (
                            <Label htmlFor="sheet-custom-meta" className="text-xs pb-3">
                              Custom
                              <Button
                                variant="secondary"
                                className="w-16 h-5 text-xs"
                                onClick={handAddFileMetadata}
                              >
                                <PlusIcon className="h-3 w-3" />
                                Add
                              </Button>
                            </Label>
                          ) : (
                            <Label htmlFor="sheet-custom-meta" className="text-xs">
                              Custom
                            </Label>
                          )}
                          {Object.keys(editingMetadata).filter(
                            (key: string) =>
                              !default_metadata_keys.includes(key),
                          ).length === 0 && (
                            <p className="text-xs text-muted-foreground">
                              {t('knowledgebase.noCustomMetadataClickEdit')}
                            </p>
                          )}
                          {isEditingMetadata
                            ? Object.keys(editingMetadata)
                              .filter(
                                (key: string) =>
                                  !default_metadata_keys.includes(
                                    key,
                                  ),
                              )
                              .map((key: string) => (
                                <div
                                  className="flex space-x-2 items-center"
                                  key={key}
                                >
                                  {key !== '' ? (
                                    <div className="flex h-4 w-[128px] items-center truncate py-1">
                                      <span>{key}</span>
                                    </div>
                                  ) : (
                                    <Select
                                      onValueChange={(value) =>
                                        selectMetadataKey(value)
                                      }
                                      defaultOpen={true}
                                    >
                                      <SelectTrigger className="w-[120px] h-6 min-h-6 text-xs data-[size=default]:h-6 data-[size=sm]:h-6">
                                        <SelectValue placeholder={t('knowledgebase.selectMetadata')} />
                                      </SelectTrigger>
                                      <SelectContent className="w-[88px] text-xs">
                                        <SelectGroup>
                                          {availableMetadataKeys.map(
                                            (m_key) => (
                                              <SelectItem
                                                key={m_key}
                                                value={m_key}
                                                className="text-xs h-5"
                                              >
                                                {m_key}
                                              </SelectItem>
                                            ),
                                          )}
                                        </SelectGroup>
                                      </SelectContent>
                                    </Select>
                                  )}
                                  <div className="flex space-x-2 max-w-xs shrink-0">
                                    {metadataValueTypes[key] !==
                                      'datetime' ? (
                                        <Input
                                          type={
                                            metadataValueTypes[key] === 'number' ? 'number' : 'text'
                                          }
                                          className="w-[280px] border-transparent focus:shadow-xs radius-md h-5 grow p-0.5 text-xs rounded-md"
                                          value={
                                            editingMetadata[key] ?? ''
                                          }
                                          onChange={(e) => {
                                            const inputValue = e.target.value;
                                            const valueType = metadataValueTypes[key] || 'string';
                                            let processedValue: any = inputValue;
                                            
                                            // 对于number类型，尝试转换为数字
                                            if (valueType === 'number' && inputValue !== '') {
                                              const numValue = parseFloat(inputValue);
                                              processedValue = isNaN(numValue) ? inputValue : numValue;
                                            }
                                            
                                            setEditingMetadata({
                                              ...editingMetadata,
                                              [key]: processedValue,
                                            });
                                          }}
                                        />
                                      ) : (
                                        <DatetimeInput
                                          value={
                                            (() => {
                                              const val = editingMetadata[key];
                                              if (val === null || val === undefined || val === '') {
                                                return new Date().getTime(); // 默认当前时间（毫秒）
                                              }
                                              const timestamp = typeof val === 'number' ? val : parseFloat(String(val));
                                              return isNaN(timestamp) ? new Date().getTime() : timestamp;
                                            })()
                                          }
                                          width="md"
                                          onValueChange={(
                                            value,
                                          ) => {
                                            console.log('time input value', value);
                                            setEditingMetadata({
                                              ...editingMetadata,
                                              [key]: value,                                    
                                            });
                                          }}
                                        />
                                      )}
                                    <Button
                                      variant="outline"
                                      className="w-3 h-3 pl-3 pr-0"
                                      onClick={() =>
                                        handleDeleteMetadata(key)
                                      }
                                    >
                                      <Trash2Icon className="h-3 w-3" />
                                    </Button>
                                  </div>
                                </div>
                              ))
                            : Object.keys(editingMetadata)
                              .filter(
                                (key: string) =>
                                  !default_metadata_keys.includes(
                                    key,
                                  ),
                              )
                              .map((key: string) => (
                                <div
                                  className="flex items-start space-x-2"
                                  key={key}
                                >
                                  <div className="system-xs-medium w-[128px] shrink-0 items-center truncate py-1 text-text-tertiary font-semibold">
                                    {key}
                                  </div>
                                  <div className="max-w-xs shrink-0">
                                    <div className="system-xs-regular py-1 text-text-secondary max-w-xs truncate">
                                      {metadataValueTypes[key] === 'datetime' 
                                        ? formatDatetimeMetadata(editingMetadata[key])
                                        : editingMetadata[key]}
                                    </div>
                                  </div>
                                </div>
                              ))}
                        </div>
                        <div className="text-xs">
                          <Label htmlFor="sheet-custom-meta" className="text-xs pb-2">
                            {t('knowledgebase.builtinMetadata')}
                          </Label>
                          {Object.keys(editingMetadata)
                            .filter((key) =>
                              default_metadata_keys.includes(key),
                            )
                            .map((key) => (
                              <div
                                className="flex items-start space-x-2"
                                key={key}
                              >
                                <div className="system-xs-medium w-[128px] shrink-0 items-center truncate py-1 text-text-tertiary font-semibold">
                                  {key}
                                </div>
                                <div className="max-w-xs shrink-0">
                                  <div className="system-xs-regular py-1 text-text-secondary truncate">
                                    {metadataValueTypes[key] === 'datetime' 
                                      ? formatDatetimeMetadata(editingMetadata[key])
                                      : editingMetadata[key]}
                                  </div>
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                      <div className="flex flex-col gap-2">
                        {metadataEditError !== '' && (
                          <Alert variant="destructive" className="text-xs py-2">
                            <AlertCircleIcon className="h-3 w-3" />
                            <AlertDescription className="text-xs">
                              <p>{metadataEditError}</p>
                            </AlertDescription>
                          </Alert>
                        )}
                        <div className="flex gap-2 justify-end">
                          {isEditingMetadata ? (
                            <Button
                              type="button"
                              onClick={saveEditMetadata}
                              size="sm"
                              className="text-xs h-7"
                            >
                              保存
                            </Button>
                          ) : (
                            <Button
                              type="button"
                              onClick={() =>
                                setIsEditingMetadata(true)
                              }
                              size="sm"
                              className="text-xs h-7"
                            >
                              Edit
                            </Button>
                          )}
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => {
                              setMetadataDialogOpen(false);
                              setCurrentMetadataFileId('');
                              setIsEditingMetadata(false);
                              setEditingMetadata({});
                              setMetadataEditError('');
                            }}
                          >
                            {t('common.close')}
                          </Button>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>

                  {/* 权限设置对话框 */}
                  <Dialog
                    open={roleDialogOpen}
                    onOpenChange={(open) => {
                      setRoleDialogOpen(open);
                      if (!open) {
                        setEditRoleFileId('');
                        setActiveRoleIds([]);
                        setActiveRoleNames([]);
                      }
                    }}
                  >
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle className="text-sm">{t('knowledgebase.docPermission')}</DialogTitle>
                        <DialogDescription className="text-xs">
                          {kbfiles.find(f => f.id === editRoleFileId)?.file_name || ''}
                        </DialogDescription>
                      </DialogHeader>
                      <div className="grid flex-1 auto-rows-min gap-4 py-1">
                        <div>
                          {activeRoleNames.length > 0 ? (
                            <div>
                              <div className="text-xs">
                                以下角色有查看/搜索该文档的权限
                              </div>

                              <div className="flex pt-3 gap-1.5 items-center">
                                {activeRoleNames.map((name) => (
                                  <Badge
                                    variant="secondary"
                                    className="h-5 text-xs"
                                    key={name}
                                  >
                                    {name}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          ) : (
                            <div className="text-xs text-muted-foreground">
                              所有角色都有查看/搜索该文档的权限。添加角色来限制文档访问。
                            </div>
                          )}
                        </div>
                        <div className="grid gap-1">
                          <div className="flex items-center">
                            <Label
                              htmlFor="kb_selection"
                              className="w-[90px] text-xs"
                            >
                              角色选择
                            </Label>
                            <div className=" pr-2 flex items-center gap-8">
                              {roles.length > 0 ? (
                                <>
                                  <DropdownMenu modal={true}>
                                    <DropdownMenuTrigger asChild>
                                      <Button
                                        variant="outline"
                                        className="text-xs text-muted-foreground h-7"
                                      >
                                        已选{activeRoleIds.length}
                                        个，可多选 <ChevronDownIcon className="h-3 w-3" />
                                      </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="w-56">
                                      <DropdownMenuLabel className="text-xs">
                                        角色
                                      </DropdownMenuLabel>
                                      <DropdownMenuSeparator />
                                      {roles.map((role) => (
                                        <DropdownMenuCheckboxItem
                                          key={role.id}
                                          checked={activeRoleIds.includes(
                                            role.id,
                                          )}
                                          onCheckedChange={(
                                            checked,
                                          ) =>
                                            handleRoleSelect(
                                              role.id,
                                              role.name,
                                              checked,
                                            )
                                          }
                                          onSelect={(e) =>
                                            e.preventDefault()
                                          }
                                          className="text-xs"
                                        >
                                          {role.name}
                                        </DropdownMenuCheckboxItem>
                                      ))}
                                    </DropdownMenuContent>
                                  </DropdownMenu>
                                  {activeRoleIds.length > 0 && (
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="h-6 w-6 text-xs text-muted-foreground"
                                      onClick={clearAllRoles}
                                      title={t('knowledgebase.clearAllRoles')}
                                    >
                                      清空选择<XCircle className="h-3 w-3" />
                                    </Button>
                                  )}
                                </>
                              ) : (
                                <div>
                                  <p className="text-xs text-muted-foreground">
                                    {t('knowledgebase.noRoleConfig')}
                                  </p>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="flex gap-2 justify-end">
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs h-7"
                          onClick={() => {
                            setRoleDialogOpen(false);
                            setEditRoleFileId('');
                            setActiveRoleIds([]);
                            setActiveRoleNames([]);
                          }}
                        >
                          {t('common.cancel')}
                        </Button>
                        <Button 
                          onClick={saveFilePermission}
                          size="sm"
                          className="text-xs h-7"
                        >
                          保存
                        </Button>
                      </div>
                    </DialogContent>
                  </Dialog>

                  {/* 预览对话框 */}
                  <Dialog open={previewOpen} onOpenChange={(open) => {
                    if (!open) {
                      setPreviewOpen(false);
                      setPreviewFile(null);
                    }
                  }}>
                    <DialogContent className="flex flex-col h-[calc(100%-10rem)] !max-w-[calc(100%-20rem)]">
                      <DialogHeader className="flex-none h-1/10">
                        <DialogTitle>{previewFile?.file_name}</DialogTitle>
                        <DialogDescription>{t('knowledgebase.filePreview')}</DialogDescription>
                      </DialogHeader>
                      {previewLoading ? (
                        <div className="flex items-center justify-center h-full">
                          <Loader2 className="h-6 w-6 animate-spin" />
                        </div>
                      ) : previewError ? (
                        <div className="text-red-500">{previewError}</div>
                      ) : (
                        <div className="flex-grow overflow-y-auto">
                          {previewFile?.file_extension === '.pdf' ? (
                            <iframe
                              src={previewFile?.file_metadata?.file_url}
                              width="100%"
                              height="100%"
                              title="PDF预览"
                            ></iframe>
                          ) : previewFile?.file_extension === '.jpg' ||
                            previewFile?.file_extension === '.png' ||
                            previewFile?.file_extension === '.jpeg' ? (
                            <img
                              src={previewFile?.file_metadata?.file_url}
                              width="100%"
                              height="100%"
                              title={t('knowledgebase.imagePreview')}
                            ></img>
                          ) : previewFile?.file_extension === '.docx' ||
                            previewFile?.file_extension === '.xlsx' ||
                            previewFile?.file_extension === '.pptx' ? (
                            <iframe
                              src={previewFile?.file_metadata?.is_local ? previewFile?.file_metadata?.file_url : `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(
                                String(previewFile?.file_metadata?.file_url),
                              )}`}
                              width="100%"
                              height="100%"
                              title={t('knowledgebase.filePreview')}
                            />
                          ) : previewFile?.file_extension === '.mp4' ||
                            previewFile?.file_extension === '.avi' ||
                            previewFile?.file_extension === '.mov' ||
                            previewFile?.file_extension === '.wmv' ||
                            previewFile?.file_extension === '.flv' ||
                            previewFile?.file_extension === '.mkv' ||
                            previewFile?.file_extension === '.webm' ? (
                            <video
                              src={previewFile?.file_metadata?.file_url}
                              controls
                              autoPlay={false}
                              style={{
                                width: '100%',
                                maxHeight: '100%',
                                objectFit: 'contain',
                              }}
                            >
                              {t('knowledgebase.videoNotSupported')}
                            </video>
                          ) : previewFile?.file_extension === '.md' ||
                            previewFile?.file_extension === '.txt' ? (
                            <MarkdownViewer file_url={previewFile?.file_metadata?.file_url || ''} />
                          ) : previewFile?.file_extension === '.jsonl' ? (
                            <JsonlViewer file_url={previewFile?.file_metadata?.file_url || ''} />
                          ) : previewFile?.file_extension === '.html' ? (
                            <HtmlViewer file_url={previewFile?.file_metadata?.file_url || ''} />
                          ) : (
                            <div>
                              {t('knowledgebase.previewNotSupported')}
                              <a
                                href={previewFile?.file_metadata?.file_url}
                                className="text-blue-500 hover:underline ml-2"
                              >
                                {t('knowledgebase.downloadFile')}
                              </a>
                            </div>
                          )}
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                
                { kbfiles.length === 0 && (
                  <p className="text-muted-foreground mx-auto text-xs py-15 text-center bg-gray-50 rounded-lg">{t('knowledgebase.noFiles')}</p>
                )}
                <PaginationComponent
                  currentPage={page}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              </div>
          </TabsContent>
          <TabsContent value="retrieval_test" className="py-2 flex flex-col h-full min-h-0">
            <div className="flex flex-col flex-1 min-h-0 px-4 gap-3">
              {/* 顶部：搜索栏 + 工具栏 */}
              <div className="flex flex-col gap-2 flex-shrink-0">
                <div className="flex items-center gap-2">
                  <div className="flex-1 relative">
                    <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                    <Input
                      type="text"
                      id="search_query"
                      placeholder={t('knowledgebase.queryPlaceholder')}
                      onChange={handleQueryInputChange}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleSearchSubmit();
                        }
                      }}
                      className="w-full text-xs pl-8 h-9"
                    />
                  </div>
                  <Button
                    type="button"
                    onClick={handleSearchSubmit}
                    className="text-xs h-9 px-4"
                    size="sm"
                  >
                    <SearchIcon className="h-3.5 w-3.5 mr-1" />
                    {t('knowledgebase.startQuery')}
                  </Button>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button variant="outline" size="sm" className="text-xs h-7">
                        <FilterIcon className="h-3 w-3 mr-1" />
                        {t('knowledgebase.metadata')}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[500px] max-h-[400px] overflow-y-auto">
                      <ConditionGroupEditor
                        group={conditionGroup}
                        onChange={setConditionGroup}
                        metadataConfigs={metadataConfigs}
                        metadataValueTypes={metadataValueTypes}
                        t={t}
                      />
                    </PopoverContent>
                  </Popover>
                  <Input
                    className="w-40 text-xs h-7"
                    placeholder={t('knowledgebase.userIdPlaceholder')}
                    value={user}
                    onChange={(e) => setUser(e.target.value)}
                  />
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button variant="outline" size="sm" className="text-xs h-7">
                        <SettingsIcon className="h-3 w-3 mr-1" />
                        {t('knowledgebase.retrievalSettingsCard')}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[460px] p-0" align="start">
                      <div className="px-4 py-3 border-b border-border">
                        <h4 className="text-sm font-semibold">{t('knowledgebase.retrievalSettingsCard')}</h4>
                      </div>
                      <div className="p-4 space-y-4 max-h-[400px] overflow-y-auto">
                        {/* 检索策略 */}
                        <div className="space-y-1.5">
                          <Label className="text-xs font-medium">{t('knowledgebase.retrievalStrategy')}</Label>
                          <ToggleGroup
                            type="single"
                            value={retrievalSetting.retrieval_mode || 'hybrid'}
                            onValueChange={(value) => {
                              setRetrievalSetting((prev) => ({
                                ...prev,
                                retrieval_mode: value,
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
                        </div>

                        {/* Top-K / Similarity */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <Label htmlFor="top_k" className="text-xs font-medium">Top K</Label>
                              <span className="text-xs font-medium tabular-nums">{retrievalSetting.top_k ?? 5}</span>
                            </div>
                            <Slider
                              id="top_k"
                              defaultValue={[5]}
                              max={100}
                              min={1}
                              step={1}
                              value={[retrievalSetting.top_k ?? 5]}
                              onValueChange={(value: number[]) => {
                                setRetrievalSetting((prev) => ({ ...prev, top_k: value[0] }));
                              }}
                            />
                          </div>
                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <Label htmlFor="similarity_threshold" className="text-xs font-medium">
                                {t('knowledgebase.similarityThreshold')}
                              </Label>
                              <span className="text-xs font-medium tabular-nums">
                                {retrievalSetting.similarity_threshold?.toFixed(2) ?? '0.20'}
                              </span>
                            </div>
                            <Slider
                              id="similarity_threshold"
                              defaultValue={[0.2]}
                              max={1}
                              min={0}
                              step={0.01}
                              value={[retrievalSetting.similarity_threshold ?? 0.2]}
                              onValueChange={(value: number[]) => {
                                setRetrievalSetting((prev) => ({ ...prev, similarity_threshold: value[0] }));
                              }}
                            />
                          </div>
                        </div>

                        {retrievalSetting.retrieval_mode === 'hybrid' && !retrievalSetting.enable_rerank && (
                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <Label htmlFor="vector_weight" className="text-xs font-medium">
                                {t('knowledgebase.vectorWeight')}
                              </Label>
                              <span className="text-xs font-medium tabular-nums">
                                {retrievalSetting.vector_weight ?? 0.5}
                              </span>
                            </div>
                            <Slider
                              id="vector_weight"
                              min={0}
                              max={1}
                              step={0.1}
                              value={[retrievalSetting.vector_weight ?? 0.5]}
                              onValueChange={(value) =>
                                setRetrievalSetting((prev) => ({ ...prev, vector_weight: value[0] }))
                              }
                            />
                            <p className="text-[11px] text-muted-foreground">{t('knowledgebase.vectorWeightTip')}</p>
                          </div>
                        )}

                        {/* 开启重排序 */}
                        <div className="rounded-md bg-muted/30 p-3 space-y-3">
                          <label htmlFor="enable_rerank" className="flex items-center gap-2 cursor-pointer">
                            <Checkbox
                              id="enable_rerank"
                              checked={retrievalSetting.enable_rerank ?? false}
                              onCheckedChange={(checked) => {
                                setRetrievalSetting((prev) => ({ ...prev, enable_rerank: Boolean(checked) }));
                              }}
                              className="h-3.5 w-3.5"
                            />
                            <span className="text-xs font-medium">{t('knowledgebase.enableRerank')}</span>
                          </label>
                          {retrievalSetting.enable_rerank && (
                            <div className="grid grid-cols-2 gap-3">
                              <div className="space-y-1.5">
                                <Label htmlFor="rerank_model" className="text-xs font-medium">
                                  {t('knowledgebase.rerankModelLabel')}
                                </Label>
                                <Select
                                  value={retrievalSetting.rerank_model || ''}
                                  onValueChange={(value) => {
                                    setRetrievalSetting((prev) => ({
                                      ...prev,
                                      rerank_model: value,
                                      rerank_provider_name: retrievalSetting.rerank_provider_name || '',
                                    }));
                                  }}
                                >
                                  <SelectTrigger className="h-8 text-xs w-full">
                                    <SelectValue placeholder={t('knowledgebase.selectRerankModel')} />
                                  </SelectTrigger>
                                  <SelectContent className="text-xs">
                                    <SelectGroup>
                                      {rerankerModels.map((model) => (
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
                                    Rerank Top K
                                  </Label>
                                  <span className="text-xs font-medium tabular-nums">
                                    {retrievalSetting.rerank_top_k ?? 5}
                                  </span>
                                </div>
                                <Slider
                                  id="rerank_top_k"
                                  defaultValue={[5]}
                                  max={20}
                                  min={1}
                                  step={1}
                                  value={[retrievalSetting.rerank_top_k ?? 5]}
                                  onValueChange={(value: number[]) => {
                                    setRetrievalSetting((prev) => ({ ...prev, rerank_top_k: value[0] }));
                                  }}
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center justify-between gap-2 px-4 py-2.5 border-t border-border bg-muted/20">
                        <div className="text-[11px] text-muted-foreground flex items-center gap-1">
                          <InfoIcon className="w-3 h-3" />
                          {t('knowledgebase.saveChangesRetrievalHint')}
                        </div>
                        <Button
                          type="button"
                          onClick={handleSaveRetrievalSetting}
                          className="text-xs h-7"
                          size="sm"
                        >
                          <Save className="w-3 h-3 mr-1" />
                          {t('knowledgebase.applyToKbSettings')}
                        </Button>
                      </div>
                    </PopoverContent>
                  </Popover>
                </div>
              </div>

              {/* 结果列表 */}
              <div className="flex-1 overflow-y-auto min-h-0 pr-1">
                {searching && (
                  <div className="space-y-2 py-2">
                    {[0, 1, 2].map((i) => (
                      <div key={i} className="flex items-center space-x-3 p-3 rounded-md border border-border">
                        <Skeleton className="h-6 w-6 rounded-full" />
                        <div className="space-y-1.5 flex-1">
                          <Skeleton className="h-3 w-[40%]" />
                          <Skeleton className="h-2.5 w-[90%]" />
                          <Skeleton className="h-2.5 w-[70%]" />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {!searching && searchError && (
                  <Alert variant="destructive">
                    <AlertCircleIcon className="h-4 w-4" />
                    <AlertTitle className="text-sm">{t('knowledgebase.retrievalFailed')}</AlertTitle>
                    <AlertDescription className="text-xs mt-1">{searchError}</AlertDescription>
                  </Alert>
                )}
                {!searching && !searchError && searchrecords.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground">
                    <SearchIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <h2 className="text-xs font-medium">{t('knowledgebase.noRelatedChunks')}</h2>
                    <p className="mt-1 text-[11px]">{t('knowledgebase.tryAdjustSearch')}</p>
                  </div>
                )}
                {!searching && searchrecords.length > 0 && (
                  <div className="flex flex-col gap-1.5 w-full pb-4">
                    {searchrecords.map((chunk, i) => {
                      const isExpanded = expandedCards[i] || false;
                      return (
                        <div
                          key={i}
                          className="w-full rounded-md border border-border hover:border-primary/30 hover:bg-muted/40 transition-colors cursor-pointer px-3 py-2"
                          onClick={() => {
                            setExpandedCards((prev) => ({ ...prev, [i]: !prev[i] }));
                          }}
                        >
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <div className="flex items-center gap-1.5 flex-wrap min-w-0">
                              <Badge className="bg-rose-500/10 hover:bg-rose-500/10 text-rose-600 border-rose-500/30 shadow-none rounded-full text-[10px] h-4 px-1.5 shrink-0">
                                #{i + 1}
                              </Badge>
                              <Badge className="bg-amber-500/10 hover:bg-amber-500/10 text-amber-600 border-amber-500/30 shadow-none rounded-full text-[10px] h-4 px-1.5 shrink-0">
                                {t('knowledgebase.score')}: {chunk.score.toFixed(4)}
                              </Badge>
                              <Badge className="bg-blue-500/10 hover:bg-blue-500/10 text-blue-600 border-blue-500/30 shadow-none rounded-full text-[10px] h-4 px-1.5 shrink-0 max-w-[240px] truncate">
                                {chunk.title}
                              </Badge>
                              {chunk.metadata.rerank && (
                                <Badge className="bg-green-500/10 hover:bg-green-500/10 text-green-600 border-green-500/30 shadow-none rounded-full text-[10px] h-4 px-1.5 shrink-0">
                                  Rerank
                                </Badge>
                              )}
                            </div>
                            {isExpanded ? (
                              <ChevronUp className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                            ) : (
                              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                            )}
                          </div>
                          <div
                            className={`text-[11px] leading-relaxed break-words text-foreground/80 ${
                              !isExpanded ? 'line-clamp-2' : ''
                            }`}
                          >
                            {chunk.content.replace(/\n/g, '\\n')}
                          </div>
                          {chunk.metadata?.images_info?.length > 0 && (
                            <div className="flex gap-2 mt-2 flex-wrap">
                              {chunk.metadata.images_info.map((meta, index) => (
                                <PhotoProvider
                                  key={index}
                                  maskOpacity={0.8}
                                  overlayRender={() => (
                                    <div className="absolute left-0 bottom-0 p-3 w-full min-h-30 text-xs text-slate-300 z-50 bg-black/50">
                                      <div>{t('knowledgebase.imageDesc')}：{meta.desc}</div>
                                    </div>
                                  )}
                                >
                                  <PhotoView key={index} src={meta.url}>
                                    <img
                                      src={meta.url}
                                      className="w-8 h-8 object-cover rounded-md cursor-pointer"
                                    />
                                  </PhotoView>
                                </PhotoProvider>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                    <p className="text-[11px] text-center text-muted-foreground pt-4 pb-2">
                      {t('knowledgebase.noMoreContent')}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* 知识库设置Dialog */}
      <Dialog open={settingsDialogOpen} onOpenChange={setSettingsDialogOpen}>
        <DialogContent className="sm:max-w-3xl lg:max-w-4xl h-[90vh] flex flex-col gap-0 p-0 overflow-hidden">
          <DialogHeader className="space-y-1 px-6 pt-5 pb-3 border-b border-border flex-none">
            <DialogTitle className="flex items-center gap-2">
              <SettingsIcon className="w-4 h-4 text-primary" />
              {t('knowledgebase.kbSettings')}
            </DialogTitle>
            <DialogDescription className="text-xs">
              {knowledgebase.name}
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 min-h-0">
            <KbConfigCard
              isCreate={false}
              kbConfig={knowledgebase}
              onSaveSuccess={(updated) => {
                handleSaveSuccess(updated);
                setSettingsDialogOpen(false);
              }}
              onCancel={() => setSettingsDialogOpen(false)}
            />
          </div>
        </DialogContent>
      </Dialog>

      {/* 元数据管理Dialog */}
      <Dialog open={metadataConfigDialogOpen} onOpenChange={setMetadataConfigDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="text-sm">{t('knowledgebase.metadataConfig')}</DialogTitle>
            <DialogDescription className="text-xs">
              {t('knowledgebase.metadataConfigDesc')}
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-3 py-2">
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7 w-full"
              onClick={() => {
                setNewMetadataName('');
                setNewMetadataValueType('string');
                setNewMetadataDesc('');
                setMetadataError('');
                setEditingMetadataConfig(null);
                setMetadataEditDialogOpen(true);
              }}
            >
              <PlusIcon className="h-3 w-3 mr-1" /> {t('knowledgebase.addMetadata')}
            </Button>
            <div className="flex flex-col gap-2 max-h-[400px] overflow-y-auto">
              {metadataConfigs.length === 0 ? (
                <div className="text-center py-4 text-xs text-muted-foreground">
                  {t('knowledgebase.noMetadataConfig')}
                </div>
              ) : (
                metadataConfigs.map((metadata) => (
                  <div
                    key={metadata.id}
                    className="flex items-center justify-between p-2 border rounded hover:bg-muted/50 group h-8"
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="text-xs font-medium truncate">{metadata.name}</span>
                      <span className="text-xs text-muted-foreground shrink-0">
                        {metadata.value_type}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground shrink-0">
                        {metadata.count ?? 0} docs
                      </span>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => handleEditMetadataConfig(metadata)}
                      >
                        <Edit className="h-3 w-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                        onClick={() => handleRemoveMetadataEntry(metadata.id)}
                      >
                        <Trash2Icon className="h-3 w-3" />
                      </Button>
                    </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* 添加/编辑元数据Dialog */}
      <Dialog open={metadataEditDialogOpen} onOpenChange={(open) => {
        setMetadataEditDialogOpen(open);
        if (!open) {
          setEditingMetadataConfig(null);
          setNewMetadataName('');
          setNewMetadataValueType('string');
          setNewMetadataDesc('');
          setMetadataError('');
        }
      }}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="text-sm">
              {editingMetadataConfig ? t('knowledgebase.editMetadata') : t('knowledgebase.addMetadata')}
            </DialogTitle>
            <DialogDescription className="text-xs">
              {t('knowledgebase.metadataNameHint')}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-2 py-2">
            <div className="grid gap-2">
              <Label htmlFor="metadata_key" className="text-xs">{t('knowledgebase.metadataName')}</Label>
              <Input
                id="metadata_key"
                className="h-6 text-xs"
                value={newMetadataName}
                onChange={(e) => setNewMetadataName(e.target.value)}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="metadata_value_type" className="text-xs">
                值类型
              </Label>
              <Select
                value={newMetadataValueType}
                onValueChange={(value) => setNewMetadataValueType(value)}
              >
                <SelectTrigger className="w-[180px] h-6 text-xs">
                  <SelectValue placeholder={t('knowledgebase.selectValueType')} />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    <SelectLabel className="text-xs">{t('knowledgebase.valueType')}</SelectLabel>
                    <SelectItem value="string" className="text-xs h-5">
                      String
                    </SelectItem>
                    <SelectItem value="number" className="text-xs h-5">
                      Number
                    </SelectItem>
                    <SelectItem value="datetime" className="text-xs h-5">
                      DateTime
                    </SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="metadata_desc" className="text-xs">{t('knowledgebase.metadataDesc')}</Label>
              <Input
                id="metadata_desc"
                className="h-6 text-xs"
                placeholder={t('knowledgebase.metadataDescPlaceholder')}
                value={newMetadataDesc}
                onChange={(e) => setNewMetadataDesc(e.target.value)}
              />
            </div>
          </div>
          {metadataError ? (
            <Alert variant="destructive" className="text-xs py-2">
              <AlertCircleIcon className="h-3 w-3" />
              <AlertTitle className="text-xs">{t('knowledgebase.operationFailed')}</AlertTitle>
              <AlertDescription className="text-xs">
                <p>{metadataError}</p>
              </AlertDescription>
            </Alert>
          ) : null}
          <div className="flex gap-2 justify-end">
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7"
              onClick={() => {
                setMetadataEditDialogOpen(false);
                setEditingMetadataConfig(null);
                setNewMetadataName('');
                setNewMetadataValueType('string');
                setNewMetadataDesc('');
                setMetadataError('');
              }}
            >
              {t('common.cancel')}
            </Button>
            <Button
              type="button"
              size="sm"
              className="text-xs h-7"
              onClick={() => {
                if (editingMetadataConfig) {
                  handleUpdateMetadataConfig();
                } else {
                  handleAddMetadataConfig();
                }
              }}
            >
              保存
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* 切片配置查看对话框 */}
      <Dialog open={chunkConfigDialogOpen} onOpenChange={(open) => {
        setChunkConfigDialogOpen(open);
        if (!open) {
          setViewingChunkConfig(null);
        }
      }}>
        <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-sm">{t('knowledgebase.chunkConfig')}</DialogTitle>
            <DialogDescription className="text-xs">
              {viewingChunkConfig?.file_name}
            </DialogDescription>
          </DialogHeader>
          {viewingChunkConfig?.chunk_config && (
            <div className="space-y-4">
              {/* 基本信息 */}
              <div className="space-y-2">
                <Label className="text-xs font-semibold">{t('knowledgebase.parserType')}</Label>
                <div className="text-xs text-muted-foreground">
                  {viewingChunkConfig.chunk_config.parser_type === 'structure' ? t('knowledgebase.structureShort') :
                   viewingChunkConfig.chunk_config.parser_type === 'token' ? t('knowledgebase.tokenShort') :
                   viewingChunkConfig.chunk_config.parser_type === 'table' ? t('knowledgebase.tableShort') :
                   viewingChunkConfig.chunk_config.parser_type === 'paragraph' ? t('knowledgebase.paragraphShort') :
                   viewingChunkConfig.chunk_config.parser_type}
                </div>
              </div>

              {/* 图片理解配置 */}
              {(viewingChunkConfig.chunk_config.image_caption_model || viewingChunkConfig.chunk_config.image_caption_provider_name) && (
                <div className="space-y-2">
                  <Label className="text-xs font-semibold">{t('knowledgebase.imageCaptionConfig')}</Label>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    {viewingChunkConfig.chunk_config.image_caption_model && (
                      <div>{t('knowledgebase.modelLabel')}: {viewingChunkConfig.chunk_config.image_caption_model}</div>
                    )}
                    {viewingChunkConfig.chunk_config.image_caption_provider_name && (
                      <div>{t('knowledgebase.providerLabel')}: {viewingChunkConfig.chunk_config.image_caption_provider_name}</div>
                    )}
                  </div>
                </div>
              )}

              {/* 表格配置 */}
              {viewingChunkConfig.chunk_config.parser_type === 'table' && viewingChunkConfig.chunk_config.table_config && (
                <div className="space-y-2">
                  <Label className="text-xs font-semibold">{t('knowledgebase.tableConfigLabel')}</Label>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div>{t('knowledgebase.maxHeaderIndex')}: {viewingChunkConfig.chunk_config.table_config.header_index_max ?? t('knowledgebase.notSet')}</div>
                    <div>{t('knowledgebase.formatAsJson')}: {viewingChunkConfig.chunk_config.table_config.format_sheet_data_to_json ? t('common.yes') : t('common.no')}</div>
                    <div>{t('knowledgebase.mergeRows')}: {viewingChunkConfig.chunk_config.table_config.concat_rows ? t('common.yes') : t('common.no')}</div>
                    <div>{t('knowledgebase.rowJoiner')}: {viewingChunkConfig.chunk_config.table_config.row_joiner ? `"${viewingChunkConfig.chunk_config.table_config.row_joiner}"` : t('knowledgebase.notSet')}</div>
                    {viewingChunkConfig.chunk_config.table_config.sheet_column_filters && (
                      <div>{t('knowledgebase.columnFilters')}: {Array.isArray(viewingChunkConfig.chunk_config.table_config.sheet_column_filters) 
                        ? viewingChunkConfig.chunk_config.table_config.sheet_column_filters.join(', ')
                        : viewingChunkConfig.chunk_config.table_config.sheet_column_filters}</div>
                    )}
                  </div>
                </div>
              )}

              {/* 段落配置 */}
              {viewingChunkConfig.chunk_config.parser_type === 'paragraph' && (
                <div className="space-y-2">
                  <Label className="text-xs font-semibold">{t('knowledgebase.paragraphConfigLabel')}</Label>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div>{t('knowledgebase.separator')}: {viewingChunkConfig.chunk_config.separator ? `"${viewingChunkConfig.chunk_config.separator}"` : t('knowledgebase.notSet')}</div>
                    <div>{t('knowledgebase.chunkSize')}: {viewingChunkConfig.chunk_config.chunk_size ?? t('knowledgebase.notSet')}</div>
                    <div>{t('knowledgebase.chunkOverlap')}: {viewingChunkConfig.chunk_config.chunk_overlap ?? t('knowledgebase.notSet')}</div>
                  </div>
                </div>
              )}

              {/* 其他类型配置（structure/token） */}
              {(viewingChunkConfig.chunk_config.parser_type === 'structure' || viewingChunkConfig.chunk_config.parser_type === 'token') && (
                <div className="space-y-2">
                  <Label className="text-xs font-semibold">{t('knowledgebase.chunkConfig')}</Label>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div>{t('knowledgebase.separator')}: {viewingChunkConfig.chunk_config.separator ? `"${viewingChunkConfig.chunk_config.separator}"` : t('knowledgebase.notSet')}</div>
                    <div>{t('knowledgebase.chunkSize')}: {viewingChunkConfig.chunk_config.chunk_size ?? t('knowledgebase.notSet')}</div>
                    <div>{t('knowledgebase.chunkOverlap')}: {viewingChunkConfig.chunk_config.chunk_overlap ?? t('knowledgebase.notSet')}</div>
                  </div>
                </div>
              )}

              {/* 原始 JSON（可选，用于调试） */}
              <div className="space-y-2 border-t pt-2">
                <Label className="text-xs font-semibold">{t('knowledgebase.fullConfigJson')}</Label>
                <pre className="text-xs bg-muted p-2 rounded overflow-x-auto">
                  {JSON.stringify(viewingChunkConfig.chunk_config, null, 2)}
                </pre>
              </div>
            </div>
          )}
          <div className="flex justify-end">
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7"
              onClick={() => {
                setChunkConfigDialogOpen(false);
                setViewingChunkConfig(null);
              }}
            >
              {t('common.close')}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* 重新解析切片配置对话框 */}
      <Dialog open={reprocessChunkConfigDialogOpen} onOpenChange={(open) => {
        setReprocessChunkConfigDialogOpen(open);
        if (!open) {
          setReprocessChunkConfig(null);
          setPendingReprocessFileId(null);
          setPendingReprocessFileIds([]);
        }
      }}>
        <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-sm">
              Reprocess files
            </DialogTitle>
            <DialogDescription className="text-xs">
              Set chunk config
            </DialogDescription>
          </DialogHeader>

          {/* 待处理文件列表 */}
          <div className="border rounded-lg p-3 mb-2 max-h-32 overflow-y-auto bg-muted/30">
            <p className="text-xs text-muted-foreground mb-2 font-medium">{t('knowledgebase.pendingFilesLabel')}</p>
            {isBatchReprocess ? (
              <div className="space-y-1">
                {pendingReprocessFileIds.map((fileId, index) => {
                  const file = kbfiles.find(f => f.id === fileId);
                  return (
                    <div key={fileId} className="text-xs py-0.5 border-b last:border-b-0 text-muted-foreground">
                      {index + 1}. {file?.file_name || fileId}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">
                {kbfiles.find(f => f.id === pendingReprocessFileId)?.file_name || pendingReprocessFileId}
              </div>
            )}
          </div>

          {reprocessChunkConfig && (
            <div className="space-y-4">
              {/* 切片类型 */}
              <div className="flex gap-3 items-center flex-wrap">
                <Label htmlFor="reprocess-parserType" className="w-[120px] text-xs shrink-0">
                  {t('knowledgebase.parserType')}
                  <span className="text-destructive">*</span>
                </Label>
                <Select
                  value={reprocessChunkConfig.parser_type || 'structure'}
                  onValueChange={(value) => {
                    setReprocessChunkConfig((prev) => {
                      if (!prev) return null;
                      const newConfig: any = {
                        ...prev,
                        parser_type: value,
                      };
                      
                      // 根据新的 parser_type 初始化相应的配置
                      if (value === 'table') {
                        newConfig.table_config = prev.table_config || {
                          concat_rows: false,
                          row_joiner: '\n',
                          header_index_max: 0,
                          format_sheet_data_to_json: false,
                        };
                        // 保留chunk_size，设置默认值
                        newConfig.chunk_size = prev.chunk_size || '1000';
                        // 清除其他类型的配置
                        delete newConfig.chunk_overlap;
                        delete newConfig.separator;
                      } else if (value === 'paragraph') {
                        newConfig.separator = prev.separator || '\n\n';
                        newConfig.chunk_size = prev.chunk_size || '1000';
                        newConfig.chunk_overlap = prev.chunk_overlap || '50';
                        // 清除 table_config
                        delete newConfig.table_config;
                      } else {
                        newConfig.separator = prev.separator || '\n\n';
                        newConfig.chunk_size = prev.chunk_size || '1000';
                        newConfig.chunk_overlap = prev.chunk_overlap || '50';
                        // 清除 table_config
                        delete newConfig.table_config;
                      }
                      
                      return newConfig;
                    });
                  }}
                >
                  <SelectTrigger className="w-[200px] h-6 text-xs">
                    <SelectValue placeholder={t('knowledgebase.selectParserType')} />
                  </SelectTrigger>
                  <SelectContent className="text-xs">
                    <SelectGroup>
                      <SelectItem value="structure" className="text-xs h-5">
                        {t('knowledgebase.structure')}
                      </SelectItem>
                      <SelectItem value="token" className="text-xs h-5">
                        {t('knowledgebase.token')}
                      </SelectItem>
                      <SelectItem value="table" className="text-xs h-5">
                        {t('knowledgebase.table')}
                      </SelectItem>
                      <SelectItem value="paragraph" className="text-xs h-5">
                        {t('knowledgebase.paragraph')}
                      </SelectItem>
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground shrink-0">{t('knowledgebase.selectChunkMode')}</p>
              </div>

              {/* Table Config - 只在 parser_type === 'table' 时显示 */}
              {reprocessChunkConfig.parser_type === 'table' && (
                <div className="space-y-3">
                  <div className="flex gap-3 items-center flex-wrap">
                    <div className="flex gap-3 items-center min-w-[280px]">
                      <Label htmlFor="reprocess-table-header-index-max" className="w-[120px] text-xs shrink-0">
                        {t('knowledgebase.maxHeaderIndex')}
                      </Label>
                      <Input
                        type="number"
                        className="w-[200px] h-6 text-xs"
                        id="reprocess-table-header-index-max"
                        value={reprocessChunkConfig.table_config?.header_index_max ?? 0}
                        onChange={(e) =>
                          setReprocessChunkConfig((prev) => {
                            if (!prev) return null;
                            return {
                              ...prev,
                              table_config: {
                                ...prev.table_config,
                                header_index_max: e.target.value ? parseInt(e.target.value) : 0,
                              },
                            };
                          })
                        }
                        min="0"
                      />
                    </div>
                    <div className="flex gap-3 items-center min-w-[200px]">
                      <Label htmlFor="reprocess-table-format-json" className="w-[120px] text-xs shrink-0">
                        {t('knowledgebase.formatAsJson')}
                      </Label>
                      <Checkbox
                        id="reprocess-table-format-json"
                        checked={reprocessChunkConfig.table_config?.format_sheet_data_to_json ?? false}
                        onCheckedChange={(checked) =>
                          setReprocessChunkConfig((prev) => {
                            if (!prev) return null;
                            return {
                              ...prev,
                              table_config: {
                                ...prev.table_config,
                                format_sheet_data_to_json: checked === true,
                              },
                            };
                          })
                        }
                      />
                    </div>
                  </div>
                  <div className="flex gap-3 items-center flex-wrap">
                    <div className="flex gap-3 items-center min-w-[200px]">
                      <Label htmlFor="reprocess-table-concat-rows" className="w-[120px] text-xs shrink-0">
                        {t('knowledgebase.mergeRows')}
                      </Label>
                      <Checkbox
                        id="reprocess-table-concat-rows"
                        checked={reprocessChunkConfig.table_config?.concat_rows ?? false}
                        onCheckedChange={(checked) =>
                          setReprocessChunkConfig((prev) => {
                            if (!prev) return null;
                            return {
                              ...prev,
                              table_config: {
                                ...prev.table_config,
                                concat_rows: checked === true,
                              },
                            };
                          })
                        }
                      />
                    </div>
                    <div className="flex gap-3 items-center min-w-[280px]">
                      <Label htmlFor="reprocess-table-row-joiner" className="w-[120px] text-xs shrink-0">
                        {t('knowledgebase.rowJoiner')}
                      </Label>
                      <Input
                        type="text"
                        className="w-[200px] h-6 text-xs"
                        id="reprocess-table-row-joiner"
                        value={reprocessChunkConfig.table_config?.row_joiner || '\n'}
                        onChange={(e) =>
                          setReprocessChunkConfig((prev) => {
                            if (!prev) return null;
                            return {
                              ...prev,
                              table_config: {
                                ...prev.table_config,
                                row_joiner: e.target.value,
                              },
                            };
                          })
                        }
                      />
                    </div>
                  </div>
                  <div className="flex gap-3 items-center flex-wrap">
                    <div className="flex gap-3 items-center min-w-[320px]">
                      <Label htmlFor="reprocess-table-chunkSize" className="w-[120px] text-xs shrink-0">
                        {t('knowledgebase.chunkSize')}
                        <span className="text-destructive">*</span>
                      </Label>
                      <Input
                        type="text"
                        inputMode="numeric"
                        className="w-[200px] h-6 text-xs"
                        id="reprocess-table-chunkSize"
                        value={reprocessChunkConfig.chunk_size ?? ''}
                        placeholder="1000"
                        onChange={(e) => {
                          const value = e.target.value;
                          // 只允许数字和空字符串
                          if (value === '' || /^\d+$/.test(value)) {
                            setReprocessChunkConfig((prev) => {
                              if (!prev) return null;
                              return {
                                ...prev,
                                chunk_size: value,
                              };
                            });
                          }
                        }}
                        required
                      />
                      <p className="text-xs text-muted-foreground shrink-0">{t('knowledgebase.recommendedValue', { value: '1000' })}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Paragraph Config - 只在 parser_type === 'paragraph' 时显示 */}
              {reprocessChunkConfig.parser_type === 'paragraph' && (
                <div className="space-y-3">
                  <div className="flex gap-3 items-center flex-wrap">
                    <Label htmlFor="reprocess-paragraph-separator" className="w-[120px] text-xs shrink-0">
                      {t('knowledgebase.separator')}
                      <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      type="text"
                      className="w-[200px] h-6 text-xs"
                      id="reprocess-paragraph-separator"
                      value={reprocessChunkConfig.separator || '\n\n'}
                      onChange={(e) =>
                        setReprocessChunkConfig((prev) => {
                          if (!prev) return null;
                          return {
                            ...prev,
                            separator: e.target.value,
                          };
                        })
                      }
                    />
                  </div>
                  <div className="flex gap-3 items-center flex-wrap">
                    <div className="flex gap-3 items-center min-w-[320px]">
                      <Label htmlFor="reprocess-chunkSize" className="w-[120px] text-xs shrink-0">
                        {t('knowledgebase.chunkSize')}
                        <span className="text-destructive">*</span>
                      </Label>
                      <Input
                        type="text"
                        inputMode="numeric"
                        className="w-[200px] h-6 text-xs"
                        id="reprocess-chunkSize"
                        value={reprocessChunkConfig.chunk_size ?? ''}
                        placeholder="1000"
                        onChange={(e) => {
                          const value = e.target.value;
                          // 只允许数字和空字符串
                          if (value === '' || /^\d+$/.test(value)) {
                            setReprocessChunkConfig((prev) => {
                              if (!prev) return null;
                              return {
                                ...prev,
                                chunk_size: value,
                              };
                            });
                          }
                        }}
                        required
                      />
                      <p className="text-xs text-muted-foreground shrink-0">{t('knowledgebase.recommendedValue', { value: '1000' })}</p>
                    </div>
                    <div className="flex gap-3 items-center min-w-[320px]">
                      <Label htmlFor="reprocess-chunkOverlap" className="w-[120px] text-xs shrink-0">
                        {t('knowledgebase.chunkOverlap')}
                        <span className="text-destructive">*</span>
                      </Label>
                      <Input
                        type="text"
                        inputMode="numeric"
                        className="w-[200px] h-6 text-xs"
                        id="reprocess-chunkOverlap"
                        value={reprocessChunkConfig.chunk_overlap ?? ''}
                        placeholder="50"
                        onChange={(e) => {
                          const value = e.target.value;
                          // 只允许数字和空字符串
                          if (value === '' || /^\d+$/.test(value)) {
                            setReprocessChunkConfig((prev) => {
                              if (!prev) return null;
                              return {
                                ...prev,
                                chunk_overlap: value,
                              };
                            });
                          }
                        }}
                      />
                      <p className="text-xs text-muted-foreground shrink-0">{t('knowledgebase.recommendedValue', { value: '50' })}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Default Config - 只在 parser_type 为 'structure' 或 'token' 时显示 */}
              {(reprocessChunkConfig.parser_type === 'structure' || reprocessChunkConfig.parser_type === 'token') && (
                <div className="flex gap-3 items-center flex-wrap">
                  <div className="flex gap-3 items-center min-w-[320px]">
                    <Label htmlFor="reprocess-chunkSize-default" className="w-[120px] text-xs shrink-0">
                      {t('knowledgebase.chunkSize')}
                      <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      type="text"
                      inputMode="numeric"
                      className="w-[200px] h-6 text-xs"
                      id="reprocess-chunkSize-default"
                      value={reprocessChunkConfig.chunk_size ?? ''}
                      placeholder="1000"
                      onChange={(e) => {
                        const value = e.target.value;
                        // 只允许数字和空字符串
                        if (value === '' || /^\d+$/.test(value)) {
                          setReprocessChunkConfig((prev) => {
                            if (!prev) return null;
                            return {
                              ...prev,
                              chunk_size: value,
                            };
                          });
                        }
                      }}
                      required
                    />
                    <p className="text-xs text-muted-foreground shrink-0">{t('knowledgebase.recommendedValue', { value: '1000' })}</p>
                  </div>
                  <div className="flex gap-3 items-center min-w-[320px]">
                    <Label htmlFor="reprocess-chunkOverlap-default" className="w-[120px] text-xs shrink-0">
                      {t('knowledgebase.chunkOverlap')}
                      <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      type="text"
                      inputMode="numeric"
                      className="w-[200px] h-6 text-xs"
                      id="reprocess-chunkOverlap-default"
                      value={reprocessChunkConfig.chunk_overlap ?? ''}
                      placeholder="50"
                      onChange={(e) => {
                        const value = e.target.value;
                        // 只允许数字和空字符串
                        if (value === '' || /^\d+$/.test(value)) {
                          setReprocessChunkConfig((prev) => {
                            if (!prev) return null;
                            return {
                              ...prev,
                              chunk_overlap: value,
                            };
                          });
                        }
                      }}
                    />
                    <p className="text-xs text-muted-foreground shrink-0">{t('knowledgebase.recommendedValue', { value: '50' })}</p>
                  </div>
                </div>
              )}

              {/* 图片理解模型 */}
              <div className="flex gap-3 items-center flex-wrap">
                <Label htmlFor="reprocess-image-caption-model" className="w-[120px] text-xs shrink-0">
                  {t('knowledgebase.imageCaptionModelLabel')}
                </Label>
                <Select
                  value={reprocessChunkConfig.image_caption_model || 'DISABLED'}
                  onValueChange={(value) => {
                    const selectedModel = visionModels.find(m => m.model_id === value);
                    setReprocessChunkConfig((prev) => {
                      if (!prev) return null;
                      return {
                        ...prev,
                        image_caption_model: value !== "DISABLED" ? value : undefined,
                        image_caption_provider_name: selectedModel?.provider_name || prev.image_caption_provider_name,
                      };
                    });
                  }}
                >
                  <SelectTrigger className="w-[200px] h-6 text-xs">
                    <SelectValue placeholder={t('knowledgebase.selectImageModel')} />
                  </SelectTrigger>
                  <SelectContent className="text-xs">
                    <SelectGroup>
                      <SelectItem value="DISABLED" className="text-xs h-5">
                        {t('knowledgebase.disableImageModel')}
                      </SelectItem>
                      {visionModels.map((model) => (
                        <SelectItem key={model.id} value={model.model_id} className="text-xs h-5">
                          {model.model_id} ({model.model})
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground shrink-0">{t('knowledgebase.imageModelHint')}</p>
              </div>
            </div>
          )}
          <div className="flex justify-end gap-2 pt-4">
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7"
              onClick={() => {
                setReprocessChunkConfigDialogOpen(false);
                setReprocessChunkConfig(null);
                setPendingReprocessFileId(null);
                setPendingReprocessFileIds([]);
              }}
            >
              {t('common.cancel')}
            </Button>
            <Button
              variant="default"
              size="sm"
              className="text-xs h-7"
              onClick={() => {
                if (isBatchReprocess) {
                  confirmBatchReprocessFiles();
                } else {
                  confirmReprocessFile();
                }
              }}
              disabled={reprocessing}
            >
              {reprocessing ? t('knowledgebase.processing') : t('knowledgebase.confirmReparse')}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
