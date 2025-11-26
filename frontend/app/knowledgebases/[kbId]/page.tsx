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

interface MetadataCondition {
  name: string;
  comparison_operator: string;
  value: string | number;
}

export default function KnowledgeBaseDetailPage(
  { params } : { params: Promise<{ kbId: string }> }
) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [knowledgebase, setKnowledgeBase] = useState<KbConfig>(); // 知识库列表
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
  const [searchrecords, setSearchRecords] = useState(Array<SearchRecord>); // 搜索结果
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null); // 搜索错误信息
  const [expandedCards, setExpandedCards] = useState<Record<number, boolean>>({}); // 展开的卡片索引
  const [logicalOperator, setLogicalOperator] = useState<string>('and');
  const [metadataConditions, setMetadataConditions] = useState<
    MetadataCondition[]
  >([]);
  const [fileSource, setFileSource] = useState('');
  const [fileSourceOpen, setFileSourceOpen] = useState(false);
  const [currentFileId, setCurrentFileId] = useState<string>('');
  const { kbId } = use(params);

  let isRefreshing = false;
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
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
    vector_weight?: number;
    enable_rerank?: boolean;
    rerank_model?: string;
    top_k?: number;
    similarity_threshold?: number;
    rerank_top_k?: number;
  }>({});
  const [rerankerModels, setRerankerModels] = useState<Array<{id: string; model_id: string; model_name: string}>>([]);
  const [retrievalSettingOpen, setRetrievalSettingOpen] = useState(true);

  const default_comparator = [
    'contains',
    'not contains',
    'start with',
    'end with',
    'is',
    'is not',
    'empty',
    'not empty',
    '=',
    '≠',
    '>',
    '<',
    '≥',
    '≤',
    'before',
    'after',
  ];
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
      const search_result = await fetch(`/api/retrieval`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: kbquery,
          user_id: user,
          knowledge_id: kbId,
          retrieval_setting: retrievalSetting,
          metadata_condition: {
            conditions: metadataConditions,
            logical_operator: logicalOperator,
          },
        }),
      });

      const search_json = await search_result.json();
      console.log('搜索知识库结果:', search_json);

      // 检查响应中的 status_code 或 code 字段
      const statusCode = search_json.status_code || search_json.code;
      if (statusCode && statusCode !== 200) {
        const errorMessage = search_json.message || search_json.error || '搜索失败';
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

      // 成功情况
      setSearchRecords(search_json.records || []);
      setSearchError(null);
    } catch (err: any) {
      const errorMessage = err.message || '搜索知识库失败';
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
      const files_res = await fetch(url, { signal: controller.signal, });
      if (!files_res.ok) throw new Error('获取知识库文件列表失败');

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
      const metaRes = await fetch(`/api/config/knowledgebases/${kbId}/metadata`);
      if (!metaRes.ok) throw new Error('获取知识库元数据失败');
      const metadata_json = await metaRes.json();
      const metadata_data = metadata_json.data as MetadataConfig[];
      const valueTypes = Object.fromEntries(
        metadata_data.map((metadata) => [metadata.name, metadata.value_type]),
      ) as { [key: string]: string };

      setMetadataValueTypes({ ...valueTypes, '': 'string' });
      setMetadataConfigs(metadata_data);
    } catch (err: any) {
      toast.error(err.message);
    }
  }, [kbId]);

  const fetchKbConfigs = useCallback(async () => {
    try {
      const [kbRes, metaRes, rerankerRes] = await Promise.all([
        fetch(`/api/config/knowledgebases/${kbId}`),
        fetch(`/api/config/knowledgebases/${kbId}/metadata`),
        fetch(`/api/config/rerankers`),
      ]);

      if (!kbRes.ok) throw new Error('获取知识库配置失败');
      const json_data = await kbRes.json();
      const kb_data = json_data.data;

      setKnowledgeBase(kb_data); // 更新状态
      console.log('知识库详情数据:', kb_data);

      // 初始化检索设置，从 knowledgebase.retrieval_config 获取默认值
      if (kb_data?.retrieval_config) {
        setRetrievalSetting({
          retrieval_mode: kb_data.retrieval_config.retrieval_mode || 'hybrid',
          vector_weight: kb_data.retrieval_config.vector_weight ?? 0.5,
          enable_rerank: kb_data.retrieval_config.enable_rerank ?? false,
          rerank_model: kb_data.retrieval_config.rerank_model || '',
          top_k: kb_data.retrieval_config.top_k ?? 5,
          similarity_threshold: kb_data.retrieval_config.similarity_threshold ?? 0.2,
          rerank_top_k: kb_data.retrieval_config.rerank_top_k ?? 5,
        });
      }

      if (!metaRes.ok) throw new Error('获取知识库元数据失败');
      const metadata_json = await metaRes.json();
      const metadata_data = metadata_json.data as MetadataConfig[];
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
    } catch (err: any) {
      toast.error(err.message);
    }
  }, [kbId]);

  useEffect(() => {
    fetchKbConfigs();
  }, [fetchKbConfigs]);

  if (!knowledgebase) {
    return <div className="p-6">加载中...</div>;
  }

  const handleSaveSuccess = async (kb: KbConfig) => {
    toast.success("知识库配置保存成功");
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

      // 调用更新知识库接口，只更新retrieval_config
      const res = await fetch(`/api/config/knowledgebases/${kbId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          retrieval_config: retrieval_config,
        }),
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

      toast.success("检索设置已保存至知识库配置");
    } catch (err: any) {
      console.error('保存检索设置失败:', err);
      toast.error(err.message || '保存检索设置失败');
    }
  };


  const handleReprocessFile = async (file_id: string) => {
    try {
      const res = await fetch(
        `/api/config/knowledgebases/${kbId}/files/${file_id}`,
        {
          method: 'PUT',
        },
      );
      if (!res.ok) throw new Error(`重新解析 ${file_id} 失败`);
      toast.success("文件入队成功。");
    } catch (error: any) {
      toast.error(error.message);
    } finally {
      fetchKbFiles();
    }
  };


  const handleDeleteFile = async (file_id: string) => {
    setDeleting(true);
    try {
      const res = await fetch(
        `/api/config/knowledgebases/${kbId}/files/${file_id}`,
        {
          method: 'DELETE',
        },
      );
      if (!res.ok) throw new Error(`删除 ${file_id} 失败`);
      toast.success("文件删除成功。");
    } catch (error: any) {
      toast.error(error.message);
    } finally {
      setDeleting(false);
      fetchKbFiles();
    }
  };

  const handleBatchDeleteFiles = async () => {
    if (selectedFiles.size === 0) {
      toast.error("请至少选择一个文件");
      setShowBatchDeleteDialog(false);
      return;
    }

    setShowBatchDeleteDialog(false);
    setDeleting(true);
    try {
      const res = await fetch(
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
      toast.success(result.message || `成功删除 ${selectedFiles.size} 个文件`);
      setSelectedFiles(new Set()); // 清空选择
    } catch (error: any) {
      toast.error(error.message || "批量删除失败");
    } finally {
      setDeleting(false);
      fetchKbFiles();
    }
  };

  const handleBatchReprocessFiles = async () => {
    if (selectedFiles.size === 0) {
      toast.error("请至少选择一个文件");
      setShowBatchReprocessDialog(false);
      return;
    }

    setShowBatchReprocessDialog(false);
    setReprocessing(true);
    try {
      const res = await fetch(
        `/api/config/knowledgebases/${kbId}/files/batch`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            operation: 'reprocess',
            file_id_list: Array.from(selectedFiles),
          }),
        },
      );
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.message || `批量重新解析失败`);
      }
      const result = await res.json();
      toast.success(result.message || `成功将 ${selectedFiles.size} 个文件加入重新处理队列`);
      setSelectedFiles(new Set()); // 清空选择
    } catch (error: any) {
      toast.error(error.message || "批量重新解析失败");
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
      const res = await fetch(
        `/api/config/knowledgebases/${kbId}/files/${fileId}`,
      );
      if (!res.ok) throw new Error('获取知识库文件失败');
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
      setPreviewError(err?.message || '加载失败');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSaveFileSource = async () => {
    if (!currentFileId) return;
    
    try {
      const res = await fetch(
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
      if (!res.ok) throw new Error('源链接保存失败');

      const fileObj = kbfiles.filter((file) => file.id === currentFileId)[0];
      if (fileObj) {
        fileObj.file_source = fileSource;
      }
      setFileSourceOpen(false);
      setCurrentFileId('');
      toast.success("源链接保存成功");
    } catch (error: any) {
      toast.error(error.message);
    }
  };

  const selectMetadataKey = async (metadata_key: string) => {
    setMetadataEditError('');
    const emptyKeys = Object.keys(editingMetadata).filter(
      (key) => editingMetadata[key] === '',
    );
    if (emptyKeys.length > 1) throw new Error('有多于一个新建项。');
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
      const file_res = await fetch(
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
        '没有可用的自定义的元数据配置，你可以先去知识库设置页面添加。',
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
      const roleRes = await fetch(`/api/config/roles?size=100`);
      if (!roleRes.ok) {
        alert('查询角色失败');
        return;
      }
      const all_roles = (await roleRes.json()).data.items;
      setRoles(all_roles);

      const permission_name = file_id;
      const res = await fetch(
        `/api/config/roles/permissions?name=${permission_name}&size=100`,
      );
      if (!res.ok) {
        alert('查询文件permission失败');
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
      const roleRes = await fetch(
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
        alert('更新文件角色失败');
        return;
      }
      console.log('更新文件角色成功：', await roleRes.json());
      setRoleDialogOpen(false);
      setEditRoleFileId('');
      toast.success("权限设置保存成功");
    } catch (error: any) {
      toast.error(error.message);
    }
  };

  const handleFileUpload = async (files: FileList | null) => {
    console.log('##handleFileUpload', files);
    if (!files) {
      alert('文件列表为空！');
      return;
    }
    setUploadDialogOpen(false); // 关闭Dialog
    setUploading(true);

    // 文件校验 (Demo功能，后续调整优化)
    const validFiles = Array.from(files).filter((file) => {
      // const isValidType = ['application/pdf', 'application/msword'].includes(file.type);
      const isValidSize = file.size <= 1000 * 1024 * 1024;
      // return isValidType && isValidSize;
      return isValidSize;
    });

    if (validFiles.length === 0) {
      alert("请选择有效的文件（如 PDF 或 Word，且小于 1GB）");
      setUploading(false);
      return;
    }

    // 上传文件
    const formData = new FormData();
    validFiles.forEach((file) => {
      formData.append('files', file);
    });

    try {
      // 生产环境上传大文件直连
      const API_PREFIX = process.env.NEXT_PUBLIC_DEVELOP_MODE === "true" ? "/api" : "/v1"; // 你的后端地址
      console.log("上传后端地址前缀: ", API_PREFIX)
      const res = await fetch(
        `${API_PREFIX}/config/knowledgebases/${kbId}/files`,
        {
          method: 'POST',
          body: formData,
        },
      );
      const upload_result = await res.json();
      if (upload_result.code !== 200) {
        throw new Error(upload_result.message);
      }
      console.log('上传成功:', upload_result);
      toast.success("上传成功。")
    } catch (error: any) {
      console.error('上传失败:', error.message);
      toast.error("上传失败: " + error.message);
    } finally {
      setUploading(false);
      // 清空文件选择框
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // 清空 input 的值
      }
      setPage(1);
      fetchKbFiles();
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
      return { valid: false, error: `元数据 '${name}' 的值不能为空` };
    }

    if (valueType === 'string') {
      return { valid: true, convertedValue: String(value) };
    } else if (valueType === 'number') {
      const numValue = typeof value === 'number' ? value : parseFloat(String(value));
      if (isNaN(numValue)) {
        return { valid: false, error: `元数据 '${name}' 的值 '${value}' 不是有效的数字` };
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
            return { valid: false, error: `元数据 '${name}' 的值 '${value}' 不是有效的时间戳或日期` };
          }
        }
      } else if (value instanceof Date) {
        timestamp = value.getTime(); // 转换为秒级时间戳
      } else {
        return { valid: false, error: `元数据 '${name}' 的值类型不正确` };
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
      setMetadataEditError('无法保存空的元数据名称。');
      return;
    }

    try {
      // 验证所有metadata值是否符合类型要求
      const metadata_entries = [];
      for (const name of Object.keys(editingMetadata).filter((name) => !default_metadata_keys.includes(name))) {
        const valueType = metadataValueTypes[name] || 'string';
        const validation = validateMetadataValue(editingMetadata[name], valueType, name);
        if (!validation.valid) {
          setMetadataEditError(validation.error || '元数据值验证失败');
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
      const res = await fetch(
        `/api/config/knowledgebases/${kbId}/files/${currentMetadataFileId}/metadata`,
        {
          method: 'POST',
          body: JSON.stringify(bodyData),
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );
      if (!res.ok) throw Error('保存metadata失败');
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
      toast.success("元数据保存成功");
    } catch (error: any) {
      console.log('保存metadata失败', error);
      toast.error(error.message || '保存metadata失败');
    } finally {
      setMetadataEditError('');
    }
  };

  const handleAddMetadataConfig = async () => {
    if (!newMetadataName) {
      setMetadataError('必须填入元数据名称。');
      return;
    }

    if (metadataConfigs.some((config) => config.name === newMetadataName)) {
      setMetadataError(`元数据名称 '${newMetadataName}' 已经存在。`);
      return;
    }

    const metadata_url = `/api/config/knowledgebases/${kbId}/metadata`;
    try {
      const res = await fetch(metadata_url, {
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
      toast.success('添加元数据成功');
    } catch (err: any) {
      console.log('保存知识库失败', err.message);
      setMetadataError(err.message);
    }
  };

  const handleRemoveMetadataEntry = async (id: string) => {
    const metadata_url = `/api/config/knowledgebases/${kbId}/metadata/${id}`;
    try {
      const res = await fetch(metadata_url, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`删除metadata失败: ${await res.text()}`);

      // 重新获取metadata列表以获取最新的count信息
      await fetchMetadataConfigs();
      toast.success('删除元数据成功');
    } catch (err: any) {
      console.log('删除元数据失败。', err.message);
      toast.error(err.message || '删除元数据失败');
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
      setMetadataError('必须填入元数据名称。');
      return;
    }

    if (metadataConfigs.some((config) => config.name === newMetadataName && config.id !== editingMetadataConfig.id)) {
      setMetadataError(`元数据名称 '${newMetadataName}' 已经存在。`);
      return;
    }

    const metadata_url = `/api/config/knowledgebases/${kbId}/metadata/${editingMetadataConfig.id}`;
    try {
      const res = await fetch(metadata_url, {
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
      toast.success('更新元数据成功');
    } catch (err: any) {
      console.log('更新元数据失败', err.message);
      setMetadataError(err.message);
    }
  };

  const addCondition = () => {
    const newCondition = {
      name: '',
      comparison_operator: '',
      value: '',
    };
    setMetadataConditions([...metadataConditions, newCondition]);
  };

  const deleteCondition = (i: number) => {
    const newConditionArray = metadataConditions.filter((v, idx) => idx !== i);
    setMetadataConditions(newConditionArray);
  };

  const setConditionName = (i: number, name: string) => {
    const newConditions = metadataConditions.map((condition, idx) => {
      if (idx === i) {
        if (metadataValueTypes[name] === 'datetime') {
          return {
            name: name,
            value: new Date().getTime(),
            comparison_operator: condition.comparison_operator,
          };
        }
        return {
          name: name,
          value: condition.value,
          comparison_operator: condition.comparison_operator,
        };
      }
      return condition;
    });
    setMetadataConditions(newConditions);
  };

  const setConditionValue = (i: number, value: string | number) => {
    const newConditions = metadataConditions.map((condition, idx) => {
      if (idx === i) {
        return {
          name: condition.name,
          value: value,
          comparison_operator: condition.comparison_operator,
        };
      }
      return condition;
    });
    setMetadataConditions(newConditions);
  };

  const setConditionOp = (i: number, op: string) => {
    const newConditions = metadataConditions.map((condition, idx) => {
      if (idx === i) {
        return {
          name: condition.name,
          value: condition.value,
          comparison_operator: op,
        };
      }
      return condition;
    });
    setMetadataConditions(newConditions);
  };

  return (
    <div className="flex flex-col h-screen pt-0 space-y-0">
      <div className="absolute top-2 left-12 py-0 flex items-center z-10">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button
                  variant="link"
                  className="px-0"
                  onClick={() => router.push('/knowledgebases')}
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
        <div className="flex gap-2 items-center ml-4">
          <Badge variant="secondary" className="text-xs bg-muted text-muted-foreground">
            ID: {knowledgebase.id}
          </Badge>
          {knowledgebase.description && (
            <Badge variant="secondary" className="text-xs bg-muted text-muted-foreground max-w-[200px] truncate">
              {knowledgebase.description}
            </Badge>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-6">
        <Tabs defaultValue="details">
          <TabsList className="py-0 bg-muted rounded-lg flex-none">
            <TabsTrigger value="details" className="py-1 px-2">
              <span className="text-xs">文件管理</span>
            </TabsTrigger>
            <TabsTrigger value="settings" className="py-1 px-2">
              <span className="text-xs">知识库设置</span>
            </TabsTrigger>
            <TabsTrigger value="retrieval_test" className="py-1 px-2">
              <span className="text-xs">检索测试</span>
            </TabsTrigger>
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
                    placeholder="搜索..."
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
                            处理中...
                          </>
                        ) : (
                          <>
                            <CirclePlayIcon className="mr-2 h-4 w-4" />
                            批量重新解析 ({selectedFiles.size})
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
                            删除中...
                          </>
                        ) : (
                          <>
                            <Trash2Icon className="mr-2 h-4 w-4" />
                            批量删除 ({selectedFiles.size})
                          </>
                        )}
                      </Button>
                      <AlertDialog open={showBatchReprocessDialog} onOpenChange={setShowBatchReprocessDialog}>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>确认批量重新解析？</AlertDialogTitle>
                            <AlertDialogDescription>
                              您即将重新解析 {selectedFiles.size} 个文件，这些文件将被重新处理并更新。请确认是否继续？
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>取消</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={handleBatchReprocessFiles}
                            >
                              确认重新解析
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                      <AlertDialog open={showBatchDeleteDialog} onOpenChange={setShowBatchDeleteDialog}>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>确认批量删除？</AlertDialogTitle>
                            <AlertDialogDescription>
                              您即将删除 {selectedFiles.size} 个文件，此操作无法撤销。请仔细核对之后再确认。
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>取消</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={handleBatchDeleteFiles}
                              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            >
                              确认删除
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </>
                  )}
                </div>
                <div className="flex gap-2 items-center">
                <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
                    <DialogTrigger asChild>
                      <Button
                        variant="default"
                        className="h-6 text-xs"
                        disabled={uploading}
                        onClick={() => setUploadDialogOpen(true)}
                      >
                        <Upload className="h-3 w-3" /> 上传文件
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle>上传文件</DialogTitle>
                      </DialogHeader>
                      <div 
                        className="flex flex-col items-center justify-center py-8 px-4 cursor-pointer border-2 border-dashed rounded-lg hover:bg-muted/50 transition-colors"
                        onClick={() => {
                          document.getElementById('file-upload')?.click();
                        }}
                      >
                        <Upload className="h-12 w-12 text-muted-foreground mb-4" />
                        <p className="text-sm text-muted-foreground text-center">
                          支持的文件类型：txt, md, pdf, docx, pptx, xlsx, xls, html, jsonl, jpg, jpeg, png
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          点击选择文件
                        </p>
                      </div>
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
                      toast.success("刷新成功");
                    }}
                  > 
                    <RefreshCcwIcon className="h-3 w-3"/> 刷新
                  </Button>
                  <Button
                    variant="outline"
                    className="h-6 text-xs"
                    onClick={() => setMetadataConfigDialogOpen(true)}
                  > 
                    <Database className="h-3 w-3"/> 元数据
                  </Button>
                </div>
              </div>
              <div>
                    <Table>
                      <TableHeader>
                        <TableRow className='border-border/30 border-y'>
                          <TableHead className="w-12">
                            <Checkbox
                              checked={isAllSelected}
                              onCheckedChange={handleSelectAll}
                            />
                          </TableHead>
                          <TableHead>
                            <div className="flex gap-2 items-center max-w-[400px] text-xs text-muted-foreground">
                               文件名
                            </div>
                          </TableHead>
                          <TableHead className="text-xs text-muted-foreground">文件大小</TableHead>
                          <TableHead className="text-xs text-muted-foreground">更新时间</TableHead>
                          <TableHead>
                            <FileStatusFilter 
                              value={statusFilter as 'all' | 'succeeded' | 'failed' | 'pending' | 'parsing' | 'persisting'}
                              onValueChange={setStatusFilter}
                            />
                          </TableHead>
                          <TableHead className="text-xs text-muted-foreground">操作</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {kbfiles.map((file) => (
                          <TableRow 
                            key={file.id}
                            className="cursor-pointer hover:bg-muted/100 transition-colors h-8 border-border/30"
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
                            title="点击查看切片"
                          >
                            <TableCell className="py-1" onClick={(e) => e.stopPropagation()}>
                              <Checkbox
                                checked={selectedFiles.has(file.id)}
                                onCheckedChange={(checked) =>
                                  handleSelectFile(file.id, checked as boolean)
                                }
                              />
                            </TableCell>
                            <TableCell className="p-1">
                              <span className="truncate block w-full text-left font-medium text-xs">
                                {file.file_name}            
                              </span>
                            </TableCell>
                            <TableCell className="text-xs p-1">
                              {formatFileSize(Number(file.file_size))}
                            </TableCell>
                            <TableCell className="text-xs p-1">
                              {formatBeijingTime(file.updated_at)}
                            </TableCell>
                            <TableCell className="text-xs p-1">
                              {file.status === 'pending' ? (
                                <Badge variant="secondary" className="bg-yellow-100 text-yellow-700 hover:bg-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400">
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  等待解析
                                </Badge>
                              ) : file.status === 'parsing' ? (
                                <Badge variant="secondary" className="bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400">
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  解析中
                                </Badge>
                              ) : file.status === 'persisting' ? (
                                <Badge variant="secondary" className="bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400">
                                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                  索引中
                                </Badge>
                              ) : file.status === 'succeeded' ? (
                                <Badge variant="secondary" className="bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/20 dark:text-green-400">
                                  <CheckCircle className="mr-1 h-3 w-3" />
                                  解析成功
                                </Badge>
                              ) : file.status === 'failed' ? (
                                <HoverCard>
                                  <HoverCardTrigger asChild>
                                    <Badge variant="secondary" className="bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/20 dark:text-red-400 cursor-pointer">
                                      <XCircle className="mr-1 h-3 w-3" />
                                      解析失败
                                    </Badge>
                                  </HoverCardTrigger>
                                  <HoverCardContent className="w-80">
                                    错误原因: {file.failed_reason}
                                  </HoverCardContent>
                                </HoverCard>
                              ) : (
                                <Badge variant="secondary">{file.status}</Badge>
                              )}
                            </TableCell>
                            <TableCell className="gap-1 p-1" onClick={(e) => e.stopPropagation()}>
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
                                      <span className="text-xs font-medium">查看文件</span>
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
                                      <span className="text-xs font-medium">权限设置</span>
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
                                      <span className="text-xs font-medium">元数据</span>
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
                                      <span className="text-xs font-medium">源链接</span>
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
                                      <span className="text-xs font-medium">重新解析</span> 
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
                                      <span className="text-xs font-medium">删除</span> 
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
                        <DialogTitle className="text-sm">设置源链接</DialogTitle>
                        <DialogDescription className="text-xs">
                          {kbfiles.find(f => f.id === currentFileId)?.file_name || ''}
                        </DialogDescription>
                      </DialogHeader>
                      <div className="flex flex-col gap-3 py-2">
                        <div className="flex flex-col gap-2">
                          <Label className="text-xs">源链接</Label>
                          <Input
                            type="text"
                            placeholder="输入文件外部源链接，如语雀、飞书、钉钉文档等。"
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
                          保存
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
                          取消
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
                          <DialogTitle className="text-sm">编辑元数据</DialogTitle>
                        ) : (
                          <DialogTitle className="text-sm">查看元数据</DialogTitle>
                        )}
                        <DialogDescription className="text-xs">
                          {kbfiles.find(f => f.id === currentMetadataFileId)?.file_name || ''}
                        </DialogDescription>
                      </DialogHeader>
                      <div className="grid flex-1 auto-rows-min gap-3 py-2">
                        <div className="text-xs">
                          {isEditingMetadata ? (
                            <Label htmlFor="sheet-custom-meta" className="text-xs pb-3">
                              自定义
                              <Button
                                variant="secondary"
                                className="w-16 h-5 text-xs"
                                onClick={handAddFileMetadata}
                              >
                                <PlusIcon className="h-3 w-3" />
                                添加
                              </Button>
                            </Label>
                          ) : (
                            <Label htmlFor="sheet-custom-meta" className="text-xs">
                              自定义
                            </Label>
                          )}
                          {Object.keys(editingMetadata).filter(
                            (key: string) =>
                              !default_metadata_keys.includes(key),
                          ).length === 0 && (
                            <p className="text-xs text-muted-foreground">
                              当前没有配置自定义元数据，点击编辑添加。
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
                                        <SelectValue placeholder="选择元数据" />
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
                            内置元数据
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
                              编辑
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
                            关闭
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
                        <DialogTitle className="text-sm">文档权限设置</DialogTitle>
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
                                      title="清空所有角色"
                                    >
                                      清空选择<XCircle className="h-3 w-3" />
                                    </Button>
                                  )}
                                </>
                              ) : (
                                <div>
                                  <p className="text-xs text-muted-foreground">
                                    尚未配置角色信息，前往`权限控制`设置。
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
                          取消
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
                        <DialogDescription>文件预览</DialogDescription>
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
                              title="图片预览"
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
                              title="文件预览"
                            />
                          ) : previewFile?.file_extension === '.md' ||
                            previewFile?.file_extension === '.txt' ? (
                            <MarkdownViewer file_url={previewFile?.file_metadata?.file_url || ''} />
                          ) : previewFile?.file_extension === '.jsonl' ? (
                            <JsonlViewer file_url={previewFile?.file_metadata?.file_url || ''} />
                          ) : previewFile?.file_extension === '.html' ? (
                            <HtmlViewer file_url={previewFile?.file_metadata?.file_url || ''} />
                          ) : (
                            <div>
                              暂不支持此格式文件的在线预览，请直接下载查看
                              <a
                                href={previewFile?.file_metadata?.file_url}
                                className="text-blue-500 hover:underline ml-2"
                              >
                                下载文件
                              </a>
                            </div>
                          )}
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                
                { kbfiles.length === 0 && (
                  <p className="text-muted-foreground mx-auto text-xs py-15 text-center bg-gray-50 rounded-lg">暂无文件</p>
                )}
                <PaginationComponent
                  currentPage={page}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              </div>
          </TabsContent>
          <TabsContent value="settings" className="py-2">
            <KbConfigCard
              isCreate={false}
              kbConfig={knowledgebase}
              onSaveSuccess={handleSaveSuccess}
              onCancel={() => {}}
            ></KbConfigCard>
          </TabsContent>
          <TabsContent value="retrieval_test" className="py-2 flex flex-col h-full min-h-0">
            <div className="flex gap-3 flex-1 min-h-0 overflow-hidden">
              {/* 左侧：查询输入和检索设置 */}
              <div className="flex flex-col w-[400px] shrink-0 h-full justify-between overflow-y-auto">
                {/* 检索测试输入区域 - 左上角 */}
                <Card className="flex-[4] flex flex-col min-h-0 mb-2">
                  <CardHeader className="flex-shrink-0">
                    <div className="flex-1 min-w-[200px] relative">
                        <SearchIcon className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
                        <Input
                          type="text"
                          id="search_query"
                          placeholder="请输入查询内容"
                          onChange={handleQueryInputChange}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              handleSearchSubmit();
                            }
                          }}
                          className="w-full text-xs pl-8"
                        />
                      </div>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col min-h-0">
                    {/* 搜索框和按钮 */}
                    <div className="flex flex-col gap-2 flex-1">
                      <div className="flex flex-wrap gap-2 flex-shrink-0">
                        <Popover>
                          <PopoverTrigger asChild>
                            <Button variant="outline" size="sm" className="text-xs h-7">
                              <FilterIcon className="h-3 w-3" />
                              元数据
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-[450px]">
                            <div className="grid gap-3">
                              <div className="space-y-2">
                                <RadioGroup
                                  value={logicalOperator}
                                  onValueChange={(value) => setLogicalOperator(value)}
                                >
                                  <div className="flex items-center space-x-2">
                                    <p className="text-muted-foreground text-xs">
                                      逻辑操作符
                                    </p>

                                    <RadioGroupItem value="and" id="r1" />
                                    <Label htmlFor="r1" className="text-xs">AND</Label>
                                    <RadioGroupItem value="or" id="r2" />
                                    <Label htmlFor="r2" className="text-xs">OR</Label>
                                  </div>
                                </RadioGroup>
                              </div>
                              <div className="grid gap-2">
                                <div className="space-y-2">
                                  {metadataConditions.map((condition, i) => (
                                    <div
                                      className="flex items-center space-x-2"
                                      key={i}
                                    >
                                      <div>
                                        <Select
                                          value={condition.name}
                                          onValueChange={(value) => {
                                            setConditionName(i, value);
                                          }}
                                        >
                                          <SelectTrigger className="h-6 text-xs w-[120px]">
                                            <SelectValue placeholder="名称" />
                                          </SelectTrigger>
                                          <SelectContent className="text-xs">
                                            <SelectGroup>
                                              {metadataConfigs.map((metadata) => (
                                                <SelectItem
                                                  key={metadata.name}
                                                  value={metadata.name}
                                                  className="text-xs h-5"
                                                >
                                                  {metadata.name}
                                                </SelectItem>
                                              ))}
                                            </SelectGroup>
                                          </SelectContent>
                                        </Select>
                                      </div>
                                      <div>
                                        <Select
                                          value={condition.comparison_operator}
                                          onValueChange={(value) => {
                                            setConditionOp(i, value);
                                          }}
                                        >
                                          <SelectTrigger className="h-6 text-xs w-[80px]">
                                            <SelectValue placeholder="规则" />
                                          </SelectTrigger>
                                          <SelectContent className="w-[80px] text-xs">
                                            <SelectGroup>
                                              {default_comparator.map((op) => (
                                                <SelectItem key={op} value={op} className="text-xs h-5">
                                                  {op}
                                                </SelectItem>
                                              ))}
                                            </SelectGroup>
                                          </SelectContent>
                                        </Select>
                                      </div>
                                      <div>
                                        {metadataValueTypes[condition.name] ===
                                        'datetime' ? (
                                            <DatetimeInput
                                              value={
                                                (() => {
                                                  const val = typeof condition.value === 'number'
                                                    ? condition.value
                                                    : parseFloat(condition.value);
                                                  // 将秒级时间戳转换为毫秒级（DatetimeInput 期望毫秒级）
                                                  return isNaN(val) ? new Date().getTime() : val;
                                                })()
                                              }
                                              width="sm"
                                              onValueChange={(value) => {
                                                // 将毫秒级时间戳转换为秒级（后端存储秒级）
                                                setConditionValue(i, value);
                                              }}
                                            />
                                          ) : (
                                            <Input
                                              className="w-32 h-6 text-xs"
                                              value={condition.value.toString()}
                                              onChange={(e) =>
                                                setConditionValue(i, e.target.value)
                                              }
                                            />
                                          )}
                                      </div>
                                      <div>
                                        <Button
                                          variant="outline"
                                          onClick={() => {
                                            deleteCondition(i);
                                          }}
                                          className="w-6 h-6 p-0"
                                          size="sm"
                                        >
                                          <Trash2Icon className="w-3 h-3" />
                                        </Button>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                                <Button
                                  variant="secondary"
                                  onClick={addCondition}
                                  className="h-6 text-xs"
                                  size="sm"
                                >
                                  新增过滤规则
                                </Button>
                              </div>
                            </div>
                          </PopoverContent>
                        </Popover>
                        <Input
                          className="w-30 text-xs h-7"
                          placeholder="输入user_id"
                          value={user}
                          onChange={(e) => {
                            setUser(e.target.value);
                          }}
                        />
                        <Button
                          type="button"
                          onClick={handleSearchSubmit}
                          className="whitespace-nowrap text-xs h-7"
                          size="sm"
                        >
                          <SearchIcon className="h-3 w-3" />
                          开始查询
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* 检索设置板块 - 左下角 */}
                <Card className={`flex-[0.8] overflow-y-auto text-xs min-h-0 p-2`}>
                  <CardHeader className="px-3 pt-2">
                    <div className="flex items-center justify-between h-6">
                      <CardTitle className="text-sm">检索设置</CardTitle>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-5 w-5 p-0"
                        onClick={() => setRetrievalSettingOpen(!retrievalSettingOpen)}
                      >
                        {retrievalSettingOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                      </Button>
                    </div>
                  </CardHeader>
                  {retrievalSettingOpen && (
                    <CardContent className="space-y-4 px-3 pb-2">
                      {/* 检索策略 */}
                      <div className="flex flex-col gap-2">
                        <div className="flex gap-2 items-center">
                          <Label className="w-[80px] text-xs">检索策略</Label>
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
                            className="flex gap-x-2 overflow-visible"
                          >
                            <ToggleGroupItem
                              value="vector"
                              aria-label="向量检索"
                              className="!rounded-full px-1.5 py-0.5 text-xs data-[state=on]:bg-black data-[state=on]:text-white h-6"
                            >
                              <ScanSearch className="w-2 h-2" />
                              向量检索
                            </ToggleGroupItem>
                            <ToggleGroupItem
                              value="fulltext"
                              aria-label="全文检索"
                              className="!rounded-full px-1.5 py-0.5 text-xs data-[state=on]:bg-black data-[state=on]:text-white h-6"
                            >
                              <TextSearch className="w-2 h-2" />
                              全文检索
                            </ToggleGroupItem>
                            <ToggleGroupItem
                              value="hybrid"
                              aria-label="混合检索"
                              className="!rounded-full px-1.5 py-0.5 text-xs data-[state=on]:bg-black data-[state=on]:text-white h-6"
                            >
                              <SearchCode className="w-2 h-2" />
                              混合检索
                            </ToggleGroupItem>
                          </ToggleGroup>
                        </div>
                        {retrievalSetting.retrieval_mode === 'hybrid' && (
                          <div className="flex items-center gap-2 pt-1">
                            <Label htmlFor="vector_weight" className="w-[80px] text-xs">
                              向量权重
                            </Label>
                            <Slider
                              id="vector_weight"
                              className="w-40"
                              min={0}
                              max={1}
                              step={0.1}
                              value={[retrievalSetting.vector_weight ?? 0.5]}
                              onValueChange={(value) =>
                                setRetrievalSetting((prev) => ({
                                  ...prev,
                                  vector_weight: value[0],
                                }))
                              }
                            />
                            <span className="w-10 text-right text-xs font-medium">
                              {retrievalSetting.vector_weight ?? 0.5}
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Top-K 和相似度阈值 */}
                      <div className="flex flex-col gap-2">
                        <div className="flex gap-2 items-center">
                          <Label htmlFor="top_k" className="w-[80px] text-xs">
                            Top-K
                          </Label>
                          <Slider
                            className="w-40"
                            defaultValue={[5]}
                            max={100}
                            min={1}
                            step={1}
                            value={[retrievalSetting.top_k ?? 5]}
                            onValueChange={(value: number[]) => {
                              setRetrievalSetting((prev) => ({
                                ...prev,
                                top_k: value[0],
                              }));
                            }}
                          />
                          <span className="font-medium w-10 text-xs">
                            {retrievalSetting.top_k ?? 5}
                          </span>
                        </div>
                        <div className="flex gap-2 items-center">
                          <Label htmlFor="similarity_threshold" className="w-[80px] text-xs">
                            相似度阈值
                          </Label>
                          <Slider
                            className="w-40"
                            defaultValue={[0.2]}
                            max={1}
                            min={0}
                            step={0.01}
                            value={[retrievalSetting.similarity_threshold ?? 0.2]}
                            onValueChange={(value: number[]) => {
                              setRetrievalSetting((prev) => ({
                                ...prev,
                                similarity_threshold: value[0],
                              }));
                            }}
                          />
                          <span className="font-medium w-10 text-xs">
                            {retrievalSetting.similarity_threshold?.toFixed(2) ?? '0.20'}
                          </span>
                        </div>
                      </div>

                      {/* 开启重排序 */}
                      <div className="flex flex-col gap-2">
                        <div className="flex gap-2 items-center">
                          <Label className="w-[80px] text-xs">开启重排序</Label>
                          <Checkbox
                            id="enable_rerank"
                            checked={retrievalSetting.enable_rerank ?? false}
                            onCheckedChange={(checked) => {
                              setRetrievalSetting((prev) => ({
                                ...prev,
                                enable_rerank: Boolean(checked),
                              }));
                            }}
                            className="h-3.5 w-3.5"
                          />
                        </div>
                        {retrievalSetting.enable_rerank && (
                          <>
                            <div className="flex items-center gap-2">
                              <Label htmlFor="rerank_model" className="w-[80px] text-xs">
                                重排序模型
                              </Label>
                              <Select
                                value={retrievalSetting.rerank_model || ''}
                                onValueChange={(value) => {
                                  setRetrievalSetting((prev) => ({
                                    ...prev,
                                    rerank_model: value,
                                  }));
                                }}
                              >
                                <SelectTrigger className="w-40 h-6 text-xs">
                                  <SelectValue placeholder="请选择重排序模型" />
                                </SelectTrigger>
                                <SelectContent className="text-xs">
                                  <SelectGroup>
                                    {rerankerModels.map((model) => (
                                      <SelectItem key={model.id} value={model.model_id} className="text-xs h-5">
                                        {model.model_id}
                                      </SelectItem>
                                    ))}
                                  </SelectGroup>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="flex items-center gap-2">
                              <Label htmlFor="rerank_top_k" className="w-[80px] text-xs">
                                Rerank-Top-K
                              </Label>
                              <Slider
                                className="w-40"
                                defaultValue={[5]}
                                max={20}
                                min={1}
                                step={1}
                                value={[retrievalSetting.rerank_top_k ?? 5]}
                                onValueChange={(value: number[]) => {
                                  setRetrievalSetting((prev) => ({
                                    ...prev,
                                    rerank_top_k: value[0],
                                  }));
                                }}
                              />
                              <span className="font-medium ml-2 text-xs">
                                {retrievalSetting.rerank_top_k ?? 5}
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                      
                      {/* 保存按钮 */}
                      <div className="flex items-center border-t gap-3 pt-3">
                        <Button
                          type="button"
                          onClick={handleSaveRetrievalSetting}
                          className="whitespace-nowrap h-8 text-xs px-2 hover:bg-gray-700 text-white"
                          size="sm"
                        >
                          <Save className="w-3 h-3 mr-1" />
                          应用到知识库设置
                        </Button>
                        <div className="text-xs text-muted-foreground flex items-center gap-1"><InfoIcon className="w-4 h-4" />保存后会更改知识库检索配置</div>
                      </div>
                    </CardContent>
                  )}
                </Card>
              </div>

              {/* 右侧：查询结果 */}
              <div className="flex-1 overflow-y-auto min-h-0">
                {/* 搜索结果提示 */}
                {searching && (
                  <div className="flex items-center space-x-3 p-3">
                    <Skeleton className="h-8 w-8 rounded-full" />
                    <div className="space-y-2">
                      <Skeleton className="h-3 w-[250px]" />
                      <Skeleton className="h-3 w-[200px]" />
                    </div>
                  </div>
                )}
                {!searching && searchError && (
                  <div className="p-6">
                    <Alert variant="destructive">
                      <AlertCircleIcon className="h-4 w-4" />
                      <AlertTitle className="text-sm">检索失败</AlertTitle>
                      <AlertDescription className="text-xs mt-2">
                        {searchError}
                      </AlertDescription>
                    </Alert>
                  </div>
                )}
                {!searching && !searchError && searchrecords.length === 0 && (
                  <div className="text-center py-6 text-gray-500">
                    <h2 className="text-sm">没有找到相关的切片</h2>
                    <p className="mt-2 text-xs">尝试调整搜索条件</p>
                  </div>
                )}
                {!searching && (
                  <div className="flex flex-col gap-2 w-full max-w-full overflow-x-hidden pb-4">
                    {searchrecords.map((chunk, i) => {
                      const isExpanded = expandedCards[i] || false;
                      return (
                        <Card
                          key={i}
                          className="w-full max-w-full border shadow-none hover:bg-muted/50 transition-colors cursor-pointer py-2 gap-1 overflow-hidden"
                          onClick={() => {
                            setExpandedCards(prev => ({
                              ...prev,
                              [i]: !prev[i]
                            }));
                          }}
                        >
                          <CardHeader className="px-3 py-0">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-1.5 flex-wrap min-w-0">
                                <Badge className="bg-red-600/10 dark:bg-red-600/20 hover:bg-red-600/10 text-red-500 border-red-600/60 shadow-none rounded-full text-xs h-5 shrink-0">
                                  {i + 1}
                                </Badge>
                                <Badge className="bg-amber-600/10 dark:bg-amber-600/20 hover:bg-amber-600/10 text-amber-500 border-amber-600/60 shadow-none rounded-full text-xs h-5 shrink-0">
                                  分数: {chunk.score.toFixed(4)}
                                </Badge>
                                <Badge className="bg-blue-600/10 dark:bg-blue-600/20 hover:bg-blue-600/10 text-blue-500 border-blue-600/60 shadow-none rounded-full text-xs h-5 shrink-0">
                                  {chunk.title}
                                </Badge>
                                {chunk.metadata.rerank && (
                                  <Badge className="bg-green-600/10 dark:bg-green-600/20 hover:bg-green-600/10 text-green-500 border-green-600/60 shadow-none rounded-full text-xs h-5 shrink-0">
                                    Rerank
                                  </Badge>
                                )}
                              </div>
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4 text-muted-foreground shrink-0" />
                              ) : (
                                <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
                              )}
                            </div>
                          </CardHeader>
                          <CardContent 
                            className={`px-3 pb-0 overflow-hidden transition-all duration-200 ${
                              isExpanded ? 'max-h-none' : ''
                            }`}
                          >
                            <div className={`text-xs leading-relaxed break-words ${
                              !isExpanded ? 'line-clamp-3' : ''
                            }`}>
                              {chunk.content.replace(/\n/g, '\\n')}
                            </div>
                            {chunk.metadata?.images_info?.length > 0 && (
                              <div className="flex gap-2 mt-2 flex-wrap">
                                {chunk.metadata.images_info.map((meta, index) => (
                                  <PhotoProvider
                                    key={index}
                                    maskOpacity={0.8}
                                    overlayRender={() => {
                                      return (
                                        <div className="absolute left-0 bottom-0 p-3 w-full min-h-30 text-xs text-slate-300 z-50 bg-black/50">
                                          <div>图片描述：{meta.desc}</div>
                                        </div>
                                      );
                                    }}
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
                          </CardContent>
                        </Card>
                      );
                    })}
                    { searchrecords.length > 0 && (
                      <p className="text-xs text-center text-muted-foreground pt-4 pb-4"> 没有更多内容了 </p>
                    )}

                  </div>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* 元数据管理Dialog */}
      <Dialog open={metadataConfigDialogOpen} onOpenChange={setMetadataConfigDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="text-sm">元数据配置</DialogTitle>
            <DialogDescription className="text-xs">
              管理知识库的元数据配置
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
              <PlusIcon className="h-3 w-3 mr-1" /> 添加元数据
            </Button>
            <div className="flex flex-col gap-2 max-h-[400px] overflow-y-auto">
              {metadataConfigs.length === 0 ? (
                <div className="text-center py-4 text-xs text-muted-foreground">
                  暂无元数据配置
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
              {editingMetadataConfig ? '编辑元数据' : '添加元数据'}
            </DialogTitle>
            <DialogDescription className="text-xs">
              请设定一个元数据名称（英文和数字），如city, category，用于在知识库内检索。
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-2 py-2">
            <div className="grid gap-2">
              <Label htmlFor="metadata_key" className="text-xs">元数据名称</Label>
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
                  <SelectValue placeholder="选择值类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    <SelectLabel className="text-xs">值类型</SelectLabel>
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
              <Label htmlFor="metadata_desc" className="text-xs">元数据描述</Label>
              <Input
                id="metadata_desc"
                className="h-6 text-xs"
                placeholder="输入元数据相关描述。"
                value={newMetadataDesc}
                onChange={(e) => setNewMetadataDesc(e.target.value)}
              />
            </div>
          </div>
          {metadataError ? (
            <Alert variant="destructive" className="text-xs py-2">
              <AlertCircleIcon className="h-3 w-3" />
              <AlertTitle className="text-xs">操作失败</AlertTitle>
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
              取消
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
    </div>
  );
}
