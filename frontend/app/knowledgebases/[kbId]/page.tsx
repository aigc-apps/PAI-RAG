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

import {
  Loader2,
  CheckCircle,
  XCircle,
  Trash2Icon,
  AlertCircleIcon,
  SearchIcon,
  ChevronDownIcon,
  RefreshCcwIcon,
  Search,
} from 'lucide-react';
import { PreviewButton } from '@/app/knowledgebases/[kbId]/preview-button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { PlusIcon, FilterIcon } from 'lucide-react';
import * as Toast from '@radix-ui/react-toast';
import { KbConfig, KbConfigCard, MetadataConfig } from '../kbconfig';
import { formatFileSize, formatBeijingTime } from '../utils/utils';
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
import { Checkbox } from '@/components/ui/checkbox';
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
  file_metadata: {
    [key: string]: any;
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
  const [searchrecords, setSearchRecords] = useState(Array<SearchRecord>); // 搜索结果
  const [searching, setSearching] = useState(false);
  const [logicalOperator, setLogicalOperator] = useState<string>('and');
  const [metadataConditions, setMetadataConditions] = useState<
    MetadataCondition[]
  >([]);
  const [fileSource, setFileSource] = useState('');
  const [fileSourceOpen, setFileSourceOpen] = useState<Record<string, boolean>>(
    {},
  );
  const { kbId } = use(params);

  let isRefreshing = false;
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [isEditingMetadata, setIsEditingMetadata] = useState(false);
  const [editingMetadata, setEditingMetadata] = useState<{ [k: string]: any }>(
    {},
  );
  const [metadataConfigs, setMetadataConfigs] = useState<MetadataConfig[]>([]);
  const [metadataValueTypes, setMetadataValueTypes] = useState<{
    [k: string]: any;
  }>({});
  const [metadataEditError, setMetadataEditError] = useState<string>('');
  const [availableMetadataKeys, setAvailableMetadataKeys] = useState<string[]>(
    [],
  );

  const [roles, setRoles] = useState<Role[]>([]);
  const [openRole, setOpenRole] = useState(false);
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
  const [vectorDbType, setVectorDbType] = useState<string>('local');
  
  // 不支持全文检索和混合检索的向量数据库类型列表
  const VECTOR_DB_TYPES_WITHOUT_FULLTEXT = ['local', 'opensearch', 'hologres'];
  
  const isFulltextSupported = !VECTOR_DB_TYPES_WITHOUT_FULLTEXT.includes(vectorDbType);

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
    console.log('handleSearchSubmit');
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
    if (!search_result.ok) throw new Error('搜索知识库失败');

    const search_json = await search_result.json();
    console.log('搜索知识库结果:', search_json);
    setSearchRecords(search_json.records);
    setSearching(false);
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

  useEffect(() => {
    fetchKbFiles();
  }, [fetchKbFiles, page, statusFilter, fileQuery]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  useEffect(() => {
    const fetchKbConfigs = async () => {
      try {
        const [kbRes, metaRes, rerankerRes, vectordbRes] = await Promise.all([
          fetch(`/api/config/knowledgebases/${kbId}`),
          fetch(`/api/config/knowledgebases/${kbId}/metadata`),
          fetch(`/api/config/rerankers`),
          fetch(`/api/config/vectordb`),
        ]);

        if (!kbRes.ok) throw new Error('获取知识库配置失败');
        const json_data = await kbRes.json();
        const kb_data = json_data.data;

        setKnowledgeBase(kb_data); // 更新状态
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
    };
    fetchKbConfigs();
  }, []);

  if (!knowledgebase) {
    return <div className="p-6">加载中...</div>;
  }

  const handleSaveSuccess = (kb: KbConfig) => {
      toast.success("知识库配置保存成功");
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

  const handleSaveFileSource = async (file_id: string) => {
    try {
      const res = await fetch(
        `/api/config/knowledgebases/${kbId}/files/${file_id}/source`,
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

      const fileObj = kbfiles.filter((file) => file.id === file_id)[0];
      if (fileObj) {
        fileObj.file_source = fileSource;
      }
      setFileSourceOpen((prev) => ({ ...prev, [file_id]: false }));
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
      setOpenRole(false);
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

  const saveEditMetadata = async (file_id: string) => {
    const hasEmptyEntry = Object.keys(editingMetadata).some(
      (key) => editingMetadata[key] === '',
    );
    if (hasEmptyEntry) {
      setMetadataEditError('无法保存空的元数据名称。');
      return;
    }

    try {
      const metadata_enties = Object.keys(editingMetadata)
        .filter((name) => !default_metadata_keys.includes(name))
        .map((name) => ({
          name: name,
          metadata_id: get_metadata_id(name),
          value: editingMetadata[name],
        }));
      const bodyData = {
        entries: metadata_enties,
      };
      const res = await fetch(
        `/api/config/knowledgebases/${kbId}/files/${file_id}/metadata`,
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
        (file) => file.id === file_id,
      );
      updated_kbfiles[target_file_index] = file_result;
      setKbFiles(updated_kbfiles);
      console.log('更新文件成功：', updated_kbfiles);
      setIsEditingMetadata(false);
    } catch (error: any) {
      console.log('保存metadata失败', error);
    } finally {
      setMetadataEditError('');
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
      <div className="px-4 py-2 flex">
        <div className="gap-1 flex items-center">
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
        </div>
        <div className="max-w-120 ml-auto ">
          <div className="gap-3 text-xs">
            <span className="font-medium">ID: </span>
            {knowledgebase.id}
          </div>
          <div className="gap-3 text-xs truncate">
            <span className="font-medium">描述: </span>
            {knowledgebase.description}
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-2">
        <Tabs defaultValue="details">
          <TabsList className="py-4 bg-muted rounded-lg flex-none">
            <TabsTrigger value="details" className="p-4">
              文件管理
            </TabsTrigger>
            <TabsTrigger value="settings" className="p-4">
              知识库设置
            </TabsTrigger>
            <TabsTrigger value="retrieval_test" className="p-4">
              检索测试
            </TabsTrigger>
          </TabsList>
          <TabsContent value="details" className="py-3">
            <Card className="mb-4">
              <CardHeader>
                <CardTitle>
                  <div className="flex items-center">
                    <div className="flex items-center justify-between w-full">
                      <Button
                        onClick={() =>
                          document.getElementById('file-upload')?.click()
                        }
                        disabled={uploading} // 上传时禁用按钮
                      >
                        {uploading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            上传中...
                          </>
                        ) : (
                          <>
                            上传文件
                            <PlusIcon className="mr-2 h-6 w-6" />
                          </>
                        )}
                      </Button>
                      <div className="flex gap-2 items-center">
                        <input
                          id="file-upload"
                          type="file"
                          className="hidden"
                          ref={fileInputRef}
                          onChange={(e) => handleFileUpload(e.target.files)}
                          multiple
                        />
                        <Button
                          variant="outline"
                          className="ml-4 h-8"
                          onClick={() => {
                            fetchKbFiles();
                            toast.success("刷新成功");
                          }}
                        > 刷新
                          <RefreshCcwIcon/>
                        </Button>
                        <div className="text-xs text-muted-foreground ">
                          支持的文件类型：txt, md, pdf, docx, pptx, xlsx, xls, html,
                          jsonl, jpg, jpeg, png{' '}
                        </div>
                      </div>
                    </div>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                  <div>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>
                            <div className="flex gap-2 items-center max-w-[400px]">                            文件名
                          <Search className="h-6 w-6 text-muted-foreground" />
                          <Input
                            value={fileQuery}
                            onChange={(e)=>{setFileQuery(e.target.value)}}
                            type="search_files"
                            placeholder="Search filename..."/>
                            </div>
                          </TableHead>
                          <TableHead>文件大小</TableHead>
                          <TableHead>上传时间</TableHead>
                          <TableHead>更新时间</TableHead>
                          <TableHead>
                            <div className="flex items-center">
                              <Select value={statusFilter} onValueChange={setStatusFilter}>
                              <SelectTrigger className="w-[100px] bg-muted/50 hover:bg-muted">
                                <SelectValue placeholder="全部状态" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="all">全部</SelectItem>
                                <SelectItem value="succeeded"><span className="text-green-500">成功</span></SelectItem>
                                <SelectItem value="failed"><span className="text-red-500">失败</span></SelectItem>
                                <SelectItem value="pending"><span className="text-yellow-500">等待中</span></SelectItem>
                                <SelectItem value="parsing"><span className="text-blue-500">解析中</span></SelectItem>
                                <SelectItem value="persisting"><span className="text-blue-500">索引中</span></SelectItem>
                              </SelectContent>
                            </Select>

                            </div>
                          </TableHead>
                          <TableHead>操作</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {kbfiles.map((file) => (
                          <TableRow key={file.id}>
                            <TableCell>
                              <Button
                                variant="link"
                                className="font-medium text-blue-600 max-w-[360px]"
                                onClick={() =>
                                  router.push(
                                    `/knowledgebases/${kbId}/files/${file.id}`,
                                  )
                                }
                              > 
                              <span className="truncate block w-full text-left">
                                {file.file_name}            
                              </span>
                              </Button>
                            </TableCell>
                            <TableCell className="text-xs">
                              {formatFileSize(Number(file.file_size))}
                            </TableCell>
                            <TableCell className="text-xs">
                              {formatBeijingTime(file.created_at)}
                            </TableCell>
                            <TableCell className="text-xs">
                              {formatBeijingTime(file.updated_at)}
                            </TableCell>
                            <TableCell className="text-xs">
                              {file.status === 'pending' ? (
                                <div className="flex items-center text-yellow-500">
                                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                                  等待解析
                                </div>
                              ) : file.status === 'parsing' ? (
                                <div className="flex items-center text-blue-500">
                                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                                  解析中
                                </div>
                              ) : file.status === 'persisting' ? (
                                <div className="flex items-center text-blue-500">
                                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                                  索引中
                                </div>
                              ) : file.status === 'succeeded' ? (
                                <div className="flex items-center text-green-500">
                                  <CheckCircle className="mr-1 h-4 w-4" />
                                  解析成功
                                </div>
                              ) : file.status === 'failed' ? (
                                    <HoverCard>
                                      <HoverCardTrigger asChild>
                                        <div className="flex items-center text-red-500">
                                          <XCircle className="mr-1 h-4 w-4" />
                                          解析失败
                                        </div>
                                      </HoverCardTrigger>
                                      <HoverCardContent className="w-80">
                                        错误原因: {file.failed_reason}
                                      </HoverCardContent>
                                    </HoverCard>

                              ) : (
                                <span>{file.status}</span> // 兜底显示原始状态
                              )}
                            </TableCell>
                            <TableCell className="gap-1">
                              <PreviewButton
                                kbId={kbId}
                                fileId={file.id}
                              />

                              <Popover
                                open={fileSourceOpen[file.id] ?? false}
                                onOpenChange={(open) => {
                                  if (open) {
                                    setFileSource(file.file_source);
                                  }
                                  setFileSourceOpen((prev) => ({
                                    ...prev,
                                    [file.id]: open,
                                  }));
                                }}
                              >
                              <Button
                                variant="link"
                                className="text-sm text-blue-600 pl-3 pr-0"
                                onClick={() =>
                                  router.push(
                                    `/knowledgebases/${kbId}/files/${file.id}`,
                                  )
                                }
                              >
                                切片
                              </Button>

                              <Sheet open={openRole} onOpenChange={setOpenRole}>
                                <SheetTrigger asChild>
                                  <Button
                                    variant="link"
                                    onClick={() => {
                                      checkFileRole(file.id);
                                    }}
                                    className="text-sm text-blue-600 pl-3 pr-0"
                                  >
                                    权限
                                  </Button>
                                </SheetTrigger>
                                <SheetContent>
                                  <SheetHeader>
                                    <SheetTitle>文档权限设置</SheetTitle>
                                  </SheetHeader>
                                  <div className="grid flex-1 auto-rows-min gap-6 px-4">
                                    <div>
                                      {activeRoleNames.length > 0 ? (
                                        <div>
                                          <div className="text-sm">
                                            以下角色有查看/搜索该文档的权限
                                          </div>

                                          <div className="flex pt-3 gap-1.5 items-center">
                                            {activeRoleNames.map((name) => (
                                              <Badge
                                                variant="secondary"
                                                className="h-6"
                                                key={name}
                                              >
                                                {name}
                                              </Badge>
                                            ))}
                                          </div>
                                        </div>
                                      ) : (
                                        <div>
                                          所有角色都有查看/搜索该文档的权限。添加角色来限制文档访问。
                                        </div>
                                      )}
                                    </div>
                                    <div className="grid gap-3">
                                      <div className="flex">
                                        <Label
                                          htmlFor="kb_selection"
                                          className="w-[90px]"
                                        >
                                          角色选择
                                        </Label>
                                        <div className="pl-6 pr-6">
                                          {roles.length > 0 ? (
                                            <DropdownMenu modal={true}>
                                              <DropdownMenuTrigger asChild>
                                                <Button
                                                  variant="outline"
                                                  className="text-sm text-muted-foreground"
                                                >
                                                  已选{activeRoleIds.length}
                                                  个，可多选 <ChevronDownIcon />
                                                </Button>
                                              </DropdownMenuTrigger>
                                              <DropdownMenuContent className="w-56">
                                                <DropdownMenuLabel>
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
                                                  >
                                                    {role.name}
                                                  </DropdownMenuCheckboxItem>
                                                ))}
                                              </DropdownMenuContent>
                                            </DropdownMenu>
                                          ) : (
                                            <div>
                                              <p className="text-sm text-muted-foreground">
                                                尚未配置角色信息，前往`权限控制`设置。
                                              </p>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                  <div className="flex flex-col gap-4 pb-6 px-6">
                                    <Button onClick={saveFilePermission}>
                                      保存
                                    </Button>
                                    <Button onClick={clearAllRoles}>
                                      重置（设为所有角色可访问）
                                    </Button>

                                    <Button
                                      variant="outline"
                                      onClick={() => setOpenRole(false)}
                                    >
                                      取消
                                    </Button>
                                  </div>
                                </SheetContent>
                              </Sheet>
                              <Sheet>
                                <SheetTrigger asChild>
                                  <Button
                                    variant="link"
                                    className="text-sm text-blue-600 pl-3 pr-0"
                                    onClick={() => handleOpenMetadata(file.id)}
                                  >
                                    元数据
                                  </Button>
                                </SheetTrigger>
                                <SheetContent className="sm:max-w-[750px] w-[600px] sm:w-[540px]">
                                  <SheetHeader>
                                    {isEditingMetadata ? (
                                      <SheetTitle>编辑元数据</SheetTitle>
                                    ) : (
                                      <SheetTitle>查看元数据</SheetTitle>
                                    )}
                                  </SheetHeader>
                                  <div className="grid flex-1 auto-rows-min gap-2 px-4">
                                    <div className="space-y-1 text-xs">
                                      {isEditingMetadata ? (
                                        <Label htmlFor="sheet-custom-meta">
                                          自定义
                                          <Button
                                            variant="secondary"
                                            className="w-16 h-5"
                                            onClick={handAddFileMetadata}
                                          >
                                            <PlusIcon className="h-3 w-3" />
                                            添加
                                          </Button>
                                        </Label>
                                      ) : (
                                        <Label htmlFor="sheet-custom-meta">
                                          自定义
                                        </Label>
                                      )}
                                      {Object.keys(editingMetadata).filter(
                                        (key: string) =>
                                          !default_metadata_keys.includes(key),
                                      ).length === 0 && (
                                        <p>
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
                                              className="flex items-start space-x-2"
                                              key={key}
                                            >
                                              {key !== '' ? (
                                                <div className="system-xs-medium w-[128px] shrink-0 items-center truncate py-1 text-text-tertiary font-semibold">
                                                  {key}
                                                </div>
                                              ) : (
                                                <Select
                                                  onValueChange={(value) =>
                                                    selectMetadataKey(value)
                                                  }
                                                  defaultOpen={true}
                                                >
                                                  <SelectTrigger className="w-[88px] h-4 text-xs system-xs-medium w-[128px] shrink-0 items-center">
                                                    <SelectValue placeholder="选择元数据名称" />
                                                  </SelectTrigger>
                                                  <SelectContent className="w-[88px] text-xs">
                                                    <SelectGroup>
                                                      {availableMetadataKeys.map(
                                                        (m_key) => (
                                                          <SelectItem
                                                            key={m_key}
                                                            value={m_key}
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
                                                        metadataValueTypes[key]
                                                      }
                                                      className="w-[280px] border-transparent focus:shadow-xs radius-md h-5 grow p-0.5 text-xs rounded-md"
                                                      value={
                                                        editingMetadata[key]
                                                      }
                                                      onChange={(e) => {
                                                        setEditingMetadata({
                                                          ...editingMetadata,
                                                          [key]: e.target.value,
                                                        });
                                                      }}
                                                    />
                                                  ) : (
                                                    <DatetimeInput
                                                      value={
                                                        editingMetadata[key]
                                                      }
                                                      width="md"
                                                      onValueChange={(
                                                        value,
                                                      ) => {
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
                                                  {editingMetadata[key]}
                                                </div>
                                              </div>
                                            </div>
                                          ))}
                                    </div>
                                    <div className="text-xs">
                                      <Label htmlFor="sheet-custom-meta">
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
                                                {editingMetadata[key]}
                                              </div>
                                            </div>
                                          </div>
                                        ))}
                                    </div>
                                  </div>
                                  <SheetFooter>
                                    {metadataEditError !== '' && (
                                      <Alert variant="destructive">
                                        <AlertCircleIcon />
                                        <AlertDescription>
                                          <p>{metadataEditError}</p>
                                        </AlertDescription>
                                      </Alert>
                                    )}
                                    {isEditingMetadata ? (
                                      <Button
                                        type="button"
                                        onClick={() =>
                                          saveEditMetadata(file.id)
                                        }
                                      >
                                        保存
                                      </Button>
                                    ) : (
                                      <Button
                                        type="button"
                                        onClick={() =>
                                          setIsEditingMetadata(true)
                                        }
                                      >
                                        编辑
                                      </Button>
                                    )}

                                    <SheetClose asChild>
                                      <Button
                                        variant="outline"
                                        onClick={() =>
                                          setIsEditingMetadata(false)
                                        }
                                      >
                                        Close
                                      </Button>
                                    </SheetClose>
                                  </SheetFooter>
                                </SheetContent>
                              </Sheet>

                                                              <PopoverTrigger asChild>
                                  <Button
                                    variant="link"
                                    className="text-sm text-blue-600 pl-3 pr-0"
                                  >
                                    源链接
                                  </Button>
                                </PopoverTrigger>
                                <PopoverContent className="w-160">
                                  <div className="flex gap-3">
                                    <Label>{file.file_name}</Label>
                                    <Input
                                      type="text"
                                      className="w-130"
                                      placeholder="输入文件外部源链接，如语雀、飞书、钉钉文档等。"
                                      value={fileSource || ''}
                                      onChange={(e) => {
                                        setFileSource(e.target.value);
                                      }}
                                    />
                                    <Button
                                      onClick={() =>
                                        handleSaveFileSource(file.id)
                                      }
                                    >
                                      {' '}
                                      保存{' '}
                                    </Button>
                                  </div>
                                </PopoverContent>
                              </Popover>

                              <Button
                                variant="link"
                                className="text-sm text-blue-600 pr-0"
                                onClick={() => handleReprocessFile(file.id)}
                              >
                                  重新解析
                              </Button>

                              <Button
                                variant="link"
                                className="text-sm text-blue-600"
                                onClick={() => handleDeleteFile(file.id)}
                              >
                                  删除
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                
                { kbfiles.length === 0 && (
                  <p className="text-muted-foreground mx-auto">暂无文件</p>
                )}
                <PaginationComponent
                  currentPage={page}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="settings" className="py-4">
            <KbConfigCard
              isCreate={false}
              kbConfig={knowledgebase}
              metadataConfigs={metadataConfigs}
              onSaveSuccess={handleSaveSuccess}
              onCancel={() => {}}
            ></KbConfigCard>
          </TabsContent>
          <TabsContent value="retrieval_test" className="py-4">
            <div className="flex gap-4 h-full">
              {/* 左侧：查询输入和检索设置 */}
              <div className="flex flex-col w-[450px] shrink-0 h-full justify-between">
                {/* 检索测试输入区域 - 左上角 */}
                <Card className="flex-[4] flex flex-col min-h-0 mb-3">
                  <CardHeader className="pb-4 flex-shrink-0">
                    <CardTitle className="text-lg">检索测试</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col min-h-0">
                    {/* 搜索框和按钮 */}
                    <div className="flex flex-col gap-3 flex-1">
                      <div className="flex-1 min-w-[200px]">
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
                          className="w-full h-28 text-lg"
                        />
                      </div>
                      <div className="flex flex-wrap gap-2 flex-shrink-0">
                        <Popover>
                          <PopoverTrigger asChild>
                            <Button variant="outline">
                              <FilterIcon />
                              元数据
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-[450px]">
                            <div className="grid gap-4">
                              <div className="space-y-2">
                                <RadioGroup
                                  value={logicalOperator}
                                  onValueChange={(value) => setLogicalOperator(value)}
                                >
                                  <div className="flex items-center space-x-2">
                                    <p className="text-muted-foreground text-sm">
                                      逻辑操作符
                                    </p>

                                    <RadioGroupItem value="and" id="r1" />
                                    <Label htmlFor="r1">AND</Label>
                                    <RadioGroupItem value="or" id="r2" />
                                    <Label htmlFor="r2">OR</Label>
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
                                          <SelectTrigger className="h-4 text-xs system-xs-medium shrink-0 items-center">
                                            <SelectValue placeholder="名称" />
                                          </SelectTrigger>
                                          <SelectContent className="text-xs">
                                            <SelectGroup>
                                              {metadataConfigs.map((metadata) => (
                                                <SelectItem
                                                  key={metadata.name}
                                                  value={metadata.name}
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
                                          <SelectTrigger className="h-4 text-xs system-xs-medium shrink-0 items-center">
                                            <SelectValue placeholder="规则" />
                                          </SelectTrigger>
                                          <SelectContent className="w-[80px] text-xs">
                                            <SelectGroup>
                                              {default_comparator.map((op) => (
                                                <SelectItem key={op} value={op}>
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
                                                typeof condition.value === 'number'
                                                  ? condition.value
                                                  : parseFloat(condition.value)
                                              }
                                              width="sm"
                                              onValueChange={(value) => {
                                                setConditionValue(i, value);
                                              }}
                                            />
                                          ) : (
                                            <Input
                                              className="w-128px"
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
                                          className="w-6"
                                        >
                                          <Trash2Icon className="w-4 h-4" />
                                        </Button>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                                <Button
                                  variant="secondary"
                                  onClick={addCondition}
                                  className="h-6 text-xs"
                                >
                                  新增过滤规则
                                </Button>
                              </div>
                            </div>
                          </PopoverContent>
                        </Popover>
                        <Input
                          className="w-30 text-xs"
                          placeholder="输入user_id"
                          value={user}
                          onChange={(e) => {
                            setUser(e.target.value);
                          }}
                        />
                        <Button
                          type="button"
                          onClick={handleSearchSubmit}
                          className="whitespace-nowrap"
                        >
                          <SearchIcon />
                          开始查询
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* 检索设置板块 - 左下角 */}
                <Card className={`flex-[0.8] overflow-y-auto text-xs min-h-0 ${!retrievalSettingOpen ? 'p-0' : ''}`}>
                  <CardHeader className={retrievalSettingOpen ? "pb-2 px-3 pt-3" : "py-0 px-3"}>
                    <div className="flex items-center justify-between h-7">
                      <CardTitle className="text-lg">检索设置</CardTitle>
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
                    <CardContent className="space-y-2 px-3 pb-3">
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
                                className="!rounded-full px-1.5 py-0.5 text-xs data-[state=on]:bg-black data-[state=on]:text-white"
                              >
                                <TextSearch className="w-2 h-2 mr-0.5" />
                                全文检索
                              </ToggleGroupItem>
                            )}
                            {isFulltextSupported && (
                              <ToggleGroupItem
                                value="hybrid"
                                aria-label="混合检索"
                                className="!rounded-full px-1.5 py-0.5 text-xs data-[state=on]:bg-black data-[state=on]:text-white"
                              >
                                <SearchCode className="w-2 h-2 mr-0.5" />
                                混合检索
                              </ToggleGroupItem>
                            )}
                          </ToggleGroup>
                        </div>
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
                        {retrievalSetting.retrieval_mode === 'hybrid' && !retrievalSetting.enable_rerank && (
                          <div className="flex gap-2 items-center">
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
                        {retrievalSetting.retrieval_mode === 'hybrid' && !retrievalSetting.enable_rerank && (
                          <p className="text-xs text-muted-foreground ml-[88px]">
                            向量权重仅在未开启重排序时生效
                          </p>
                        )}
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
                                <SelectTrigger className="w-40 h-7 text-xs">
                                  <SelectValue placeholder="请选择重排序模型" />
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
                      <div className="flex flex-col items-end pt-2 border-t gap-1">
                        <Button
                          type="button"
                          onClick={handleSaveRetrievalSetting}
                          className="whitespace-nowrap h-7 text-xs px-3 bg-black hover:bg-black/90 text-white"
                          size="sm"
                        >
                          <Save className="w-3 h-3 mr-1" />
                          保存至知识库设置
                        </Button>
                        <p className="text-xs text-muted-foreground">保存后会更改知识库检索配置</p>
                      </div>
                    </CardContent>
                  )}
                </Card>
              </div>

              {/* 右侧：查询结果 */}
              <div className="flex-1 overflow-y-auto">
                {/* 搜索结果提示 */}
                {searching && (
                  <div className="flex items-center space-x-4 p-4">
                    <Skeleton className="h-12 w-12 rounded-full" />
                    <div className="space-y-2">
                      <Skeleton className="h-4 w-[250px]" />
                      <Skeleton className="h-4 w-[200px]" />
                    </div>
                  </div>
                )}
                {!searching && searchrecords.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <h2>没有找到相关的切片</h2>
                    <p className="mt-2 text-sm">尝试调整搜索条件</p>
                  </div>
                )}
                {!searching && (
                  <div className="gap-6 p-4 w-full">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      {searchrecords.map((chunk, i) => (
                        <Card
                          key={i}
                          className="flex flex-col max-h-64 gap-0 pb-0 py-4"
                        >
                          <CardHeader className="gap-1 pb-0 ">
                            <CardTitle className="flex justify-start">
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge className="bg-red-600/10 dark:bg-red-600/20 hover:bg-red-600/10 text-red-500 border-red-600/60 shadow-none rounded-full">
                                  {i + 1}
                                </Badge>
                                <Badge className="bg-amber-600/10 dark:bg-amber-600/20 hover:bg-amber-600/10 text-amber-500 border-amber-600/60 shadow-none rounded-full">
                                  分数: {chunk.score.toFixed(4)}
                                </Badge>
                                <Badge className="bg-blue-600/10 dark:bg-blue-600/20 hover:bg-blue-600/10 text-blue-500 border-blue-600/60 shadow-none rounded-full">
                                  {chunk.title}
                                </Badge>
                                {chunk.metadata.rerank && (
                                  <Badge className="bg-green-600/10 dark:bg-green-600/20 hover:bg-green-600/10 text-green-500 border-green-600/60 shadow-none rounded-full">
                                    Rerank
                                  </Badge>
                                )}
                              </div>
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="bg-gray-200/10 flex-grow overflow-y-auto overflow-x-auto pr-3 p-3 pb-2 mt-1 mb-1">
                            <div className="whitespace-pre-wrap break-words text-sm leading-relaxed whitespace-normal pr-2">
                              {chunk.content}
                            </div>
                          </CardContent>
                          <CardFooter className="shrink-0 gap-2">
                            {chunk.metadata?.images_info?.length > 0 && (
                              <div className="flex gap-2 mt-4">
                                {chunk.metadata.images_info.map((meta, index) => (
                                  <PhotoProvider
                                    key={index}
                                    maskOpacity={0.8}
                                    overlayRender={() => {
                                      return (
                                        <div className="absolute left-0 bottom-0 p-4 w-full min-h-30 text-sm text-slate-300 z-50 bg-black/50">
                                          <div>图片描述：{meta.desc}</div>
                                        </div>
                                      );
                                    }}
                                  >
                                    <PhotoView key={index} src={meta.url}>
                                      <img
                                        src={meta.url}
                                        className="w-10 h-10 object-cover rounded-md cursor-pointer"
                                      />
                                    </PhotoView>
                                  </PhotoProvider>
                                ))}
                              </div>
                            )}
                          </CardFooter>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
