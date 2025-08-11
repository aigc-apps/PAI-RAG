"use client";
import React, { useState, useEffect, useCallback, useRef } from "react";
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
import { Label } from "@/components/ui/label";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  SheetFooter,
  SheetClose,
} from "@/components/ui/sheet";

import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

import {
  Loader2,
  CheckCircle,
  XCircle,
  Trash2Icon,
  AlertCircleIcon,
  SearchIcon,
  ChevronDownIcon,
} from "lucide-react";
import { PreviewButton } from "@/app/knowledgebase/details/preview-button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { PlusIcon, FilterIcon } from "lucide-react";
import * as Toast from "@radix-ui/react-toast";
import { KbConfig, KbConfigCard, MetadataConfig } from "../kbconfig";
import { formatFileSize, formatBeijingTime } from "../utils/utils";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import { PhotoProvider, PhotoView } from "react-photo-view";
import "react-photo-view/dist/react-photo-view.css";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Skeleton } from "@/components/ui/skeleton";
import { DatetimeInput } from "../datetime";
import { Role } from "@/app/config/role/role";

interface KnowledgeBaseFile {
  id: string;
  file_name: string;
  file_size: string;
  status: string;
  file_source: string;
  created_at: string;
  updated_at: string;
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

interface EmbeddingModel {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
}

interface MetadataCondition {
  name: string;
  comparison_operator: string;
  value: string | number;
}

export default function KnowledgeBaseDetailPage({
  knowledgebase_id,
  setActiveTab,
}: {
  knowledgebase_id: string;
  setActiveTab: (tab: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [knowledgebase, setKnowledgeBase] = useState<KbConfig>(); // 知识库列表
  const [kbfiles, setKbFiles] = useState(Array<KnowledgeBaseFile>); // 知识库列表
  const [page, setPage] = useState(1);
  const pageRef = useRef(page);
  const [totalPages, setTotalPages] = useState(1);
  const fileSizePerPage = 8;
  const [kbquery, setKbQuery] = useState(""); //查询
  const [searchrecords, setSearchRecords] = useState(Array<SearchRecord>); // 搜索结果
  const [searching, setSearching] = useState(false);
  const [logicalOperator, setLogicalOperator] = useState<string>("and");
  const [metadataConditions, setMetadataConditions] = useState<
    MetadataCondition[]
  >([]);
  const [fileSource, setFileSource] = useState("");
  const [fileSourceOpen, setFileSourceOpen] = useState<Record<string, boolean>>(
    {},
  );

  const [knowledgebasesloading, setKnowledgeBasesLoading] = useState(true); // 加载状态
  const [knowledgebasesrror, setKnowledgeBasesError] = useState(""); // 错误信息
  const [embeddingmodels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(""); // 错误信息
  const [toastState, setToastState] = useState({
    open: false,
    title: "",
    description: "",
    variant: "default" as "default" | "destructive",
  });
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
  const [metadataEditError, setMetadataEditError] = useState<string>("");
  const [availableMetadataKeys, setAvailableMetadataKeys] = useState<string[]>(
    [],
  );

  const [roles, setRoles] = useState<Role[]>([]);
  const [openRole, setOpenRole] = useState(false);
  const [editRoleFileId, setEditRoleFileId] = useState("");
  const [activeRoleIds, setActiveRoleIds] = useState<string[]>([]);
  const [activeRoleNames, setActiveRoleNames] = useState<string[]>([]);
  const [user, setUser] = useState("");

  const default_comparator = [
    "contains",
    "not contains",
    "start with",
    "end with",
    "is",
    "is not",
    "empty",
    "not empty",
    "=",
    "≠",
    ">",
    "<",
    "≥",
    "≤",
    "before",
    "after",
  ];
  const default_metadata_keys = [
    "file_name",
    "file_path",
    "file_size",
    "file_extension",
    "file_url",
    "doc_id",
  ];

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const [embRes] = await Promise.all([fetch("/v1/config/embeddings")]);

        const embData = (await embRes.json())?.data.items || [];
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
    console.log("handleSearchSubmit");
    const search_result = await fetch("/v1/retrieval", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: kbquery,
        user_id: user,
        knowledge_id: knowledgebase_id,
        metadata_condition: {
          conditions: metadataConditions,
          logical_operator: logicalOperator,
        },
      }),
    });
    if (!search_result.ok) throw new Error("搜索知识库失败");

    const search_json = await search_result.json();
    console.log("搜索知识库结果:", search_json);
    setSearchRecords(search_json.records);
    setSearching(false);
  };

  useEffect(() => {
    pageRef.current = page;
  }, [page]);

  const fetchKbMetadata = async () => {
    const res = await fetch(
      `/v1/config/knowledgebases/${knowledgebase_id}/metadata`,
    );
    if (!res.ok) throw new Error("获取知识库元数据失败");
    const metadata_json = await res.json();
    const metadata_data = metadata_json.data as MetadataConfig[];
    const valueTypes = Object.fromEntries(
      metadata_data.map((metadata) => [metadata.name, metadata.value_type]),
    ) as { [key: string]: string };

    console.log("知识库元数据: ", metadata_data, valueTypes);

    setMetadataValueTypes({ ...valueTypes, "": "string" });
    setMetadataConfigs(metadata_data);
  };

  const fetchKbFiles = useCallback(async () => {
    const url = `/v1/config/knowledgebases/${knowledgebase_id}/files?page=${pageRef.current}&size=${fileSizePerPage}`;

    try {
      const files_res = await fetch(url);
      if (!files_res.ok) throw new Error("获取知识库文件列表失败");

      const file_json_data = await files_res.json();
      console.log("获取知识库文件reponse:", file_json_data);
      const data = file_json_data.data.items;
      setKbFiles(data || []);
      setTotalPages(file_json_data.data.pages);

      const kb_files = data as KnowledgeBaseFile[];
      const files_unfinished = kb_files.some(
        (file) => file.status !== "succeeded" && file.status !== "failed",
      );

      if (files_unfinished) {
        console.log("存在未完成的文件，继续检查状态。");
        setTimeout(() => {
          fetchKbFiles(); // 依赖 ref 获取最新 page
        }, 3000);
      } else {
        console.log("文件已上传完成。");
      }
    } catch (err) {
      console.error("获取知识库文件失败:", err);
    }
  }, [knowledgebase_id]);

  useEffect(() => {
    fetchKbFiles();
  }, [fetchKbFiles, page]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  useEffect(() => {
    const fetchKbConfigs = async () => {
      try {
        const res = await fetch(
          `/v1/config/knowledgebases/${knowledgebase_id}`,
        );
        if (!res.ok) throw new Error("获取知识库列表失败");
        const json_data = await res.json();
        const kb_data = json_data.data;

        setKnowledgeBase(kb_data); // 更新状态
        console.log("知识库详情数据:", kb_data);
      } catch (err: any) {
        setKnowledgeBasesError(err || "加载失败");
      } finally {
        setKnowledgeBasesLoading(false);
      }
    };
    fetchKbConfigs();
    fetchKbMetadata();
  }, []);

  if (!knowledgebase) {
    return <div className="p-6">加载中...</div>;
  }

  const handleSaveSuccess = (kb: KbConfig) => {
    setToastState({
      open: true,
      title: `知识库${knowledgebase_id} 配置已修改`,
      description: "修改的模型配置已成功保存",
      variant: "default",
    });
    console.log(`update ${knowledgebase_id}`);
  };

  const handleDeleteFile = async (file_id: string) => {
    setDeleting(true);
    try {
      const res = await fetch(
        `/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}`,
        {
          method: "DELETE",
        },
      );
      if (!res.ok) throw new Error(`删除 ${file_id} 失败`);
      console.log("delete file result:", res.text());
    } catch (error) {
      console.error("删除失败:", error);
    } finally {
      setDeleting(false);
      fetchKbFiles();
    }
  };

  const handleSaveFileSource = async (file_id: string) => {
    try {
      const res = await fetch(
        `/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}/source`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_source: fileSource,
          }),
        },
      );
      if (!res.ok) throw new Error("Failed to save file source");

      const fileObj = kbfiles.filter((file) => file.id === file_id)[0];
      if (fileObj) {
        fileObj.file_source = fileSource;
      }
      setFileSourceOpen((prev) => ({ ...prev, [file_id]: false }));
    } catch (error) {
      console.error("Error fetching file source:", error);
      throw error;
    }
  };

  const selectMetadataKey = async (metadata_key: string) => {
    setMetadataEditError("");
    const emptyKeys = Object.keys(editingMetadata).filter(
      (key) => editingMetadata[key] === "",
    );
    if (emptyKeys.length > 1) throw new Error("有多于一个新建项。");
    else if (emptyKeys.length === 0) return;
    else {
      editingMetadata[metadata_key] = editingMetadata[""];
      delete editingMetadata[""];
      const updatedUsableKeys = availableMetadataKeys.filter(
        (name) => name !== metadata_key,
      );
      setAvailableMetadataKeys(updatedUsableKeys);
      console.log("selected keys for metadata: ", editingMetadata);
      setEditingMetadata({ ...editingMetadata });
    }
  };

  const handleOpenMetadata = async (file_id: string) => {
    setMetadataEditError("");
    setIsEditingMetadata(false);
    try {
      const file_res = await fetch(
        `/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}`,
      );
      if (!file_res.ok) throw new Error(`获取 ${file_id} 失败`);
      const file_json = await file_res.json();
      setEditingMetadata(file_json.data.file_metadata);
      const usable_metadata_keys = metadataConfigs
        .map((metadata) => metadata.name)
        .filter((name) => !(name in file_json.data.file_metadata));
      setAvailableMetadataKeys(usable_metadata_keys);
      console.log("可用的metadata名称：", availableMetadataKeys);
    } catch (err) {
      console.error("获取文件失败:", err);
    }
  };

  const handAddFileMetadata = () => {
    if (availableMetadataKeys.length === 0) {
      setMetadataEditError(
        "没有可用的自定义的元数据配置，你可以先去知识库设置页面添加。",
      );
      return;
    }
    const hasEmptyEntry = Object.keys(editingMetadata).some(
      (key) => editingMetadata[key] === "",
    );
    if (!hasEmptyEntry) {
      editingMetadata[""] = "";
      setEditingMetadata({ ...editingMetadata });
      setMetadataEditError("");
    } else {
      console.log("已经有一个待添加的项目了。");
      setMetadataEditError("");
    }
  };

  const handleDeleteMetadata = (name: string) => {
    console.log("删除metadata:", name, editingMetadata);
    if (name in editingMetadata) {
      delete editingMetadata[name];
      setEditingMetadata(editingMetadata);
      const usable_metadata_keys = metadataConfigs
        .map((metadata) => metadata.name)
        .filter((name) => !(name in editingMetadata));
      setAvailableMetadataKeys(usable_metadata_keys);
      console.log("可用的metadata名称：", availableMetadataKeys);

      setMetadataEditError("");
      console.log("已删除metadata:", name, editingMetadata);
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
      const roleRes = await fetch("/v1/config/roles?size=100");
      if (!roleRes.ok) {
        alert("查询角色失败");
        return;
      }
      const all_roles = (await roleRes.json()).data.items;
      setRoles(all_roles);

      const permission_name = file_id;
      const res = await fetch(
        `/v1/config/roles/permissions?name=${permission_name}&size=100`,
      );
      if (!res.ok) {
        alert("查询文件permission失败");
        return;
      }

      const permission_res = await res.json();
      const role_ids = permission_res.data.items.map(
        (item: any) => item.role_id,
      );
      const role_names = all_roles
        .filter((role: any) => role_ids.includes(role.id))
        .map((role: any) => role.name);
      console.log("role_ids:", role_ids);
      console.log("role_names:", role_names);

      setActiveRoleIds(role_ids);
      setActiveRoleNames(role_names);
    } catch (error) {
      console.error("获取文件角色信息失败: ", error);
    }
  };

  const saveFilePermission = async () => {
    try {
      const roleRes = await fetch(
        `/v1/config/roles/permissions/files/${editRoleFileId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            role_ids: activeRoleIds,
          }),
        },
      );
      if (!roleRes.ok) {
        alert("更新文件角色失败");
        return;
      }
      console.log("更新文件角色成功：", await roleRes.json());
      setOpenRole(false);
    } catch (error) {
      console.error("上传失败:", error);
    }
  };

  const handleFileUpload = async (files: FileList | null) => {
    console.log("##handleFileUpload", files);
    if (!files) {
      alert("文件列表为空！");
      return;
    }
    setUploading(true);

    // 文件校验 (Demo功能，后续调整优化)
    const validFiles = Array.from(files).filter((file) => {
      // const isValidType = ['application/pdf', 'application/msword'].includes(file.type);
      const isValidSize = file.size <= 10 * 1024 * 1024;
      // return isValidType && isValidSize;
      return isValidSize;
    });

    if (validFiles.length === 0) {
      alert("请选择有效的文件（如 PDF 或 Word，且小于 10MB）");
      setUploading(false);
      return;
    }

    // 上传文件
    const formData = new FormData();
    validFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const res = await fetch(
        `/v1/config/knowledgebases/${knowledgebase_id}/files`,
        {
          method: "POST",
          body: formData,
        },
      );
      if (!res.ok) {
        alert("上传失败");
        return;
      }
      const upload_result = await res.json();
      console.log("上传成功:", upload_result);
    } catch (error) {
      console.error("上传失败:", error);
    } finally {
      setUploading(false);
      // 清空文件选择框
      if (fileInputRef.current) {
        fileInputRef.current.value = ""; // 清空 input 的值
      }
      setPage(1);
      fetchKbFiles();
    }
  };

  const get_metadata_id = (name: string) => {
    console.log("get id", metadataConfigs, name);
    return metadataConfigs.filter((metadata) => metadata.name === name)[0].id;
  };

  const saveEditMetadata = async (file_id: string) => {
    const hasEmptyEntry = Object.keys(editingMetadata).some(
      (key) => editingMetadata[key] === "",
    );
    if (hasEmptyEntry) {
      setMetadataEditError("无法保存空的元数据名称。");
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
        `/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}/metadata`,
        {
          method: "POST",
          body: JSON.stringify(bodyData),
          headers: {
            "Content-Type": "application/json",
          },
        },
      );
      if (!res.ok) throw Error("保存metadata失败");
      const file_result = (await res.json()).data as KnowledgeBaseFile;
      const updated_kbfiles = kbfiles;
      const target_file_index = updated_kbfiles.findIndex(
        (file) => file.id === file_id,
      );
      updated_kbfiles[target_file_index] = file_result;
      setKbFiles(updated_kbfiles);
      console.log("更新文件成功：", updated_kbfiles);
      setIsEditingMetadata(false);
    } catch (error: any) {
      console.log("保存metadata失败", error);
    } finally {
      setMetadataEditError("");
    }
  };

  const addCondition = () => {
    const newCondition = {
      name: "",
      comparison_operator: "",
      value: "",
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
        if (metadataValueTypes[name] === "datetime") {
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
    <div className="flex flex-col h-screen w-full">
      <div className="flex-none">
        <div className="p-4 flex">
          <div className="gap-1 flex items-center">
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() => setActiveTab("/knowledgebase")}
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
          <TabsContent value="details" className="py-4">
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>
                  <div className="flex justify-between items-center">
                    <Button
                      onClick={() =>
                        document.getElementById("file-upload")?.click()
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
                    <input
                      id="file-upload"
                      type="file"
                      className="hidden"
                      ref={fileInputRef}
                      onChange={(e) => handleFileUpload(e.target.files)}
                    />
                    <div className="text-xs text-muted-foreground">
                      支持的文件类型：txt, md, pdf, docx, pptx, xlsx, xls, html,
                      jsonl, jpg, jpeg, png{" "}
                    </div>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {kbfiles && kbfiles.length > 0 ? (
                  <div>
                    <Table className="min-w-full">
                      <TableHeader>
                        <TableRow>
                          <TableHead>文件名</TableHead>
                          <TableHead>文件大小</TableHead>
                          <TableHead>上传时间</TableHead>
                          <TableHead>更新时间</TableHead>
                          <TableHead>状态</TableHead>
                          <TableHead>操作</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {kbfiles.map((file) => (
                          <TableRow key={file.id}>
                            <TableCell>
                              <Button
                                variant="link"
                                className="font-medium text-blue-600"
                                onClick={() =>
                                  setActiveTab(
                                    `/knowledgebase/chunks/${knowledgebase_id}__${file.id}`,
                                  )
                                }
                              >
                                {file.file_name}
                              </Button>
                            </TableCell>
                            <TableCell>
                              {formatFileSize(Number(file.file_size))}
                            </TableCell>
                            <TableCell>
                              {formatBeijingTime(file.created_at)}
                            </TableCell>
                            <TableCell>
                              {formatBeijingTime(file.updated_at)}
                            </TableCell>
                            <TableCell>
                              {file.status === "pending" ? (
                                <div className="flex items-center text-yellow-500">
                                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                                  等待解析
                                </div>
                              ) : file.status === "parsing" ? (
                                <div className="flex items-center text-blue-500">
                                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                                  解析中
                                </div>
                              ) : file.status === "persisting" ? (
                                <div className="flex items-center text-blue-500">
                                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                                  索引中
                                </div>
                              ) : file.status === "succeeded" ? (
                                <div className="flex items-center text-green-500">
                                  <CheckCircle className="mr-1 h-4 w-4" />
                                  解析成功
                                </div>
                              ) : file.status === "failed" ? (
                                <div className="flex items-center text-red-500">
                                  <XCircle className="mr-1 h-4 w-4" />
                                  解析失败
                                </div>
                              ) : (
                                <span>{file.status}</span> // 兜底显示原始状态
                              )}
                            </TableCell>
                            <TableCell>
                              <PreviewButton
                                kbId={knowledgebase_id}
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
                                <PopoverTrigger asChild>
                                  <Button
                                    variant="link"
                                    className="text-sm text-blue-600"
                                  >
                                    源连接
                                  </Button>
                                </PopoverTrigger>
                                <PopoverContent className="w-160">
                                  <div className="flex gap-3">
                                    <Label>{file.file_name}</Label>
                                    <Input
                                      type="text"
                                      className="w-130"
                                      placeholder="输入文件外部源链接，如语雀、飞书、钉钉文档等。"
                                      value={fileSource || ""}
                                      onChange={(e) => {
                                        setFileSource(e.target.value);
                                      }}
                                    />
                                    <Button
                                      onClick={() =>
                                        handleSaveFileSource(file.id)
                                      }
                                    >
                                      {" "}
                                      保存{" "}
                                    </Button>
                                  </div>
                                </PopoverContent>
                              </Popover>

                              <Button
                                variant="link"
                                className="text-sm text-blue-600"
                                onClick={() =>
                                  setActiveTab(
                                    `/knowledgebase/chunks/${knowledgebase_id}__${file.id}`,
                                  )
                                }
                              >
                                查看切片
                              </Button>

                              <Sheet open={openRole} onOpenChange={setOpenRole}>
                                <SheetTrigger asChild>
                                  <Button
                                    variant="link"
                                    onClick={() => {
                                      checkFileRole(file.id);
                                    }}
                                    className="text-sm text-blue-600"
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
                                    className="text-sm text-blue-600"
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
                                                {key !== "" ? (
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
                                                  "datetime" ? (
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
                                                    className="w-3 h-3"
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
                                    {metadataEditError !== "" && (
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
                              <Button
                                variant="link"
                                className="text-sm text-blue-600"
                                onClick={() => handleDeleteFile(file.id)}
                              >
                                {deleting ? (
                                  <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    删除中...
                                  </>
                                ) : (
                                  <>删除</>
                                )}
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
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
            <div className="space-y-4">
              {/* 搜索框和按钮 */}
              <div className="flex flex-wrap gap-2 mb-6">
                <div className="flex-1 min-w-[200px] max-w-[640px]">
                  <Input
                    type="text"
                    id="search_query"
                    placeholder="请输入查询内容"
                    onChange={handleQueryInputChange}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        handleSearchSubmit();
                      }
                    }}
                    className="w-full"
                  />
                </div>
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
                                "datetime" ? (
                                  <DatetimeInput
                                    value={
                                      typeof condition.value === "number"
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
              {/* 搜索结果提示 */}
              {searching && (
                <div className="flex items-center space-x-4">
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
                  <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-4 gap-4">
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
                                  overlayRender={({}) => {
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
