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

interface KnowledgeBaseFile {
  id: string;
  file_name: string;
  file_size: string;
  status: string;
  created_at: string;
  update_at: string;
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
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
        const [embRes] = await Promise.all([
          fetch(`${API_BASE}/v1/config/embeddings`),
        ]);

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
    const API_BASE =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
    const search_result = await fetch(
      `${API_BASE}/v1/config/knowledgebases/retrieval`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: kbquery,
          knowledgebase_id: knowledgebase_id,
          metadata_condition: {
            conditions: metadataConditions,
            logical_operator: logicalOperator,
          },
        }),
      },
    );
    if (!search_result.ok) throw new Error("搜索知识库失败");

    const search_json = await search_result.json();
    console.log("搜索知识库结果:", search_json);
    setSearchRecords(search_json.data.records);
    setSearching(false);
  };

  useEffect(() => {
    pageRef.current = page;
  }, [page]);

  const fetchKbMetadata = async () => {
    const API_BASE =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";

    const res = await fetch(
      `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/metadata`,
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
    const API_BASE =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
    const url = `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files?page=${pageRef.current}&size=${fileSizePerPage}`;

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
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
        const res = await fetch(
          `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}`,
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
      const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
      const res = await fetch(
        `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}`,
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

  const selectMetadataKey = async (metadata_key: string) => {
    setMetadataEditError("");
    const emptyKeys = Object.keys(editingMetadata).filter(
      (key) => editingMetadata[key] === "",
    );
    if (emptyKeys.length > 1) throw new Error(`有多于一个新建项。`);
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
      const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
      const file_res = await fetch(
        `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}`,
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
      const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
      const res = await fetch(
        `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files`,
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
      const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8680";
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
        `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}/metadata`,
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
      let updated_kbfiles = kbfiles;
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
    <div className="flex flex-col h-screen">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="mb-6 flex items-center gap-2">
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
                  <BreadcrumbPage>{knowledgebase.name}</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-4">
        <Tabs defaultValue="details">
          <TabsList className="py-4 bg-muted rounded-lg flex-none">
            <TabsTrigger value="details" className="p-4">
              文件列表
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
                <CardTitle>知识库：{knowledgebase.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between items-center">
                  <p className="text-muted-foreground mb-4">
                    ID：{knowledgebase.id}
                  </p>
                  <p className="text-muted-foreground mb-4">
                    描述：{knowledgebase.description}
                  </p>
                  <p className="text-muted-foreground mb-4">
                    支持的文件类型：txt, md, pdf, docx, pptx, xlsx, xls, html,
                    jsonl, jpg, jpeg, png{" "}
                  </p>
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
                        <PlusIcon className="mr-2 h-4 w-4" />
                        上传文件
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
                </div>

                {kbfiles && kbfiles.length > 0 ? (
                  <>
                    <h3 className="text-lg font-semibold mt-6 mb-3">
                      文件列表
                    </h3>
                    <ScrollArea className="h-[480px] rounded-md border overflow-x-auto">
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
                                {formatBeijingTime(file.update_at)}
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
                                <Sheet>
                                  <SheetTrigger asChild>
                                    <Button
                                      variant="link"
                                      className="text-sm text-blue-600"
                                      onClick={() =>
                                        handleOpenMetadata(file.id)
                                      }
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
                                            !default_metadata_keys.includes(
                                              key,
                                            ),
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
                                                          metadataValueTypes[
                                                            key
                                                          ]
                                                        }
                                                        className="w-[280px] border-transparent focus:shadow-xs radius-md h-5 grow p-0.5 text-xs rounded-md"
                                                        value={
                                                          editingMetadata[key]
                                                        }
                                                        onChange={(e) => {
                                                          setEditingMetadata({
                                                            ...editingMetadata,
                                                            [key]:
                                                              e.target.value,
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
                                                        handleDeleteMetadata(
                                                          key,
                                                        )
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
                                  onClick={() =>
                                    setActiveTab(
                                      `/knowledgebase/chunks/${knowledgebase_id}__${file.id}`,
                                    )
                                  }
                                >
                                  查看切片列表
                                </Button>
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
                                    <>删除文件</>
                                  )}
                                </Button>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  </>
                ) : (
                  <p className="text-muted-foreground">暂无文件</p>
                )}
              </CardContent>
              <CardFooter>
                <PaginationComponent
                  currentPage={page}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              </CardFooter>
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
                      <Card key={i} className="flex flex-col max-h-80">
                        <CardHeader>
                          <CardTitle className="flex justify-start">
                            <div className="flex items-center gap-3 flex-wrap">
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
                        <CardContent className="flex-grow overflow-y-auto">
                          <ScrollArea className="h-full pr-4">
                            <div className="text-gray-600 whitespace-pre-wrap">
                              {chunk.content}
                            </div>
                          </ScrollArea>
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
