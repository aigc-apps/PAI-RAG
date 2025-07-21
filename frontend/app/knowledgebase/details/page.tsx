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
import { Loader2, CheckCircle, XCircle } from "lucide-react";
import { PreviewButton } from "@/app/knowledgebase/details/preview-button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { PlusIcon } from "lucide-react";
import * as Toast from "@radix-ui/react-toast";
import { KbConfig, KbConfigCard } from "../kbconfig";
import { formatFileSize, formatBeijingTime } from "../utils/utils";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import { PhotoProvider, PhotoView } from "react-photo-view";
import "react-photo-view/dist/react-photo-view.css";
import { Badge } from "@/components/ui/badge";

interface KnowledgeBaseFile {
  id: string;
  file_name: string;
  file_size: string;
  status: string;
  created_at: string;
  update_at: string;
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
  };
}

interface EmbeddingModel {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
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
  // 递归更新嵌套对象
  const updateNestedObject = (
    obj: Record<string, any>,
    keys: string[],
    val: any,
  ): any => {
    const [currentKey, ...rest] = keys;
    const value = Array.isArray(val) ? val[0] : val;

    if (rest.length === 0) {
      return {
        ...obj,
        [currentKey]: value,
      };
    }

    return {
      ...obj,
      [currentKey]: updateNestedObject(obj[currentKey], rest, value),
    };
  };

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const [embRes] = await Promise.all([
          fetch(`http://localhost:${port}/v1/config/embeddings`),
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

  const handleQueryInputChange = (
    e:
      | React.ChangeEvent<HTMLInputElement>
      | React.ChangeEvent<HTMLTextAreaElement>,
  ) => {
    const { id, value } = e.target;
    setKbQuery(value);
  };

  const handleSearchSubmit = async () => {
    console.log("handleSearchSubmit");
    const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
    const search_result = await fetch(
      `http://localhost:${port}/v1/config/knowledgebases/retrieval`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: kbquery,
          knowledgebase_id: knowledgebase_id,
        }),
      },
    );
    if (!search_result.ok) throw new Error("搜索知识库失败");

    const search_json = await search_result.json();
    console.log("搜索知识库结果:", search_json);
    setSearchRecords(search_json.data.records);
  };

  useEffect(() => {
    pageRef.current = page;
  }, [page]);

  const fetchKbFiles = useCallback(async () => {
    const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
    const url = `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}/files?page=${pageRef.current}&size=${fileSizePerPage}`;

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
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const res = await fetch(
          `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}`,
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
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(
        `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}`,
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
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
      const res = await fetch(
        `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}/files`,
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
              onSaveSuccess={handleSaveSuccess}
              onCancel={() => {}}
            ></KbConfigCard>
          </TabsContent>
          <TabsContent value="retrieval_test" className="py-4">
            <div className="space-y-4">
              {/* 搜索框和按钮 */}
              <div className="flex flex-wrap gap-2 mb-6">
                <div className="flex-1 min-w-[200px] max-w-[1000px]">
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
                <Button
                  type="button"
                  onClick={handleSearchSubmit}
                  className="whitespace-nowrap"
                >
                  查询
                </Button>
              </div>
              {/* 搜索结果提示 */}
              {searchrecords.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <h2>没有找到相关的切片</h2>
                  <p className="mt-2 text-sm">尝试调整搜索条件</p>
                </div>
              )}
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
