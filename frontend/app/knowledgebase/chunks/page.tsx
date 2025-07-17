"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Edit } from "lucide-react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PhotoProvider, PhotoView } from "react-photo-view";
import "react-photo-view/dist/react-photo-view.css";

interface KnowledgeBase {
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
    similarity_threshold: string; // 相似度分数阈值
    rerank_model: string; // rerank模型名称
    vector_weight?: string; // 向量检索权重（仅 hybrid 时使用）
  };
}

interface KnowledgeBaseFile {
  id: string;
  file_name: string;
  file_size: string;
  file_extension: string;
  file_metadata: {
    file_url: string;
  };
  update_at: string;
}

interface ImageInfo {
  url: string;
  desc: string;
}

interface KbFileChunk {
  id: string;
  text: string;
  chunk_metadata: {
    images_info: Array<ImageInfo>;
  };
  status: string;
  active: boolean;
  created_at: string;
  update_at: string;
}

// 状态映射
const statusMap: Record<string, string> = {
  succeeded: "bg-blue-100 text-blue-800",
  failed: "bg-green-100 text-green-800",
  pending: "bg-yellow-100 text-yellow-800",
};

const activeMap: Record<string, string> = {
  false: "bg-red-100 text-red-800",
  true: "bg-green-100 text-green-800",
};
export default function KnowledgeBaseFileChunksPage({
  knowledgebase_file_id,
  setActiveTab,
}: {
  knowledgebase_file_id: string;
  setActiveTab: (tab: string) => void;
}) {
  const [knowledgebase_id, file_id] = knowledgebase_file_id.split("__");
  const [knowledgebase, setKnowledgeBase] = useState<KnowledgeBase>(); // 知识库详情
  const [knowledgebaseloading, setKnowledgeBaseLoading] = useState(true); // 知识库加载状态
  const [knowledgebaseerror, setKnowledgeBaseError] = useState(""); // 知识库错误信息

  const [kbfile, setKbFile] = useState<KnowledgeBaseFile>(); // 文件详情
  const [kbfileloading, setKbFileLoading] = useState(true); // 文件加载状态
  const [kbfileerror, setKbFileError] = useState(""); // 文件错误信息

  const [kbfilechunks, setKbFileChunks] = useState(Array<KbFileChunk>); // 文件切片列表详情
  const [kbfilechunksloading, setKbFileChunksLoading] = useState(true); // 文件加载状态
  const [kbfilechunkserror, setKbFilChunksError] = useState(""); // 文件错误信息

  const [isZoomed, setIsZoomed] = useState(false);

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
        setKnowledgeBaseError(err || "加载失败");
      } finally {
        setKnowledgeBaseLoading(false);
      }
    };
    const fetchKbFile = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const res = await fetch(
          `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}`,
        );
        if (!res.ok) throw new Error("获取知识库文件失败");
        const json_data = await res.json();
        const kb_file_data = json_data.data;

        setKbFile(kb_file_data); // 更新状态
        console.log("知识库文件详情数据:", kb_file_data);
      } catch (err: any) {
        setKbFileError(err || "加载失败");
      } finally {
        setKbFileLoading(false);
      }
    };

    const fetchKbFileChunks = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const res = await fetch(
          `http://localhost:${port}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}/chunks`,
        );
        if (!res.ok) throw new Error("获取知识库文件切片列表失败");
        const json_data = await res.json();
        const kb_file_chunks_data = json_data.data;

        setKbFileChunks(kb_file_chunks_data); // 更新状态
        console.log("知识库文件切片列表详情数据:", kb_file_chunks_data);
      } catch (err: any) {
        setKbFilChunksError(err || "加载失败");
      } finally {
        setKbFileChunksLoading(false);
      }
    };
    fetchKbConfigs();
    fetchKbFile();
    fetchKbFileChunks();
  }, []);
  if (!knowledgebase || !kbfile) {
    return <div className="p-6">加载中...</div>;
  }
  return (
    <div className="flex flex-col h-screen">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="mb-2 flex items-center gap-2">
            {/* 面包屑导航 */}
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
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() =>
                        setActiveTab(
                          `/knowledgebase/details/${knowledgebase.id}`,
                        )
                      }
                    >
                      {knowledgebase.name}
                    </Button>
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem>
                  <BreadcrumbPage>{kbfile.file_name}</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
          <div className="mb-2 flex items-center gap-2">
            <Button
              variant="outline"
              className="h-8 w-8"
              onClick={() =>
                setActiveTab(`/knowledgebase/details/${knowledgebase.id}`)
              }
            >
              <ArrowLeft />
            </Button>
            <h1 className="text-xl font-bold pl-2">文件切片列表</h1>
          </div>
        </div>
      </div>
      {/* 可滚动内容区域 */}
      <div className="flex-1 overflow-y-auto">
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          {kbfilechunksloading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">加载中...</p>
            </div>
          ) : kbfilechunkserror ? (
            <div className="py-12 text-center text-red-500">
              <p>切片列表加载失败</p>
            </div>
          ) : kbfilechunks.length === 0 ? (
            <h3 className="text-lg font-medium text-gray-700 py-6">暂无模型</h3>
          ) : (
            <div className="gap-6 p-4 w-full">
              <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                {kbfilechunks.map((chunk) => (
                  <Card key={chunk.id} className="flex flex-col max-h-80">
                    <CardHeader>
                      <CardTitle className="flex justify-between items-start">
                        <Badge className={activeMap[String(chunk.active)]}>
                          {chunk.active ? "已激活" : "未激活"}
                        </Badge>
                        {/* TODO: 修改切片的状态（是否激活） */}
                        <Switch
                          checked={chunk.active}
                          className="ml-auto rounded-full transition-color"
                          // onCheckedChange={() => handleActivateToggle(chunk.active)}
                        />
                        <button
                          className="text-black-500 hover:text-black-700 px-2"
                          // onClick={() => handleEditClick(config)}
                        >
                          <Edit className="w-5 h-5" />
                        </button>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow overflow-y-auto">
                      <ScrollArea className="h-full pr-4">
                        <div className="text-gray-600 whitespace-pre-wrap">
                          {chunk.text}
                        </div>
                      </ScrollArea>
                    </CardContent>
                    <CardFooter className="shrink-0 gap-2">
                      {chunk.chunk_metadata.images_info.map((meta, index) => (
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
                            <img src={meta.url} className="w-10 h-10" />
                          </PhotoView>
                        </PhotoProvider>
                      ))}
                    </CardFooter>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
