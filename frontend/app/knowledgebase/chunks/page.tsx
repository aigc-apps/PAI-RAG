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
import { PhotoProvider, PhotoView } from "react-photo-view";
import "react-photo-view/dist/react-photo-view.css";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import parse, { Element, HTMLReactParserOptions } from "html-react-parser";

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
    enable_rerank: boolean;
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
  updated_at: string;
}

interface ImageInfo {
  url: string;
  desc: string;
}

interface KbFileChunk {
  id: string;
  file_id: string;
  kb_id: string;
  text: string;
  chunk_metadata: {
    images_info: Array<ImageInfo>;
  };
  status: string;
  active: boolean;
  created_at: string;
  updated_at: string;
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

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const chunksSizePerPage = 8;

  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editText, setEditText] = useState("");
  const [selectedChunk, setSelectedChunk] = useState<KbFileChunk | null>(null);

  useEffect(() => {
    const fetchKbConfigs = async () => {
      try {
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
        const res = await fetch(
          `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}`,
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
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
        const res = await fetch(
          `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}`,
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
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
        const res = await fetch(
          `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}/chunks?page=${page}&size=${chunksSizePerPage}`,
        );
        if (!res.ok) throw new Error("获取知识库文件切片列表失败");
        const json_data = await res.json();
        const kb_file_chunks_data = json_data.data.items;
        setTotalPages(json_data.data.pages);
        setKbFileChunks(kb_file_chunks_data || []); // 更新状态
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
  }, [page]);
  if (!knowledgebase || !kbfile) {
    return <div className="p-6">加载中...</div>;
  }

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const handleActivateToggle = async (chunk: KbFileChunk) => {
    chunk.active = !chunk.active;
    const API_BASE =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
    const url = `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}/chunks/${chunk.id}`;

    const res = await fetch(url, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chunk), // 包装为数组
    });

    if (!res.ok) throw new Error(`修改 ${chunk.id} 配置失败`);
    setKbFileChunks((prev) =>
      prev.map((c) => (c.id === chunk.id ? { ...c, active: chunk.active } : c)),
    );
  };

  const handleEditClick = (chunk: KbFileChunk) => {
    setSelectedChunk(chunk);
    setEditText(chunk.text);
    setIsEditOpen(true);
  };

  // --- 1. 通用的 isHtmlContent 函数 ---
  // 检查文本是否包含 <html> 标签（无论是作为开头还是嵌入其中）
  const isHtmlContent = (text: string) => {
    const trimmedText = text?.trim();
    // 只要包含 <html> 就认为是包含 HTML 内容
    return trimmedText?.includes("<html>");
  };

  // --- 2. 提取 <body> 内容的函数 (用于纯 HTML 文档) ---
  const extractBodyContent = (htmlString: string) => {
    const bodyStart = htmlString.indexOf("<body");
    const bodyEnd = htmlString.lastIndexOf("</body>");

    if (bodyStart !== -1 && bodyEnd !== -1) {
      const bodyOpenEnd = htmlString.indexOf(">", bodyStart);
      if (bodyOpenEnd !== -1) {
        return htmlString.substring(bodyOpenEnd + 1, bodyEnd).trim();
      }
    }
    return htmlString; // Fallback
  };

  // --- 3. 提取所有 <html>...</html> 块的函数 (用于嵌入 HTML 的文本) ---
  const extractHtmlBlocks = (text: string): string[] => {
    const htmlBlocks: string[] = [];
    const prefix = "<html>";
    const suffix = "</html>";
    let startIndex = 0;

    while (startIndex < text.length) {
      const blockStart = text.indexOf(prefix, startIndex);
      if (blockStart === -1) break; // No more <html> found

      const blockEnd = text.indexOf(suffix, blockStart);
      if (blockEnd === -1) break; // Unmatched <html>, stop processing

      // Extract the full <html>...</html> block
      const htmlBlock = text.substring(blockStart, blockEnd + suffix.length);
      htmlBlocks.push(htmlBlock);

      // Move the search start index past the end of the current block
      startIndex = blockEnd + suffix.length;
    }

    return htmlBlocks;
  };

  // --- 4. 定义 HTMLReactParserOptions (为表格和单元格添加样式) ---
  const tableOptions: HTMLReactParserOptions = {
    replace(domNode) {
      // 处理 <table> 标签
      if (domNode.type === "tag" && domNode.name === "table") {
        const element = domNode as Element;
        const existingTableClass =
          element.attribs.className || element.attribs.class || "";
        // 添加边框、边框合并、居中、上下边距
        const tableBorderClasses =
          "border border-collapse border-gray-500 mx-auto my-2";
        const updatedTableClassName = existingTableClass
          ? `${existingTableClass} ${tableBorderClasses}`
          : tableBorderClasses;
        element.attribs.className = updatedTableClassName;
        delete element.attribs.class;
        return undefined;
      }

      // 处理 <td> 和 <th> 标签
      if (
        domNode.type === "tag" &&
        (domNode.name === "td" || domNode.name === "th")
      ) {
        const element = domNode as Element;
        const existingCellClass =
          element.attribs.className || element.attribs.class || "";
        // 添加内边距和边框
        const cellBorderClasses = "border border-gray-500 p-2";
        const updatedCellClassName = existingCellClass
          ? `${existingCellClass} ${cellBorderClasses}`
          : cellBorderClasses;
        element.attribs.className = updatedCellClassName;
        delete element.attribs.class;
        return undefined;
      }

      // 对于其他元素，使用默认行为
      return undefined;
    },
  };

  // --- 5. 主要的 HTML 渲染函数 ---
  const renderHtmlContent = (text: string) => {
    const trimmedText = text.trim();

    // 情况 1: chunk.text 本身就是一个完整的 HTML 文档
    if (trimmedText.startsWith("<html>")) {
      try {
        // 提取 <body> 内容并解析
        const bodyContent = extractBodyContent(trimmedText);
        return parse(bodyContent, tableOptions);
      } catch (error) {
        console.error("Error parsing standalone HTML document:", error);
        // 如果解析失败，回退到显示原始文本
        return <pre>{text}</pre>;
      }
    }
    // 情况 2: chunk.text 是包含 HTML 块的文本
    else if (isHtmlContent(text)) {
      try {
        // 提取所有 <html>...</html> 块
        const htmlBlocks = extractHtmlBlocks(text);

        if (htmlBlocks.length > 0) {
          // 对每个 HTML 块提取 <body> 内容并解析，然后将结果组合起来
          return (
            <>
              {/* 渲染 HTML 块之前的部分文本 */}
              {text.substring(0, text.indexOf("<html>"))}
              {/* 渲染每个解析后的 HTML 块 */}
              {htmlBlocks.map((htmlBlock, index) => {
                const bodyContent = extractBodyContent(htmlBlock);
                // 使用 React Fragment 包裹，以防 parse 返回多个根元素
                // key 用于 React 列表渲染
                return (
                  <React.Fragment key={index}>
                    {parse(bodyContent, tableOptions)}
                  </React.Fragment>
                );
              })}
              {/* 渲染最后一个 HTML 块之后的部分文本 */}
              {text.substring(text.lastIndexOf("</html>") + "</html>".length)}
            </>
          );
        } else {
          return text;
        }
      } catch (error) {
        console.error("Error parsing embedded HTML blocks:", error);
        // 如果解析嵌入的 HTML 失败，回退到显示原始文本
        return <pre>{text}</pre>;
      }
    }
    // 情况 3: 不包含 HTML，按普通文本渲染
    else {
      return text;
    }
  };

  const handleSaveEdit = async () => {
    if (!selectedChunk) return;
    selectedChunk.text = editText;
    const API_BASE =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
    const url = `${API_BASE}/v1/config/knowledgebases/${knowledgebase_id}/files/${file_id}/chunks/${selectedChunk.id}`;

    try {
      const response = await fetch(url, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selectedChunk),
      });

      if (!response.ok) throw new Error("更新失败");

      // 更新本地状态
      setKbFileChunks((prev) =>
        prev.map((c) =>
          c.id === selectedChunk.id ? { ...c, text: selectedChunk.text } : c,
        ),
      );
      setIsEditOpen(false);
    } catch (err) {
      console.error("编辑失败:", err);
      // 可添加错误提示（如 toast）
    }
  };

  return (
    <div className="flex flex-col h-screen w-full">
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
      <div className="overflow-y-auto h-4/5">
        <div className="flex py-2 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          {kbfilechunksloading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">加载中...</p>
            </div>
          ) : kbfilechunkserror ? (
            <div className="py-12 text-center text-red-500">
              <p>切片列表加载失败</p>
            </div>
          ) : kbfilechunks.length === 0 ? (
            <h3 className="text-lg font-medium text-gray-700 py-6">暂无切片</h3>
          ) : (
            <div className="gap-4 p-4 w-full">
              <div className="grid grid-cols-4 items-center gap-6">
                {kbfilechunks.map((chunk) => (
                  <Card key={chunk.id} className="h-80 p-4 gap-4">
                    <CardHeader>
                      <CardTitle className="flex justify-between items-start">
                        <Badge className={activeMap[String(chunk.active)]}>
                          {chunk.active ? "已激活" : "未激活"}
                        </Badge>
                        <Switch
                          checked={chunk.active}
                          className="ml-auto rounded-full transition-color"
                          onCheckedChange={() => handleActivateToggle(chunk)}
                        />
                        <button
                          className="text-black-500 hover:text-black-700 px-2"
                          onClick={() => handleEditClick(chunk)}
                        >
                          <Edit className="w-5 h-5" />
                        </button>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="bg-gray-200/10 flex-grow overflow-y-auto overflow-x-auto pr-3 p-3 pb-2 mt-1 mb-1">
                      <div className="whitespace-pre-wrap break-words text-sm leading-relaxed whitespace-normal pr-2">
                        {renderHtmlContent(chunk.text)}
                      </div>
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
      <div>
        <PaginationComponent
          currentPage={page}
          totalPages={totalPages}
          onPageChange={handlePageChange}
        />
      </div>
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>编辑切片内容</DialogTitle>
            <DialogDescription>修改文本并保存</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <label className="block mb-2 text-sm font-medium">文本内容</label>
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full h-40 p-2 border rounded-md"
              placeholder="请输入新内容"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditOpen(false)}>
              取消
            </Button>
            <Button onClick={handleSaveEdit}>保存</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
