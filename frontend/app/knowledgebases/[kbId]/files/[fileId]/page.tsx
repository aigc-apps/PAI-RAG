'use client';
import React, { useState, useEffect, use } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Edit, Plus, Trash2Icon } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
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
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { useRouter } from 'next/navigation';
import { htmlRender } from "@/app/knowledgebases/[kbId]/viewer/htmlRender";
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

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
    token_count?: number;
  };
  status: string;
  active: boolean;
  created_at: string;
  updated_at: string;
}

// 状态映射
const statusMap: Record<string, string> = {
  succeeded: 'bg-blue-100 text-blue-800',
  failed: 'bg-green-100 text-green-800',
  pending: 'bg-yellow-100 text-yellow-800',
};

const activeMap: Record<string, string> = {
  false: 'bg-red-100 text-red-800',
  true: 'bg-green-100 text-green-800',
};
export default function KnowledgeBaseFileChunksPage(  
  { params } : { params: Promise<{ kbId: string, fileId: string }> }
) {
  const {kbId, fileId} = use(params);
  const [knowledgebase, setKnowledgeBase] = useState<KnowledgeBase>(); // 知识库详情
  const [knowledgebaseloading, setKnowledgeBaseLoading] = useState(true); // 知识库加载状态
  const [knowledgebaseerror, setKnowledgeBaseError] = useState(''); // 知识库错误信息

  const [kbfile, setKbFile] = useState<KnowledgeBaseFile>(); // 文件详情
  const [kbfileloading, setKbFileLoading] = useState(true); // 文件加载状态
  const [kbfileerror, setKbFileError] = useState(''); // 文件错误信息

  const [kbfilechunks, setKbFileChunks] = useState(Array<KbFileChunk>); // 文件切片列表详情
  const [kbfilechunksloading, setKbFileChunksLoading] = useState(true); // 文件加载状态
  const [kbfilechunkserror, setKbFilChunksError] = useState(''); // 文件错误信息

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const chunksSizePerPage = 8;

  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editText, setEditText] = useState('');
  const [selectedChunk, setSelectedChunk] = useState<KbFileChunk | null>(null);

  const [isAddOpen, setIsAddOpen] = useState(false);
  const [newChunkText, setNewChunkText] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const { tenantFetch } = useTenantFetch();
  const router = useRouter();

  useEffect(() => {
    const fetchKbConfigs = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/knowledgebases/${kbId}`,
        );
        if (!res.ok) throw new Error('获取知识库列表失败');
        const json_data = await res.json();
        const kb_data = json_data.data;

        setKnowledgeBase(kb_data); // 更新状态
        console.log('知识库详情数据:', kb_data);
      } catch (err: any) {
        setKnowledgeBaseError(err || '加载失败');
      } finally {
        setKnowledgeBaseLoading(false);
      }
    };
    const fetchKbFile = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/knowledgebases/${kbId}/files/${fileId}`,
        );
        if (!res.ok) throw new Error('获取知识库文件失败');
        const json_data = await res.json();
        const kb_file_data = json_data.data;

        setKbFile(kb_file_data); // 更新状态
        console.log('知识库文件详情数据:', kb_file_data);
      } catch (err: any) {
        setKbFileError(err || '加载失败');
      } finally {
        setKbFileLoading(false);
      }
    };

    const fetchKbFileChunks = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks?page=${page}&size=${chunksSizePerPage}`,
        );
        if (!res.ok) throw new Error('获取知识库文件切片列表失败');
        const json_data = await res.json();
        const kb_file_chunks_data = json_data.data.items;
        setTotalPages(json_data.data.pages);
        setKbFileChunks(kb_file_chunks_data || []); // 更新状态
        console.log('知识库文件切片列表详情数据:', kb_file_chunks_data);
      } catch (err: any) {
        setKbFilChunksError(err || '加载失败');
      } finally {
        setKbFileChunksLoading(false);
      }
    };
    fetchKbConfigs();
    fetchKbFile();
    fetchKbFileChunks();
  }, [page, fileId, kbId]);
  if (!knowledgebase || !kbfile) {
    return <div className="p-6">加载中...</div>;
  }

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const handleActivateToggle = async (chunk: KbFileChunk) => {
    chunk.active = !chunk.active;
    const url = `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks/${chunk.id}`;

    const res = await tenantFetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
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

  const handleDeleteClick = async (chunk: KbFileChunk) => {
    if (!confirm('确定要删除这个切片吗？此操作不可恢复。')) {
      return;
    }

    try {
      const url = `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks/${chunk.id}`;
      const response = await tenantFetch(url, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || '删除切片失败');
      }

      // 从本地状态中移除
      setKbFileChunks((prev) => prev.filter((c) => c.id !== chunk.id));

      // 如果当前页没有数据了，返回上一页
      if (kbfilechunks.length === 1 && page > 1) {
        setPage(page - 1);
      } else {
        // 刷新切片列表
        const res = await tenantFetch(
          `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks?page=${page}&size=${chunksSizePerPage}`,
        );
        if (res.ok) {
          const json_data = await res.json();
          setKbFileChunks(json_data.data.items || []);
          setTotalPages(json_data.data.pages);
        }
      }
    } catch (err: any) {
      console.error('删除切片失败:', err);
      alert(err.message || '删除切片失败');
    }
  };

  const handleSaveEdit = async () => {
    if (!selectedChunk) return;
    selectedChunk.text = editText;
    const url = `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks/${selectedChunk.id}`;

    try {
      const response = await tenantFetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selectedChunk),
      });

      if (!response.ok) throw new Error('更新失败');

      // 更新本地状态
      setKbFileChunks((prev) =>
        prev.map((c) =>
          c.id === selectedChunk.id ? { ...c, text: selectedChunk.text } : c,
        ),
      );
      setIsEditOpen(false);
    } catch (err) {
      console.error('编辑失败:', err);
      // 可添加错误提示（如 toast）
    }
  };

  const handleAddChunk = async () => {
    if (!newChunkText.trim()) {
      alert('请输入切片文本');
      return;
    }

    setIsAdding(true);
    try {
      const url = `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks`;
      const response = await tenantFetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: newChunkText,
          chunk_metadata: {},
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || '添加切片失败');
      }

      // 重置状态
      setNewChunkText('');
      setIsAddOpen(false);

      // 刷新切片列表
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks?page=${page}&size=${chunksSizePerPage}`,
      );
      if (res.ok) {
        const json_data = await res.json();
        setKbFileChunks(json_data.data.items || []);
        setTotalPages(json_data.data.pages);
      }
    } catch (err: any) {
      console.error('添加切片失败:', err);
      alert(err.message || '添加切片失败');
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-full">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="flex items-center gap-2">
            {/* 面包屑导航 */}
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
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() =>
                        router.push(
                          `/knowledgebases/${knowledgebase.id}`,
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
          <div className="mb-2 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                className="h-8 w-8"
                onClick={() =>
                  router.push(`/knowledgebases/${knowledgebase.id}`)
                }
              >
                <ArrowLeft />
              </Button>
              <h1 className="text-xl font-medium pl-2">文件切片列表</h1>
            </div>
            <Button
              variant="default"
              className="h-8"
              onClick={() => setIsAddOpen(true)}
            >
              <Plus className="w-4 h-4 mr-1" />
              新建切片
            </Button>
          </div>
        </div>
      </div>
      {/* 可滚动内容区域 */}
      <div className="overflow-y-auto h-4/5">
        <div className="flex border-dashed border-gray-200 rounded-xl p-0">
          {kbfilechunksloading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">加载中...</p>
            </div>
          ) : kbfilechunkserror ? (
            <div className="py-12 text-center text-red-500">
              <p>切片列表加载失败</p>
            </div>
          ) : kbfilechunks.length === 0 ? (
            <div className="py-12 text-center text-red-500">
              <h3 className="text-lg font-medium text-gray-700 py-6">暂无切片</h3>
            </div>
          ) : (
            <div className="gap-1 px-3 py-0 w-full">
              <div className="grid grid-cols-4 items-center gap-2">
                {kbfilechunks.map((chunk) => (
                  <Card key={chunk.id} className="h-70 px-0 pt-3 pb-1 gap-2 group relative">
                    <CardHeader>
                      <CardTitle className="flex justify-between items-start">
                        <div className="flex items-center gap-2">
                          <Badge className={activeMap[String(chunk.active)]}>
                            {chunk.active ? '已启用' : '未启用'}
                          </Badge>
                          {chunk.chunk_metadata?.token_count !== undefined && (
                            <Badge variant="outline" className="text-xs">
                              tokens: {chunk.chunk_metadata.token_count}
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <Switch
                            checked={chunk.active}
                            className="rounded-full transition-color"
                            onCheckedChange={() => handleActivateToggle(chunk)}
                          />
                          <button
                            className="text-black-500 hover:text-black-700 px-2"
                            onClick={() => handleEditClick(chunk)}
                          >
                            <Edit className="w-5 h-5" />
                          </button>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="bg-gray-200/10 flex-grow overflow-y-auto overflow-x-auto pr-3 p-2 pb-2">
                      <div className="whitespace-pre-wrap break-words text-sm leading-relaxed whitespace-normal pr-2">
                        {htmlRender(chunk.text)}
                      </div>
                    </CardContent>
                    <CardFooter className="shrink-0 gap-2">
                      {chunk.chunk_metadata.images_info?.map((meta, index) => (
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
                            <img src={meta.url} className="w-10 h-10" />
                          </PhotoView>
                        </PhotoProvider>
                      ))}
                    </CardFooter>
                    {/* 悬浮显示的删除按钮 */}
                    <Button
                      variant="destructive"
                      size="icon"
                      className="absolute bottom-2 right-2 w-8 h-8 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                      onClick={() => handleDeleteClick(chunk)}
                    >
                      <Trash2Icon className="w-4 h-4" />
                    </Button>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="absolute left-0 bottom-0 w-full min-h-[24px] text-sm">
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
            <Label className="block mb-2 text-sm font-medium">文本内容</Label>
            <Textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full h-40"
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

      <Dialog open={isAddOpen} onOpenChange={setIsAddOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>新建切片</DialogTitle>
            <DialogDescription>输入切片文本内容</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label className="block mb-2 text-sm font-medium">切片文本</Label>
            <Textarea
              value={newChunkText}
              onChange={(e) => setNewChunkText(e.target.value)}
              className="w-full h-40"
              placeholder="请输入切片文本内容"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsAddOpen(false);
                setNewChunkText('');
              }}
              disabled={isAdding}
            >
              取消
            </Button>
            <Button onClick={handleAddChunk} disabled={isAdding || !newChunkText.trim()}>
              {isAdding ? '保存中...' : '保存'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
