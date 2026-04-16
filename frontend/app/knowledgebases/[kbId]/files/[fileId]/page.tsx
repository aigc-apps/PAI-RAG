'use client';
import React, { useState, useEffect, use } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Edit, Plus, Trash2Icon } from 'lucide-react';
import { useI18n } from '@/app/providers/i18n';
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
    parser_type: string; // Chunking type
    separator: string; // Chunking separator
    chunk_size: string; // Chunk size
    chunk_overlap: string; // Chunk overlap size
  };
  embedding_model: string; // Embedding model name
  retrieval_config: {
    retrieval_mode: string; // Index type: vector, fulltext, hybrid
    top_k: number; // Top-K value
    similarity_threshold: string; // Similarity score threshold
    enable_rerank: boolean;
    rerank_model: string; // Rerank model name
    vector_weight?: string; // Vector retrieval weight (for hybrid only)
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

// Status mapping
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
  const { t } = useI18n();

  const {kbId, fileId} = use(params);
  const [knowledgebase, setKnowledgeBase] = useState<KnowledgeBase>(); // Knowledgebase details
  const [knowledgebaseloading, setKnowledgeBaseLoading] = useState(true); // Knowledgebase loading state
  const [knowledgebaseerror, setKnowledgeBaseError] = useState(''); // Knowledgebase error message

  const [kbfile, setKbFile] = useState<KnowledgeBaseFile>(); // File details
  const [kbfileloading, setKbFileLoading] = useState(true); // File loading state
  const [kbfileerror, setKbFileError] = useState(''); // File error message

  const [kbfilechunks, setKbFileChunks] = useState(Array<KbFileChunk>); // File chunks list details
  const [kbfilechunksloading, setKbFileChunksLoading] = useState(true); // File loading state
  const [kbfilechunkserror, setKbFilChunksError] = useState(''); // File error message

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
        if (!res.ok) throw new Error(t('knowledgebase.fetchKbFailed'));
        const json_data = await res.json();
        const kb_data = json_data.data;

        setKnowledgeBase(kb_data);
        console.log(t('knowledgebase.kbDetails'), kb_data);
      } catch (err: any) {
        setKnowledgeBaseError(err || t('knowledgebase.loadError'));
      } finally {
        setKnowledgeBaseLoading(false);
      }
    };
    const fetchKbFile = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/knowledgebases/${kbId}/files/${fileId}`,
        );
        if (!res.ok) throw new Error(t('knowledgebase.fetchKbFileFailed'));
        const json_data = await res.json();
        const kb_file_data = json_data.data;

        setKbFile(kb_file_data);
        console.log(t('knowledgebase.kbFileDetails'), kb_file_data);
      } catch (err: any) {
        setKbFileError(err || t('knowledgebase.loadError'));
      } finally {
        setKbFileLoading(false);
      }
    };

    const fetchKbFileChunks = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks?page=${page}&size=${chunksSizePerPage}`,
        );
        if (!res.ok) throw new Error(t('knowledgebase.fetchChunksFailed'));
        const json_data = await res.json();
        const kb_file_chunks_data = json_data.data.items;
        setTotalPages(json_data.data.pages);
        setKbFileChunks(kb_file_chunks_data || []);
        console.log(t('knowledgebase.chunksDetails'), kb_file_chunks_data);
      } catch (err: any) {
        setKbFilChunksError(err || t('knowledgebase.loadError'));
      } finally {
        setKbFileChunksLoading(false);
      }
    };
    fetchKbConfigs();
    fetchKbFile();
    fetchKbFileChunks();
  }, [page, fileId, kbId]);
  if (!knowledgebase || !kbfile) {
    return <div className="p-6">{t('common.loading')}</div>;
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
      body: JSON.stringify(chunk),
    });

    if (!res.ok) throw new Error(t('knowledgebase.modifyConfigFailed', { id: chunk.id }));
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
    if (!confirm(t('knowledgebase.confirmDeleteChunk'))) {
      return;
    }

    try {
      const url = `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks/${chunk.id}`;
      const response = await tenantFetch(url, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || t('knowledgebase.deleteChunkFailed'));
      }

      // Remove from local state
      setKbFileChunks((prev) => prev.filter((c) => c.id !== chunk.id));

      // If current page has no data, go back to previous page
      if (kbfilechunks.length === 1 && page > 1) {
        setPage(page - 1);
      } else {
        // Refresh chunks list
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
      console.error(t('knowledgebase.deleteChunkFailed'), err);
      alert(err.message || t('knowledgebase.deleteChunkFailed'));
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

      if (!response.ok) throw new Error(t('knowledgebase.updateFailed'));

      // Update local state
      setKbFileChunks((prev) =>
        prev.map((c) =>
          c.id === selectedChunk.id ? { ...c, text: selectedChunk.text } : c,
        ),
      );
      setIsEditOpen(false);
    } catch (err) {
      console.error(t('knowledgebase.editFailed'), err);
      // Optional: add error notification (e.g., toast)
    }
  };

  const handleAddChunk = async () => {
    if (!newChunkText.trim()) {
      alert(t('knowledgebase.inputChunkText'));
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
        throw new Error(errorData.message || t('knowledgebase.addChunkFailed'));
      }

      // Reset state
      setNewChunkText('');
      setIsAddOpen(false);

      // Refresh chunks list
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks?page=${page}&size=${chunksSizePerPage}`,
      );
      if (res.ok) {
        const json_data = await res.json();
        setKbFileChunks(json_data.data.items || []);
        setTotalPages(json_data.data.pages);
      }
    } catch (err: any) {
      console.error(t('knowledgebase.addChunkFailed'), err);
      alert(err.message || t('knowledgebase.addChunkFailed'));
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-full">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="flex items-center gap-2">
            {/* Breadcrumb navigation */}
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() => router.push('/knowledgebases')}
                    >
                      {t('knowledgebase.title')}
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
              <h1 className="text-xl font-medium pl-2">{t('knowledgebase.fileChunksList')}</h1>
            </div>
            <Button
              variant="default"
              className="h-8"
              onClick={() => setIsAddOpen(true)}
            >
              <Plus className="w-4 h-4 mr-1" />
              {t('knowledgebase.newChunk')}
            </Button>
          </div>
        </div>
      </div>
      {/* Scrollable content area */}
      <div className="overflow-y-auto h-4/5">
        <div className="flex border-dashed border-gray-200 rounded-xl p-0">
          {kbfilechunksloading ? (
            <div className="py-12 text-center">
              <p className="text-gray-500">{t('common.loading')}</p>
            </div>
          ) : kbfilechunkserror ? (
            <div className="py-12 text-center text-red-500">
              <p>{t('knowledgebase.chunksLoadFailed')}</p>
            </div>
          ) : kbfilechunks.length === 0 ? (
            <div className="py-12 text-center text-red-500">
              <h3 className="text-lg font-medium text-gray-700 py-6">{t('knowledgebase.noChunks')}</h3>
            </div>
          ) : (
            <div className="gap-1 px-3 py-0 w-full">
              <div className="grid grid-cols-4 items-center gap-2">
                {kbfilechunks.map((chunk) => (
                  <Card key={chunk.id} className="h-70 px-0 pt-3 pb-1 gap-2 group relative chunk-card">
                    <CardHeader>
                      <CardTitle className="flex justify-between items-start">
                        <div className="flex items-center gap-2">
                          <Badge className={activeMap[String(chunk.active)]}>
                            {chunk.active ? t('knowledgebase.enabled') : t('knowledgebase.disabled')}
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
                                <div>{t('knowledgebase.imageDesc')}：{meta.desc}</div>
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
                    {/* Hover delete button */}
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
            <DialogTitle>{t('knowledgebase.editChunk')}</DialogTitle>
            <DialogDescription>{t('knowledgebase.editAndSave')}</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label className="block mb-2 text-sm font-medium">{t('knowledgebase.textContent')}</Label>
            <Textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full h-40"
              placeholder={t('knowledgebase.enterNewContent')}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleSaveEdit}>{t('common.save')}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isAddOpen} onOpenChange={setIsAddOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t('knowledgebase.newChunk')}</DialogTitle>
            <DialogDescription>{t('knowledgebase.enterChunkContent')}</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label className="block mb-2 text-sm font-medium">{t('knowledgebase.chunkText')}</Label>
            <Textarea
              value={newChunkText}
              onChange={(e) => setNewChunkText(e.target.value)}
              className="w-full h-40"
              placeholder={t('knowledgebase.enterChunkText')}
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
              {t('common.cancel')}
            </Button>
            <Button onClick={handleAddChunk} disabled={isAdding || !newChunkText.trim()}>
              {isAdding ? t('common.saving') : t('common.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
