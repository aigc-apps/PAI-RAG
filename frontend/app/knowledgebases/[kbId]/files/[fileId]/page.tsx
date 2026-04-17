'use client';
import React, { useState, useEffect, use } from 'react';
import { Button } from '@/components/ui/button';
import {
  Plus,
  Pencil,
  Trash2,
  MoreHorizontal,
  Hash,
  FileText,
  Save,
} from 'lucide-react';
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
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { PhotoProvider, PhotoView } from 'react-photo-view';
import 'react-photo-view/dist/react-photo-view.css';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from '@/components/ui/dialog';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Label } from '@/components/ui/label';
import { useRouter } from 'next/navigation';
import { htmlRender } from '@/app/knowledgebases/[kbId]/viewer/htmlRender';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { HeaderPortal } from '@/components/header-portal';
import { PageLoading, Loading, Spinner } from '@/components/ui/loading';

interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  chunk_config: {
    parser_type: string;
    separator: string;
    chunk_size: string;
    chunk_overlap: string;
  };
  embedding_model: string;
  retrieval_config: {
    retrieval_mode: string;
    top_k: number;
    similarity_threshold: string;
    enable_rerank: boolean;
    rerank_model: string;
    vector_weight?: string;
  };
}

interface KnowledgeBaseFile {
  id: string;
  file_name: string;
  file_size: string;
  file_extension: string;
  file_metadata: { file_url: string };
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

export default function KnowledgeBaseFileChunksPage({
  params,
}: {
  params: Promise<{ kbId: string; fileId: string }>;
}) {
  const { t } = useI18n();
  const { kbId, fileId } = use(params);
  const [knowledgebase, setKnowledgeBase] = useState<KnowledgeBase>();
  const [kbfile, setKbFile] = useState<KnowledgeBaseFile>();
  const [kbfilechunks, setKbFileChunks] = useState(Array<KbFileChunk>);
  const [kbfilechunksloading, setKbFileChunksLoading] = useState(true);

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const chunksSizePerPage = 12;

  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editText, setEditText] = useState('');
  const [selectedChunk, setSelectedChunk] = useState<KbFileChunk | null>(null);

  const [isAddOpen, setIsAddOpen] = useState(false);
  const [newChunkText, setNewChunkText] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<KbFileChunk | null>(null);

  const { tenantFetch } = useTenantFetch();
  const router = useRouter();

  const reloadChunks = async (nextPage = page) => {
    const res = await tenantFetch(
      `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks?page=${nextPage}&size=${chunksSizePerPage}`,
    );
    if (res.ok) {
      const json = await res.json();
      setKbFileChunks(json.data.items || []);
      setTotalPages(json.data.pages);
    }
  };

  // Belt-and-suspenders: Radix sometimes leaves <body style="pointer-events:none">
  // when a Dialog is opened from a DropdownMenuItem. Whenever every modal is
  // closed, reset the body styles so the page stays interactive.
  useEffect(() => {
    const anyOpen = isEditOpen || isAddOpen || !!deleteTarget;
    if (!anyOpen) {
      const id = setTimeout(() => {
        document.body.style.pointerEvents = '';
      }, 300);
      return () => clearTimeout(id);
    }
  }, [isEditOpen, isAddOpen, deleteTarget]);

  useEffect(() => {
    const run = async () => {
      try {
        const [kbRes, fileRes, chunksRes] = await Promise.all([
          tenantFetch(`/api/config/knowledgebases/${kbId}`),
          tenantFetch(`/api/config/knowledgebases/${kbId}/files/${fileId}`),
          tenantFetch(
            `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks?page=${page}&size=${chunksSizePerPage}`,
          ),
        ]);
        if (kbRes.ok) setKnowledgeBase((await kbRes.json()).data);
        if (fileRes.ok) setKbFile((await fileRes.json()).data);
        if (chunksRes.ok) {
          const json = await chunksRes.json();
          setKbFileChunks(json.data.items || []);
          setTotalPages(json.data.pages);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setKbFileChunksLoading(false);
      }
    };
    run();
  }, [page, fileId, kbId, tenantFetch]);

  if (!knowledgebase || !kbfile) {
    return <PageLoading className="h-full" />;
  }

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const handleActivateToggle = async (chunk: KbFileChunk) => {
    const next = { ...chunk, active: !chunk.active };
    const res = await tenantFetch(
      `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks/${chunk.id}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next),
      },
    );
    if (!res.ok) return;
    setKbFileChunks((prev) =>
      prev.map((c) => (c.id === chunk.id ? next : c)),
    );
  };

  const handleEditClick = (chunk: KbFileChunk) => {
    setSelectedChunk(chunk);
    setEditText(chunk.text);
    setIsEditOpen(true);
  };

  const handleSaveEdit = async () => {
    if (!selectedChunk) return;
    const updated = { ...selectedChunk, text: editText };
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks/${selectedChunk.id}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updated),
        },
      );
      if (!res.ok) throw new Error(t('knowledgebase.updateFailed'));
      setKbFileChunks((prev) =>
        prev.map((c) => (c.id === selectedChunk.id ? updated : c)),
      );
      setIsEditOpen(false);
    } catch (err) {
      console.error(err);
    }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks/${deleteTarget.id}`,
        { method: 'DELETE' },
      );
      if (!res.ok) throw new Error(t('knowledgebase.deleteChunkFailed'));
      setKbFileChunks((prev) => prev.filter((c) => c.id !== deleteTarget.id));
      if (kbfilechunks.length === 1 && page > 1) {
        setPage(page - 1);
      } else {
        await reloadChunks();
      }
    } catch (err: any) {
      console.error(err);
    } finally {
      setDeleteTarget(null);
    }
  };

  const handleAddChunk = async () => {
    if (!newChunkText.trim()) return;
    setIsAdding(true);
    try {
      const res = await tenantFetch(
        `/api/config/knowledgebases/${kbId}/files/${fileId}/chunks`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: newChunkText, chunk_metadata: {} }),
        },
      );
      if (!res.ok) throw new Error(t('knowledgebase.addChunkFailed'));
      setNewChunkText('');
      setIsAddOpen(false);
      await reloadChunks();
    } catch (err) {
      console.error(err);
    } finally {
      setIsAdding(false);
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
              <BreadcrumbLink asChild>
                <Button
                  variant="link"
                  className="px-0 h-auto"
                  onClick={() => router.push(`/knowledgebases/${knowledgebase.id}`)}
                >
                  {knowledgebase.name}
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage className="font-semibold flex items-center gap-1">
                <FileText className="w-3.5 h-3.5 text-primary" />
                {kbfile.file_name}
              </BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        {kbfilechunks.length > 0 && (
          <Badge variant="secondary" className="text-[10px] font-mono bg-muted text-muted-foreground">
            {kbfilechunks.length} / page
          </Badge>
        )}
        <div className="ml-auto">
          <Button size="sm" onClick={() => setIsAddOpen(true)}>
            <Plus className="w-4 h-4 mr-1" />
            {t('knowledgebase.newChunk')}
          </Button>
        </div>
      </HeaderPortal>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-6">
          {kbfilechunksloading ? (
            <PageLoading />
          ) : kbfilechunks.length === 0 ? (
            <div className="empty-state mt-8">
              <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
                <FileText className="w-7 h-7" />
              </div>
              <p className="text-base font-semibold mb-1">{t('knowledgebase.noChunks')}</p>
              <Button size="sm" className="mt-3" onClick={() => setIsAddOpen(true)}>
                <Plus className="w-4 h-4 mr-1" />
                {t('knowledgebase.newChunk')}
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {kbfilechunks.map((chunk, idx) => (
                <ChunkCard
                  key={chunk.id}
                  chunk={chunk}
                  index={(page - 1) * chunksSizePerPage + idx + 1}
                  onToggle={() => handleActivateToggle(chunk)}
                  onEdit={() => handleEditClick(chunk)}
                  onDelete={() => setDeleteTarget(chunk)}
                  t={t}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {!kbfilechunksloading && kbfilechunks.length > 0 && (
        <div className="flex-none border-t border-border bg-background/60 backdrop-blur-sm py-2">
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      )}

      {/* Edit */}
      <Dialog
        open={isEditOpen}
        onOpenChange={(open) => {
          setIsEditOpen(open);
          if (!open) {
            // Hard-reset in case Radix leaves <body> with pointer-events:none
            setTimeout(() => {
              document.body.style.pointerEvents = '';
            }, 0);
          }
        }}
      >
        <DialogContent
          className="sm:max-w-2xl gap-0 p-0 overflow-hidden"
          onCloseAutoFocus={(e) => e.preventDefault()}
        >
          <DialogHeader className="px-5 pt-5 pb-3 border-b border-border">
            <DialogTitle className="flex items-center gap-2 text-sm font-semibold">
              <Pencil className="w-4 h-4 text-primary" />
              {t('knowledgebase.editChunk')}
            </DialogTitle>
            <DialogDescription className="text-xs mt-0.5">
              {t('knowledgebase.editAndSave')}
            </DialogDescription>
          </DialogHeader>
          <div className="px-5 py-4 space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                {t('knowledgebase.textContent')}
              </Label>
              <span className="text-[11px] text-muted-foreground tabular-nums">
                {editText.length}
              </span>
            </div>
            <Textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full h-56 text-xs font-mono leading-relaxed bg-muted/30 resize-none"
              placeholder={t('knowledgebase.enterNewContent')}
            />
          </div>
          <DialogFooter className="px-5 py-3 border-t border-border bg-muted/20">
            <DialogClose asChild>
              <Button variant="outline" size="sm">
                {t('common.cancel')}
              </Button>
            </DialogClose>
            <Button
              size="sm"
              onClick={handleSaveEdit}
              disabled={!editText.trim() || editText === selectedChunk?.text}
            >
              <Save className="w-3 h-3 mr-1" />
              {t('common.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add */}
      <Dialog
        open={isAddOpen}
        onOpenChange={(open) => {
          setIsAddOpen(open);
          if (!open) {
            setNewChunkText('');
            setTimeout(() => {
              document.body.style.pointerEvents = '';
            }, 0);
          }
        }}
      >
        <DialogContent
          className="sm:max-w-2xl gap-0 p-0 overflow-hidden"
          onCloseAutoFocus={(e) => e.preventDefault()}
        >
          <DialogHeader className="px-5 pt-5 pb-3 border-b border-border">
            <DialogTitle className="flex items-center gap-2 text-sm font-semibold">
              <Plus className="w-4 h-4 text-primary" />
              {t('knowledgebase.newChunk')}
            </DialogTitle>
            <DialogDescription className="text-xs mt-0.5">
              {t('knowledgebase.enterChunkContent')}
            </DialogDescription>
          </DialogHeader>
          <div className="px-5 py-4 space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                {t('knowledgebase.chunkText')}
              </Label>
              <span className="text-[11px] text-muted-foreground tabular-nums">
                {newChunkText.length}
              </span>
            </div>
            <Textarea
              value={newChunkText}
              onChange={(e) => setNewChunkText(e.target.value)}
              className="w-full h-56 text-xs font-mono leading-relaxed bg-muted/30 resize-none"
              placeholder={t('knowledgebase.enterChunkText')}
            />
          </div>
          <DialogFooter className="px-5 py-3 border-t border-border bg-muted/20">
            <DialogClose asChild>
              <Button variant="outline" size="sm" disabled={isAdding}>
                {t('common.cancel')}
              </Button>
            </DialogClose>
            <Button
              size="sm"
              onClick={handleAddChunk}
              disabled={isAdding || !newChunkText.trim()}
            >
              {isAdding ? (
                <>
                  <Spinner size="sm" className="mr-1" />
                  {t('common.saving')}
                </>
              ) : (
                <>
                  <Save className="w-3 h-3 mr-1" />
                  {t('common.save')}
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirm */}
      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
        title={t('knowledgebase.confirmDeleteChunk')}
        description={t('knowledgebase.deleteChunkHint') || undefined}
        onConfirm={confirmDelete}
      >
        {deleteTarget?.text && (
          <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-[11px] font-mono text-foreground/80 leading-relaxed max-h-[80px] overflow-hidden line-clamp-3">
            {deleteTarget.text}
          </div>
        )}
      </ConfirmDialog>
    </div>
  );
}

// ===== Chunk card =====

interface ChunkCardProps {
  chunk: KbFileChunk;
  index: number;
  onToggle: () => void;
  onEdit: () => void;
  onDelete: () => void;
  t: (k: string) => string;
}

function ChunkCard({ chunk, index, onToggle, onEdit, onDelete, t }: ChunkCardProps) {
  const active = chunk.active;

  return (
    <div
      className={`group relative flex flex-col rounded-lg border transition-colors overflow-hidden ${
        active ? 'border-border bg-card' : 'border-border/60 bg-muted/30 opacity-70'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-border/60 bg-background/40">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-[10px] font-mono text-muted-foreground shrink-0">
            #{String(index).padStart(2, '0')}
          </span>
          {chunk.chunk_metadata?.token_count !== undefined && (
            <Badge
              variant="outline"
              className="h-5 px-1.5 text-[10px] font-mono gap-0.5"
            >
              <Hash className="w-2.5 h-2.5" />
              {chunk.chunk_metadata.token_count}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0" data-stop-click>
          <Switch
            checked={active}
            onCheckedChange={onToggle}
            className="scale-75 -mr-1"
          />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <MoreHorizontal className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="menu-compact">
              <DropdownMenuItem
                onSelect={(e) => {
                  e.preventDefault();
                  // Let DropdownMenu finish its close/focus cleanup before
                  // mounting the Dialog; otherwise Radix leaves
                  // pointer-events: none stuck on <body>.
                  setTimeout(onEdit, 0);
                }}
              >
                <Pencil />
                {t('common.edit')}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onSelect={(e) => {
                  e.preventDefault();
                  setTimeout(onDelete, 0);
                }}
                className="text-destructive focus:text-destructive"
              >
                <Trash2 />
                {t('common.delete')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-y-auto px-3 py-2 text-xs leading-relaxed text-foreground/90 max-h-[220px]">
        <div className="whitespace-pre-wrap break-words">
          {htmlRender(chunk.text)}
        </div>
      </div>

      {/* Images footer (if any) */}
      {chunk.chunk_metadata.images_info?.length > 0 && (
        <div className="flex gap-1 px-3 py-1.5 border-t border-border/60 bg-muted/20 flex-wrap">
          {chunk.chunk_metadata.images_info.map((meta, i) => (
            <PhotoProvider
              key={i}
              maskOpacity={0.8}
              overlayRender={() => (
                <div className="absolute left-0 bottom-0 p-4 w-full min-h-30 text-sm text-slate-300 z-50 bg-black/50">
                  {t('knowledgebase.imageDesc')}：{meta.desc}
                </div>
              )}
            >
              <PhotoView src={meta.url}>
                <img
                  src={meta.url}
                  className="w-7 h-7 rounded object-cover cursor-pointer"
                />
              </PhotoView>
            </PhotoProvider>
          ))}
        </div>
      )}
    </div>
  );
}
