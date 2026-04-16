'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import {
  Plus,
  Trash2,
  Search,
  FileText,
  MoreHorizontal,
  Pencil,
  Clock,
  BookOpen,
} from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Badge } from '@/components/ui/badge';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';
import { formatFriendlyTime } from '@/lib/time-format';
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useRouter } from 'next/navigation';
import { HeaderPortal } from '@/components/header-portal';

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  updated_at: string;
  file_count?: number;
}

export default function KnowledgeBasePage() {
  const { t } = useI18n();

  const [knowledgebases, setKnowledgeBases] = useState(Array<KnowledgeBase>);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<KnowledgeBase | null>(null);
  const kbSizePerPage = 12;
  const router = useRouter();
  const { tenantFetch } = useTenantFetch();

  const fetchConfigs = useCallback(
    async (currentPage: number, query: string = '') => {
      try {
        setLoading(true);
        const queryParam = query ? `&query=${encodeURIComponent(query)}` : '';
        const res = await tenantFetch(
          `/api/config/knowledgebases?page=${currentPage}&size=${kbSizePerPage}${queryParam}`,
        );
        if (!res.ok) throw new Error(t('knowledgebase.fetchError'));
        const json_data = await res.json();
        const data = json_data.data.items;
        setKnowledgeBases(() =>
          (data || []).filter(
            (item: KnowledgeBase) => item.name !== 'default_attachments',
          ),
        );
        setTotalPages(json_data.data.pages || 1);
      } catch (err: any) {
        console.log(err || t('knowledgebase.loadError'));
      } finally {
        setLoading(false);
      }
    },
    [kbSizePerPage, tenantFetch, t],
  );

  useEffect(() => {
    fetchConfigs(page, searchQuery);
  }, [page, fetchConfigs]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setPage(1);
      fetchConfigs(1, searchQuery);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, fetchConfigs]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const deleteKnowledgebase = async (kb_id: string) => {
    try {
      const res = await tenantFetch(`/api/config/knowledgebases/${kb_id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error(t('knowledgebase.deleteError'));
      setKnowledgeBases((prev) => prev.filter((c) => c.id !== kb_id));
    } catch (err: any) {
      console.log('删除知识库出错: ', err);
    } finally {
      setDeleteTarget(null);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <h1 className="text-base font-semibold shrink-0">
          {t('knowledgebase.title')}
        </h1>
        <div className="ml-auto flex items-center gap-2">
          <div className="relative w-48 sm:w-64 hidden sm:block">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground w-3.5 h-3.5" />
            <Input
              type="text"
              placeholder={t('knowledgebase.searchPlaceholder')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 h-8 text-xs"
            />
          </div>
          <Button size="sm" onClick={() => router.push('/knowledgebases/create')}>
            <Plus className="w-4 h-4 mr-1" />
            {t('knowledgebase.create')}
          </Button>
        </div>
      </HeaderPortal>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-5">
          {/* Mobile-only search bar */}
          <div className="relative mb-4 sm:hidden">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground w-3.5 h-3.5" />
            <Input
              type="text"
              placeholder={t('knowledgebase.searchPlaceholder')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 h-8 text-xs"
            />
          </div>

          {loading ? (
            <div className="text-center py-16 text-sm text-muted-foreground">
              {t('common.loading')}
            </div>
          ) : knowledgebases.length === 0 ? (
            <div className="empty-state mt-8">
              <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
                <BookOpen className="w-7 h-7" />
              </div>
              <p className="text-base font-semibold mb-1">
                {t('knowledgebase.emptyTitle')}
              </p>
              <p className="text-sm text-muted-foreground text-center max-w-md mb-4">
                {t('knowledgebase.emptyMessage')}
              </p>
              <Button size="sm" onClick={() => router.push('/knowledgebases/create')}>
                <Plus className="w-4 h-4 mr-1" />
                {t('knowledgebase.create')}
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {knowledgebases.map((kb) => {
                const initial = (kb.name || 'K').charAt(0).toUpperCase();
                return (
                  <Card
                    onClick={(e) => {
                      const target = e.target as HTMLElement;
                      if (
                        target instanceof HTMLElement &&
                        target.closest('[data-stop-click]')
                      ) {
                        return;
                      }
                      router.push(`/knowledgebases/${kb.id}`);
                    }}
                    key={kb.id}
                    className="group relative cursor-pointer flex flex-col gap-0 p-4 rounded-xl border border-border bg-card card-hover-glow transition-all duration-200"
                  >
                    {/* Top: avatar + title + more menu */}
                    <div className="flex items-start gap-3">
                      <div className="model-icon type-embedding shrink-0">
                        {initial}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-semibold truncate leading-tight">
                          {kb.name}
                        </h3>
                        <p className="text-[11px] text-muted-foreground mt-0.5 truncate">
                          ID · {kb.id.slice(0, 8)}
                        </p>
                      </div>
                      <div
                        data-stop-click
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7 -mr-1.5"
                            >
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="menu-compact">
                            <DropdownMenuItem
                              onSelect={() => router.push(`/knowledgebases/${kb.id}`)}
                            >
                              <Pencil />
                              {t('common.edit')}
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onSelect={(e) => {
                                e.preventDefault();
                                setDeleteTarget(kb);
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

                    {/* Description */}
                    <CardContent className="px-0 pt-3 pb-3 flex-1">
                      <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
                        {kb.description || t('knowledgebase.noDescription')}
                      </p>
                    </CardContent>

                    {/* Footer: file count + time */}
                    <div className="flex items-center justify-between gap-2 pt-2 border-t border-border">
                      <Badge variant="outline" className="text-[11px] badge-tech h-5 px-1.5 gap-1">
                        <FileText className="w-3 h-3" />
                        {kb.file_count || 0}
                      </Badge>
                      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        <span className="truncate">{formatFriendlyTime(kb.updated_at)}</span>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {!loading && knowledgebases.length > 0 && (
        <div className="flex-none border-t border-border bg-background/60 backdrop-blur-sm py-2">
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      )}

      {/* Delete confirmation */}
      <AlertDialog
        open={!!deleteTarget}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t('knowledgebase.deleteConfirmTitle')}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t('knowledgebase.deleteConfirmMessage')}
              {deleteTarget && (
                <span className="block mt-2 font-medium text-foreground">
                  {deleteTarget.name}
                </span>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteTarget && deleteKnowledgebase(deleteTarget.id)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t('common.delete')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
