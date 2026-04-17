'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
} from '@/components/ui/card';
import { Plus, Trash2, MoreHorizontal, Pencil, Clock, AppWindow } from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { formatBeijingTime } from '../knowledgebases/utils/utils';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Chatbot } from './chatbot_config';
import { useRouter } from 'next/navigation';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';
import { HeaderPortal } from '@/components/header-portal';
import { PageLoading } from '@/components/ui/loading';

const ChatbotPage = () => {
  const { t } = useI18n();
  const [chatbots, setChatbots] = useState(Array<Chatbot>);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<Chatbot | null>(null);
  const pageSize = 12;
  const router = useRouter();
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        setLoading(true);
        const res = await tenantFetch(
          `/api/config/apps?page=${page}&size=${pageSize}`,
        );
        if (!res.ok) throw new Error(t('apps.fetchError'));
        const json_data = await res.json();
        const data = json_data.data.items;
        setChatbots(data || []);
        setTotalPages(json_data.data.pages || 1);
      } catch (err: unknown) {
        console.log(err || '加载失败');
      } finally {
        setLoading(false);
      }
    };

    fetchConfigs();
  }, [page, tenantFetch, t]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const deleteChatbot = async (bot_id: string) => {
    try {
      const res = await tenantFetch(`/api/config/apps/${bot_id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) throw new Error(t('apps.deleteError'));

      setChatbots((prev) => prev.filter((bot) => bot.id !== bot_id));
    } catch (err: unknown) {
      console.log('删除Chatbot失败。', err);
    } finally {
      setDeleteTarget(null);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <div className="flex items-center gap-2">
          <h1 className="text-base font-semibold">{t('apps.title')}</h1>
          <span className="text-xs text-muted-foreground hidden md:inline">
            · {t('apps.subtitle')}
          </span>
        </div>
        <div className="ml-auto">
          <Button
            size="sm"
            onClick={() => router.push('/apps/create')}
          >
            <Plus className="w-4 h-4 mr-1" />
            {t('apps.create')}
          </Button>
        </div>
      </HeaderPortal>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-6">
          {loading ? (
            <PageLoading />
          ) : chatbots.length === 0 ? (
            <div className="empty-state mt-8">
              <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
                <AppWindow className="w-7 h-7" />
              </div>
              <p className="text-base font-semibold mb-1">{t('apps.emptyTitle')}</p>
              <p className="text-sm text-muted-foreground text-center max-w-md mb-4">
                {t('apps.emptyMessage')}
              </p>
              <Button size="sm" onClick={() => router.push('/apps/create')}>
                <Plus className="w-4 h-4 mr-1" />
                {t('apps.create')}
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {chatbots.map((bot) => {
                const initial = (bot.app_id || 'A').charAt(0).toUpperCase();
                return (
                  <Card
                    onClick={(e) => {
                      const target = e.target as HTMLElement;
                      if (target instanceof HTMLElement && target.closest('[data-stop-click]')) {
                        return;
                      }
                      router.push(`/apps/${bot.app_id}`);
                    }}
                    key={bot.id}
                    className="group relative cursor-pointer flex flex-col gap-0 p-5 rounded-xl border border-border bg-card card-hover-glow transition-all duration-200"
                  >
                    {/* Top: avatar + title + more menu */}
                    <div className="flex items-start gap-3">
                      <div className="model-icon type-llm shrink-0">
                        {initial}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-semibold truncate leading-tight">
                          {bot.app_id}
                        </h3>
                        <p className="text-[11px] text-muted-foreground mt-0.5 truncate">
                          ID · {bot.id.slice(0, 8)}
                        </p>
                      </div>
                      <div
                        data-stop-click
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => e.stopPropagation()}
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
                              onSelect={() => router.push(`/apps/${bot.app_id}`)}
                            >
                              <Pencil />
                              {t('common.edit')}
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onSelect={(e) => {
                                e.preventDefault();
                                // Defer opening the dialog so DropdownMenu
                                // finishes its close/focus cleanup first —
                                // otherwise Radix leaves body.style.pointerEvents
                                // stuck at 'none' and the page freezes.
                                setTimeout(() => setDeleteTarget(bot), 0);
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
                    <CardContent className="px-0 pt-4 pb-4 flex-1">
                      <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
                        {bot.description || t('knowledgebase.noDescription')}
                      </p>
                    </CardContent>

                    {/* Footer */}
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span className="truncate">{formatBeijingTime(bot.updated_at)}</span>
                    </div>
                  </Card>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {!loading && chatbots.length > 0 && (
        <div className="flex-none border-t border-border bg-background/60 backdrop-blur-sm py-2">
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      )}

      {/* Delete confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
        title={t('apps.deleteConfirmTitle')}
        description={t('apps.deleteConfirmMessage')}
        target={deleteTarget ? { label: 'App', value: deleteTarget.app_id } : undefined}
        onConfirm={() => {
          if (deleteTarget) deleteChatbot(deleteTarget.id);
        }}
      />
    </div>
  );
};

export default ChatbotPage;
