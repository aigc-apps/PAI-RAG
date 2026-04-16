'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from '@/components/ui/card';
import { Plus, Trash2, FileQuestion, Search, FileText } from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Badge } from '@/components/ui/badge';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';
import { formatFriendlyTime } from '@/lib/time-format';

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  updated_at: string;
  file_count?: number;
}

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { useRouter } from 'next/navigation';

export default function KnowledgeBasePage() {
  const { t } = useI18n();

  const [knowledgebases, setKnowledgeBases] = useState(Array<KnowledgeBase>);
  const [knowledgebasesloading, setKnowledgeBasesLoading] = useState(true);
  const [knowledgebasesrror, setKnowledgeBasesError] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const kbSizePerPage = 6;
  const router = useRouter();
  const { tenantFetch } = useTenantFetch();

  // 获取知识库列表
  const fetchConfigs = useCallback(async (currentPage: number, query: string = '') => {
    try {
      setKnowledgeBasesLoading(true);
      const queryParam = query ? `&query=${encodeURIComponent(query)}` : '';
      const res = await tenantFetch(
        `/api/config/knowledgebases?page=${currentPage}&size=${kbSizePerPage}${queryParam}`,
      );
      if (!res.ok) throw new Error(t('knowledgebase.fetchError'));
      const json_data = await res.json();
      const data = json_data.data.items;
      setKnowledgeBases(() => {
        return (data || []).filter((item: KnowledgeBase) => item.name !== 'default_attachments');
      });
      setTotalPages(json_data.data.pages);
    } catch (err: any) {
      setKnowledgeBasesError(err || t('knowledgebase.loadError'));
    } finally {
      setKnowledgeBasesLoading(false);
    }
  }, [kbSizePerPage, tenantFetch, t]);

  // 页面变化时获取数据
  useEffect(() => {
    fetchConfigs(page, searchQuery);
  }, [page, fetchConfigs]);

  // 搜索关键词变化时触发搜索（带防抖）
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setPage(1); // 搜索时重置到第一页
      fetchConfigs(1, searchQuery);
    }, 300); // 300ms 防抖

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
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        throw new Error(t('knowledgebase.deleteError'));
      }

      // 显示成功提示（可选）

      // 删除成功后更新本地状态
      setKnowledgeBases((prev) => prev.filter((config) => config.id !== kb_id));
    } catch (err: any) {console.log('删除知识库出错: ', err);}
    // 显示错误提示
  };

  return (
    <div className="flex flex-col h-screen px-6 py-0 space-y-4">
      <div className="flex justify-between items-center h-1/10 gap-4">
        <h1 className="page-title">{t('knowledgebase.title')}</h1>
        <div className="flex items-center gap-3 flex-1 max-w-md">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              type="text"
              placeholder={t('knowledgebase.searchPlaceholder')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 w-full"
            />
          </div>
          <Button
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 w-50 shrink-0"
            onClick={() => router.push('/knowledgebases/create')}
          >
            <Plus className="w-6 h-6" />
            {t('knowledgebase.create')}
          </Button>
        </div>
      </div>

      {/* 卡片容器 */}
      <div className="h-4/5">
        {knowledgebases.length > 0 ? (
          <div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
              {knowledgebases.map((base) => (
                <Card
                  onClick={(e) => {
                    // 检查是否点击了交互元素
                    const target = e.target as HTMLElement;

                    if (target instanceof HTMLElement && target.closest('button')) {
                      console.log('按钮被点击');
                      return; // 是交互元素，不触发卡片跳转
                    }

                    router.push(`/knowledgebases/${base.id}`);
                  }}
                  key={base.id}
                  className="group flex flex-col border rounded-lg shadow-sm h-full gap-0 py-0 card-hover-glow duration-300 relative"
                >
                  {/* 右上角：删除按钮（hover时显示） */}
                  <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button 
                          variant="link" 
                          className="text-muted-foreground hover:text-destructive h-6 w-6 p-0"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </AlertDialogTrigger>
                        <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>{t('knowledgebase.deleteConfirmTitle')}</AlertDialogTitle>
                          <AlertDialogDescription>
                            {t('knowledgebase.deleteConfirmMessage')}
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={(e) => deleteKnowledgebase(base.id)}
                          >
                            {t('common.delete')}
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>

                  {/* 左上偏中间位置：知识库名称 */}
                  <CardHeader className="pb-2 flex-1 pt-4">
                    <CardTitle className="text-md font-medium pb-2 pt-1">
                      {base.name}
                    </CardTitle>
                    
                    {/* 描述 */}
                    <div className="px-0 pt-0 pb-0">
                      <p className="text-xs text-muted-foreground line-clamp-2">
                        {base.description
                          ? base.description
                          : t('knowledgebase.noDescription')}
                      </p>
                    </div>
                  </CardHeader>

                  {/* 右下角：文档数量和更新时间 Badge */}
                  <CardFooter className="px-3 pt-2 pb-3 flex justify-end items-center gap-2 mt-auto">
                    {/* 文档数量 Badge */}
                    <Badge 
                      variant="outline" 
                      className="text-xs badge-tech"
                    >
                      <FileText className="w-3 h-3" />
                      {base.file_count || 0}
                    </Badge>
                    
                    {/* 更新时间 Badge */}
                    <Badge 
                      variant="outline" 
                      className="text-xs badge-tech"
                    >
                      {formatFriendlyTime(base.updated_at)}
                    </Badge>
                  </CardFooter>
                </Card>
              ))}
            </div>
            <div className="flex justify-center items-center h-1/10 pt-6">
              <PaginationComponent
                currentPage={page}
                totalPages={totalPages}
                onPageChange={handlePageChange}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-muted-foreground mb-4">
              <FileQuestion className="w-16 h-16 mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">{t('knowledgebase.emptyTitle')}</h3>
              <p className="text-sm mb-4">
                {t('knowledgebase.emptyMessage')}
              </p>
            </div>
            <Button 
              onClick={() => router.push('/knowledgebases/create')}
              className="gap-2"
            >
              <Plus className="w-4 h-4" />
              {t('knowledgebase.create')}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
