'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect, FC } from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from '@/components/ui/card';
import { Plus, Trash2 } from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { formatBeijingTime } from '../knowledgebases/utils/utils';
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
import { Chatbot } from './chatbot_config';
import { useRouter } from 'next/navigation';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

const ChatbotPage = () => {
  const { t } = useI18n();
  const [chatbots, setChatbots] = useState(Array<Chatbot>);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const pageSize = 6;
  const router = useRouter();
  const { tenantFetch } = useTenantFetch();
  
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/apps?page=${page}&size=${pageSize}`,
        );
        if (!res.ok) throw new Error(t('apps.fetchError'));
        const json_data = await res.json();
        const data = json_data.data.items;
        setChatbots(data || []); // 更新状态
        setTotalPages(json_data.data.pages);
      } catch (err: unknown) {
        console.log(err || '加载失败');
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
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        throw new Error(t('apps.deleteError'));
      }

      // 显示成功提示（可选）

      // 删除成功后更新本地状态
      setChatbots((prev) => prev.filter((bot) => bot.id !== bot_id));
    } catch (err: unknown) {
      console.log('删除Chatbot失败。', err);
    }
    // 显示错误提示
  };

  return (
    <div className="flex flex-col h-screen px-6 space-y-2">
      <div className="flex justify-between items-center h-1/10">
        <h1 className="text-xl font-medium">{t('apps.title')}</h1>
        <Button
          className="px-4 py-2 bg-primary rounded-md text-sm font-medium hover:bg-primary/90 w-40"
          onClick={()=>{router.push('/apps/create')}}
        >
          <Plus className="w-6 h-6" />
          {t('apps.create')}
        </Button>
      </div>
      <div className="text-sm text-muted-foreground pb-2">
        {t('apps.subtitle')}
      </div>

      {/* 卡片容器 */}
      <div className="h-4/5">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
          {chatbots.map((bot) => (
            <Card
              onClick={(e) => {
                // 检查是否点击了交互元素
                const target = e.target as HTMLElement;

                if (target instanceof HTMLElement && target.closest('button')) {
                  console.log('按钮被点击');
                  return; // 是交互元素，不触发卡片跳转
                }
                router.push(`/apps/${bot.app_id}`);
              }}
              key={bot.id}
              className="flex flex-col border rounded-lg shadow-sm h-full gap-0 py-0 transition-shadow hover:shadow-md hover:bg-muted/50 duration-300"
            >
              <CardHeader>
                <CardTitle className="text-md flex pt-4 pb-1">
                  {bot.app_id}
                </CardTitle>
              </CardHeader>

              <CardContent className="pt-0 pb-0">
                <p className="text-xs text-muted-foreground line-clamp-1">
                  {bot.description
                    ? bot.description
                    : t('knowledgebase.noDescription')}
                </p>
              </CardContent>
              <CardFooter className="px-3 pt-0 flex justify-between w-full py-0">
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="link" className="text-muted-foreground">
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>{t('apps.deleteConfirmTitle')}</AlertDialogTitle>
                      <AlertDialogDescription>
                        {t('apps.deleteConfirmMessage')}
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={() => deleteChatbot(bot.id)}
                      >
                        {t('common.delete')}
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>

                <div className="text-xs text-muted-foreground line-clamp-1 truncate">
                  {formatBeijingTime(bot.updated_at)}
                </div>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
      {/* 分页组件 */}
      <div className="flex justify-center items-center h-1/10">
        <PaginationComponent
          currentPage={page}
          totalPages={totalPages}
          onPageChange={handlePageChange}
        />
      </div>
    </div>
  );
}

export default ChatbotPage;