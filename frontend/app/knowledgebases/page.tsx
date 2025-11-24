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
import { formatBeijingTime } from './utils/utils';
import { Badge } from '@/components/ui/badge';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import utc from 'dayjs/plugin/utc';
import 'dayjs/locale/zh-cn';

dayjs.extend(relativeTime);
dayjs.extend(utc);
dayjs.locale('zh-cn');

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

// 友好时间显示函数（使用 UTC 时间进行比较）
function formatFriendlyTime(utcTime: string): string {
  const now = dayjs.utc(); // 使用 UTC 时间作为当前时间
  const time = dayjs.utc(utcTime); // 将传入的时间解析为 UTC 时间
  const diffMinutes = now.diff(time, 'minute');
  const diffHours = now.diff(time, 'hour');
  const diffDays = now.diff(time, 'day');
  const diffMonths = now.diff(time, 'month');

  if (diffMinutes < 1) {
    return '刚刚';
  } else if (diffDays < 1) {
    // 今天内：显示小时或分钟
    if (diffHours < 1) {
      return `${diffMinutes}分钟前`;
    } else {
      return `${diffHours}小时前`;
    }
  } else if (diffDays < 30) {
    return `${diffDays}天前`;
  } else if (diffMonths < 1) {
    return '一个月前';
  } else if (diffMonths < 6) {
    return `${diffMonths}个月前`;
  } else {
    return '半年前';
  }
}

export default function KnowledgeBasePage() {
  const [knowledgebases, setKnowledgeBases] = useState(Array<KnowledgeBase>); // 知识库列表
  const [knowledgebasesloading, setKnowledgeBasesLoading] = useState(true); // 加载状态
  const [knowledgebasesrror, setKnowledgeBasesError] = useState(''); // 错误信息
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState(''); // 搜索关键词
  const kbSizePerPage = 6;
  const router = useRouter();

  // 获取知识库列表
  const fetchConfigs = useCallback(async (currentPage: number, query: string = '') => {
    try {
      setKnowledgeBasesLoading(true);
      const queryParam = query ? `&query=${encodeURIComponent(query)}` : '';
      const res = await fetch(
        `/api/config/knowledgebases?page=${currentPage}&size=${kbSizePerPage}${queryParam}`,
      );
      if (!res.ok) throw new Error('获取知识库列表失败');
      const json_data = await res.json();
      const data = json_data.data.items;
      setKnowledgeBases(() => {
        return (data || []).filter((item: KnowledgeBase) => item.name !== 'default_attachments');
      });
      setTotalPages(json_data.data.pages);
    } catch (err: any) {
      setKnowledgeBasesError(err || '加载失败');
    } finally {
      setKnowledgeBasesLoading(false);
    }
  }, [kbSizePerPage]);

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
      const res = await fetch(`/api/config/knowledgebases/${kb_id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        throw new Error('删除失败，请检查网络或配置');
      }

      // 显示成功提示（可选）

      // 删除成功后更新本地状态
      setKnowledgeBases((prev) => prev.filter((config) => config.id !== kb_id));
    } catch (err: any) {console.log('删除知识库出错: ', err);}
    // 显示错误提示
  };

  return (
    <div className="flex flex-col h-screen px-6 py-0 space-y-4">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center h-1/10 gap-4">
        <h1 className="text-xl font-medium">知识库</h1>
        <div className="flex items-center gap-3 flex-1 max-w-md">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              type="text"
              placeholder="搜索知识库（名称、描述、ID）"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 w-full"
            />
          </div>
          <Button
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 w-40 shrink-0"
            onClick={() => router.push('/knowledgebases/create')}
          >
            <Plus className="w-6 h-6" />
            新建知识库
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
                  className="group flex flex-col border rounded-lg shadow-sm h-full gap-0 py-0 transition-shadow hover:shadow-md hover:bg-muted/50 duration-300 relative"
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
                          <AlertDialogTitle>是否确认删除?</AlertDialogTitle>
                          <AlertDialogDescription>
                            请注意，删除知识库无法撤销。请仔细核对之后再确认。
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>取消</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={(e) => deleteKnowledgebase(base.id)}
                          >
                            删除
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
                          : '暂时还没有描述，可以去设置页面添加哦。'}
                      </p>
                    </div>
                  </CardHeader>

                  {/* 右下角：文档数量和更新时间 Badge */}
                  <CardFooter className="px-3 pt-2 pb-3 flex justify-end items-center gap-2 mt-auto">
                    {/* 文档数量 Badge */}
                    <Badge 
                      variant="outline" 
                      className="text-xs bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 dark:border-blue-400/30"
                    >
                      <FileText className="w-3 h-3" />
                      {base.file_count || 0}
                    </Badge>
                    
                    {/* 更新时间 Badge */}
                    <Badge 
                      variant="outline" 
                      className="text-xs bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 dark:border-blue-400/30"
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
          // 空状态提示
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-muted-foreground mb-4">
              <FileQuestion className="w-16 h-16 mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">暂无知识库</h3>
              <p className="text-sm mb-4">
                还没有创建任何知识库，点击下方按钮开始创建吧！
              </p>
            </div>
            <Button 
              onClick={() => router.push('/knowledgebases/create')}
              className="gap-2"
            >
              <Plus className="w-4 h-4" />
              创建知识库
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
