'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from '@/components/ui/card';
import { Plus, Trash2, FileQuestion } from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { formatBeijingTime } from './utils/utils';

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  updated_at: string;
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
  const [knowledgebases, setKnowledgeBases] = useState(Array<KnowledgeBase>); // 知识库列表
  const [knowledgebasesloading, setKnowledgeBasesLoading] = useState(true); // 加载状态
  const [knowledgebasesrror, setKnowledgeBasesError] = useState(''); // 错误信息
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const kbSizePerPage = 6;
  const router = useRouter();

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_BACKEND_URL ?? ''}/v1/config/knowledgebases?page=${page}&size=${kbSizePerPage}`,
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
    };

    fetchConfigs();
  }, [page]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };
  const deleteKnowledgebase = async (kb_id: string) => {
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL ?? ''}/v1/config/knowledgebases/${kb_id}`, {
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
    <div className="flex flex-col h-screen p-6 space-y-6">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center h-1/10">
        <h1 className="text-2xl font-bold">知识库</h1>
        <Button
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 w-40"
          onClick={() => router.push('/knowledgebases/create')}
        >
          <Plus className="w-6 h-6" />
          新建知识库
        </Button>
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
                  className="flex flex-col border rounded-lg shadow-sm h-full gap-0 py-0 transition-shadow hover:shadow-md hover:bg-muted/50 duration-300"
                >
                  <CardHeader>
                    <CardTitle className="text-md flex pt-4 pb-1">
                      {base.name}
                    </CardTitle>
                  </CardHeader>

                  <CardContent className="pt-0 pb-0">
                    <p className="text-xs text-muted-foreground line-clamp-1">
                      {base.description
                        ? base.description
                        : '暂时还没有描述，可以去设置页面添加哦。'}
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

                    <div className="text-xs text-muted-foreground line-clamp-1 truncate">
                      {formatBeijingTime(base.updated_at)}
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>
            <div className="flex justify-center items-center h-1/10">
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
