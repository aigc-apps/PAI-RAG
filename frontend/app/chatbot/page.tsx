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
import { formatBeijingTime } from '../knowledgebase/utils/utils';

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

interface PageProps {
  
}

const ChatbotPage = (setActiveTab: (tab: string) => void) => {
  const [chatbots, setChatbots] = useState(Array<Chatbot>); // 知识库列表
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const pageSize = 6;

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const res = await fetch(
          `/v1/config/chatbots?page=${page}&size=${pageSize}`,
        );
        if (!res.ok) throw new Error('获取应用列表失败');
        const json_data = await res.json();
        const data = json_data.data.items;
        setChatbots(data || []); // 更新状态
        setTotalPages(json_data.data.pages);
      } catch (err: unknown) {
        console.log(err || '加载失败');
      }
    };

    fetchConfigs();
  }, [page]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };
  const deleteChatbot = async (bot_id: string) => {
    try {
      const res = await fetch(`/v1/config/chatbots/${bot_id}`, {
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
      setChatbots((prev) => prev.filter((bot) => bot.id !== bot_id));
    } catch (err: unknown) {
      console.log('删除Chatbot失败。', err);
    }
    // 显示错误提示
  };

  return (
    <div className="flex flex-col h-screen p-6 space-y-6">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center h-1/10">
        <h1 className="text-2xl font-bold">Chat应用</h1>
        <Button
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 w-40"
          onClick={() => setActiveTab('/chatbot/create')}
        >
          <Plus className="w-6 h-6" />
          新建应用
        </Button>
      </div>
      <div className="text-sm text-muted-foreground">
        应用可以给基模型配置知识库、联网、MCP工具。
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

                setActiveTab(`/chatbot/edit/${bot.app_id}`);
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
                        请注意，删除Chat应用无法撤销。请仔细核对之后再确认。
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>取消</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={() => deleteChatbot(bot.id)}
                      >
                        删除
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