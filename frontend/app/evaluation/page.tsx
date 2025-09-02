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
import { useRouter } from 'next/navigation';

// 评估数据类型定义
interface EvalExperiment {
  id: string;
  name: string;
  description: string;
}


const EvaluationPage = () => {
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [evaluations, setEvaluations] = useState(Array<EvalExperiment>);
  const [isLoading, setIsLoading] = useState(true);
  const [evaluationerror, setEvaluationError] = useState(''); 
  const pageSize = 6;
  const router = useRouter();


  useEffect(() => {
      const fetchConfigs = async () => {
        setIsLoading(true);
        try {
          const res = await fetch(
            `/api/config/evaluation?page=${page}&size=${pageSize}`,
          );
          if (!res.ok) throw new Error('获取评估任务列表失败');
          const json_data = await res.json();
          console.log("evaluation json_data", json_data)
          const data = json_data.data.items;
          setEvaluations(data);
          setTotalPages(json_data.data.pages);
        } catch (err: any) {
          setEvaluationError(err || '加载失败');
        } finally {
          setIsLoading(false);
        }
      };
  
      fetchConfigs();
    }, [page]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  return (
    <div className="flex flex-col h-screen px-6 py-0 space-y-6">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center h-1/10">
        <h1 className="text-2xl font-bold">评估实验</h1>
        <Button
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 w-40"
          onClick={()=>{router.push('/apps/create')}}
        >
          <Plus className="w-6 h-6" />
          新建评估实验
        </Button>
      </div>
      <div className="text-sm text-muted-foreground">
        评估实验描述
      </div>

      {/* 卡片容器 */}
      <div className="h-4/5">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
          {evaluations.map((experiment) => (
            <Card
              onClick={(e) => {
                // 检查是否点击了交互元素
                const target = e.target as HTMLElement;

                if (target instanceof HTMLElement && target.closest('button')) {
                  console.log('按钮被点击');
                  return; // 是交互元素，不触发卡片跳转
                }
                router.push(`/evaluation/${experiment.id}`);
              }}
              key={experiment.id}
              className="flex flex-col border rounded-lg shadow-sm h-full gap-0 py-0 transition-shadow hover:shadow-md hover:bg-muted/50 duration-300"
            >
              <CardHeader>
                <CardTitle className="text-md flex pt-4 pb-1">
                  {experiment.name}
                </CardTitle>
              </CardHeader>

              <CardContent className="pt-0 pb-0">
                <p className="text-xs text-muted-foreground line-clamp-1">
                  {experiment.description
                    ? experiment.description
                    : '暂时还没有描述，可以去设置页面添加哦。'}
                </p>
              </CardContent>
              <CardFooter className="px-3 pt-0 flex justify-between w-full py-4">
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

export default EvaluationPage;