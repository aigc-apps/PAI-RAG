'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Plus, Trash2, Database } from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { formatBeijingTime } from '@/app/knowledgebases/utils/utils';
import { Badge } from '@/components/ui/badge';

// 评估数据类型定义
interface EvalData {
  id: string;
  name: string;
  description: string;
  created_at: string;
  dataset_count: number;
  experiments_count: number;
}


const EvaluationPage = () => {
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [evaluations, setEvaluations] = useState(Array<EvalData>);
  const [isLoading, setIsLoading] = useState(true);
  const [evaluationerror, setEvaluationError] = useState('');
  const pageSize = 6;
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isCreateLoading, setIsCreateLoading] = useState(false);
  const [evalTaskName, setEvalTaskName] = useState("");
  const [evalTaskDesc, setEvalTaskDesc] = useState("");
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

  const createNewEvalDataset = async () => {
    const data = {
      name: evalTaskName,
      description: evalTaskDesc || "默认评估任务描述",
      type: "custom"
    };

    try {
      const res = await fetch(
        `/api/config/evaluation`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        },
      );
      if (!res.ok) {
        alert('创建失败');
        return;
      }
      const upload_result = await res.json();
      console.log('创建成功:', upload_result);
      setEvaluations((prev) => [...prev, upload_result.data]); // 追加新 LLM 配置
    } catch (error) {
      console.error('创建失败:', error);
    } finally {
      setIsCreateLoading(false);
      setIsCreateOpen(false);
    }
  }

  const deleteEval = async (eval_id: string) => {
    try {
      const res = await fetch(`/api/config/evaluation/${eval_id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        throw new Error('删除失败，请检查网络或配置');
      }
      setEvaluations((prev) => prev.filter((config) => config.id !== eval_id));
    } catch (err: any) { console.log('删除评估任务出错: ', err); }
    // 显示错误提示
  }

  return (
    <div className="flex flex-col h-screen px-6 py-6">
      <div className="mb-6">
        <div className="p-6 rounded-2xl border border-primary/20">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                数据集 & 评估
              </h1>
              <p className="text-sm text-muted-foreground mt-2 max-w-2xl">
                评估数据集可以帮助你了解应用（AI助手）在不同数据集上的表现，从而选择最适合你需求的AI助手。
              </p>
            </div>
            <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
              <DialogTrigger asChild>
                <Button className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-md hover:shadow-lg transition-all duration-200 transform hover:-translate-y-0.5">
                  <Plus className="mr-2 h-4 w-4" /> 新建数据集
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                  <DialogTitle>新建评估任务</DialogTitle>
                  <DialogDescription>
                    填写相关信息并上传数据集成功后，点击保存。
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="dataset_name" className="text-right">
                      数据集名称
                    </Label>
                    <Input
                      id="dataset_name"
                      className="col-span-3"
                      onChange={(e) => setEvalTaskName(e.target.value)}
                    />
                  </div>
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="dataset_desc" className="text-right">
                      数据集描述
                    </Label>
                    <Input
                      id="dataset_desc"
                      className="col-span-3"
                      onChange={(e) => setEvalTaskDesc(e.target.value)}
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => setIsCreateOpen(false)}
                  >
                    取消
                  </Button>
                  <Button
                    onClick={createNewEvalDataset}
                    type="submit"
                    disabled={isCreateLoading}
                    className="bg-primary hover:bg-primary/90"
                  >
                    {isCreateLoading ? '提交中...' : '新建'}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>

      {/* 数据表格区 —— 卡片容器 + 悬停效果 */}
      <div className="flex-1 overflow-hidden rounded-2xl border bg-card shadow-sm hover:shadow-md transition-shadow duration-300">
        <div className="p-6 border-b">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            数据集列表
          </h2>
        </div>
        <div className="overflow-auto h-full">
          <Table className=' border-b'>
            <TableHeader>
              <TableRow className="hover:bg-muted/30 transition-colors">
                <TableHead className="w-[200px] pl-8">数据集</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>描述</TableHead>
                <TableHead>样本数</TableHead>
                <TableHead>实验数</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead className="text-right pr-6">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {evaluations.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="h-32 text-center text-muted-foreground">
                    <div className="flex flex-col items-center gap-2">
                      <Database className="h-8 w-8 text-muted-foreground/50" />
                      <span>暂无数据集</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setIsCreateOpen(true)}
                        className="mt-2"
                      >
                        <Plus className="mr-1 h-3 w-3" /> 创建第一个数据集
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                evaluations.map((evalTask) => (
                  <TableRow
                    key={evalTask.id}
                    className="cursor-pointer hover:bg-muted/30 transition-colors group border-b"
                    onClick={(e) => {
                      const target = e.target as HTMLElement;
                      if (target.closest('button')) {
                        console.log('按钮被点击');
                        return;
                      }
                      router.push(`/evaluation/${evalTask.id}`);
                    }}
                  >
                    <TableCell className="font-medium pl-8 group-hover:text-primary transition-colors">
                      {evalTask.name}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={evalTask.name === "GAIA" ? "default" : "secondary"}
                        className={evalTask.name === "GAIA" ? "bg-blue-100 text-blue-800" : "bg-purple-100 text-purple-800"}
                      >
                        {evalTask.name === "GAIA" ? "内置" : "自定义"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground max-w-md">
                      {evalTask.description || '暂时还没有描述，可以去设置页面添加哦。'}
                    </TableCell>
                    <TableCell className="font-medium">{evalTask.dataset_count}</TableCell>
                    <TableCell className="font-medium">{evalTask.experiments_count}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatBeijingTime(evalTask.created_at)}
                    </TableCell>
                    <TableCell className="text-right pr-6">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500 hover:text-red-50"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteEval(evalTask.id);
                        }}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      {/* 分页组件 —— 固定在底部，带背景 */}
      <div className="fixed bottom-0 left-0 right-0 bg-background/80 backdrop-blur-sm border-t py-3">
        <div className="flex justify-center">
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      </div>
    </div>
  );
}

export default EvaluationPage;