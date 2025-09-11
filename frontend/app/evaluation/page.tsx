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
import { Plus, Trash2 } from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { formatBeijingTime } from '@/app/knowledgebases/utils/utils';

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
      // 删除成功后更新本地状态
      setEvaluations((prev) => prev.filter((config) => config.id !== eval_id));
    } catch (err: any) { console.log('删除评估任务出错: ', err); }
    // 显示错误提示
  }

  return (
    <div className="flex flex-col h-screen px-6 py-0 space-y-6">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center h-1/10">
        <h1 className="text-2xl font-bold">数据集 & 评估</h1>
        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
          <DialogTrigger asChild>
            <Button>
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
                <Label
                  htmlFor="dataset_name"
                  className="text-right"
                >
                  数据集名称
                </Label>
                <Input
                  id="dataset_name"
                  className="col-span-3"
                  onChange={(e) => setEvalTaskName(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="dataset_desc"
                  className="text-right"
                >
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
              >
                {isCreateLoading ? '提交中...' : '新建'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

      </div>
      <div className="text-sm text-muted-foreground">
        评估数据集可以帮助你了解应用（AI助手）在不同数据集上的表现，从而选择最适合你需求的AI助手。
      </div>
      <div className="overflow-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow className='bg-muted/50 font-medium text-muted-foreground h-12'>
              <TableHead className="w-[200px] pl-6">数据集</TableHead>
              <TableHead>类型</TableHead>
              <TableHead>描述</TableHead>
              <TableHead>样本数</TableHead>
              <TableHead>实验数</TableHead>
              <TableHead>创建时间</TableHead>
              <TableHead className="text-right">操作</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {evaluations.map((evalTask) => (
              <TableRow
                key={evalTask.id}
                className="cursor-pointer hover:bg-muted/50 transition-colors"
                onClick={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.closest('button')) {
                    console.log('按钮被点击');
                    return; // 阻止跳转
                  }
                  router.push(`/evaluation/${evalTask.id}`);
                }}
              >
                <TableCell className="font-medium pl-6">{evalTask.name}</TableCell>
                <TableCell>{evalTask.name === "GAIA" ? "Built-in" : "Custom"}</TableCell>
                <TableCell className="text-sm text-muted-foreground max-w-md">
                  {evalTask.description || '暂时还没有描述，可以去设置页面添加哦。'}
                </TableCell>
                <TableCell>{evalTask.dataset_count}</TableCell>
                <TableCell>{evalTask.experiments_count}</TableCell>
                <TableCell>{formatBeijingTime(evalTask.created_at)}</TableCell>
                <TableCell className="text-right">
                  <Button variant="ghost" size="icon" className="hover:bg-red-500 hover:text-red-50" onClick={(e) => deleteEval(evalTask.id)}>
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {/* 分页组件 */}
      <div className="fixed bottom-0 left-0 right-0 flex justify-center items-center h-1/10">
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