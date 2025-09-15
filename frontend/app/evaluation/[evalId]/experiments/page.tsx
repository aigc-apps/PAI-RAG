'use client';

import React from 'react';
import { useState, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from '@/components/ui/button';
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import {
  MoreHorizontal,
  Copy,
  Eye,
  BookOpen,
  BarChart2,
  Trash2Icon,
  Loader2
} from "lucide-react";
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { formatBeijingTime } from '@/app/knowledgebases/utils/utils';
import { toast } from 'sonner';

import { EvalConfig } from '@/app/evaluation/[evalId]/types';
import { StatusBadge } from '@/app/evaluation/components/StatusBadge';
import { useExperiments } from '@/app/evaluation/[evalId]/experiments/useExperiments';

export default function EvalExperimentsDetailsPage(
  { params }: { params: Promise<{ evalId: string }> }
) {
  const { evalId } = use(params);
  const router = useRouter();

  // 页面状态
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const [evalConfig, setEvalConfig] = useState<EvalConfig>();

  // 🆕 使用自定义 Hook
  const {
    experiments,
    totalPages,
    isLoading,
    deleteExperiment
  } = useExperiments({ evalId, page, pageSize });

  // 加载评估配置
  useEffect(() => {
    const fetchEvalConfig = async () => {
      try {
        const response = await fetch(`/api/config/evaluation/${evalId}`);
        if (!response.ok) throw new Error('获取评估配置失败');
        const data = await response.json();
        setEvalConfig(data.data);
      } catch (err: any) {
        console.error('加载评估配置失败:', err);
        toast.error('加载评估配置失败');
      }
    };

    fetchEvalConfig();
  }, [evalId]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  // 格式化平均得分
  const formatScore = (score: number, status: string) => {
    if (status === "running" || status === "pending") {
      return <span className="text-muted-foreground">进行中...</span>;
    }
    if (status === "failed") {
      return <span className="text-red-500">- -</span>;
    }
    return score.toFixed(2);
  };

  // 复制实验ID
  const copyExperimentId = (id: string) => {
    navigator.clipboard.writeText(id);
    toast.success("复制成功");
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <Card className="flex flex-col h-full min-h-0 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
        <CardHeader className="shrink-0 flex md:items-center md:justify-between">
          <div>
            <CardTitle className="text-2xl font-bold flex items-center gap-2">
              <BarChart2 className="h-5 w-5" /> 运行历史
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              查看运行历史及详情
            </p>
            <span className="flex text-sm text-muted-foreground">
              新建实验请前往左侧 “<BookOpen className="h-3.5 w-3.5 inline-block mr-1 mt-1" />样本”页面选中数据并运行
            </span>
          </div>
        </CardHeader>

        <CardContent className="flex-1 min-h-0 overflow-y-auto p-0">
          <div className="rounded-md h-full min-h-0">
            <Table className='rounded-md border'>
              <TableHeader>
                <TableRow className="transition-colors">
                  <TableHead className="w-[180px]">实验ID</TableHead>
                  <TableHead className="w-[180px]">实验名称</TableHead>
                  <TableHead className="w-[180px]">实验描述</TableHead>
                  <TableHead className="w-[100px]">样本数</TableHead>
                  <TableHead className="w-[120px]">状态</TableHead>
                  <TableHead className="w-[100px]">平均得分</TableHead>
                  <TableHead className="w-[160px]">创建时间</TableHead>
                  <TableHead className="w-[160px]">完成时间</TableHead>
                  <TableHead className="w-[50px] text-right">操作</TableHead>
                </TableRow>
              </TableHeader>

              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={9} className="h-32 text-center">
                      <div className="flex items-center justify-center space-x-4">
                          <Loader2 className="h-6 w-6 animate-spin" />
                          <h4 className="font-bold">Loading Experiments</h4>
                      </div>
                    </TableCell>
                  </TableRow>
                ) : experiments.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={9} className="h-32 text-center">
                      <div className="flex flex-col items-center gap-2 text-muted-foreground">
                        <BarChart2 className="h-8 w-8" />
                        <span>暂无实验记录，请前往样本页面下选中样本进行实验</span>
                      </div>
                    </TableCell>
                  </TableRow>
                ) : (
                  experiments.map((item) => (
                    <TableRow key={item.id} className="transition-colors group">
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-1">
                          <Button
                            variant="link"
                            className="truncate max-w-[120px] font-medium group-hover:underline"
                            onClick={() => router.push(`/evaluation/${evalId}/${item.id}`)}
                          >
                            {item.id.substring(0, 8)}...
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => copyExperimentId(item.id)}
                            title="复制实验ID"
                          >
                            <Copy className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>

                      <TableCell>
                        <Badge variant="outline" className="font-mono bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100">
                          {item.name}
                        </Badge>
                      </TableCell>

                      <TableCell className="text-sm">
                        {item.description ? item.description.substring(0, 20) + "..." : "无描述"}
                      </TableCell>

                      <TableCell>
                        <Badge className="bg-purple-50 text-purple-700 hover:bg-purple-100 border-purple-200">
                          {item.samples_count}
                        </Badge>
                      </TableCell>

                      <TableCell>
                        <StatusBadge status={item.status} />
                      </TableCell>

                      <TableCell>
                        <div className={`font-bold text-lg ${item.status === 'success'
                          ? item.avg_score >= 0.8
                            ? 'text-green-600'
                            : item.avg_score >= 0.6
                              ? 'text-yellow-600'
                              : 'text-red-600'
                          : 'text-muted-foreground'
                          }`}>
                          {formatScore(item.avg_score, item.status)}
                        </div>
                      </TableCell>

                      <TableCell className="text-sm text-muted-foreground">
                        {formatBeijingTime(item.created_at)}
                      </TableCell>

                      <TableCell className="text-sm text-muted-foreground">
                        {['success', 'failed'].includes(item.status) ? formatBeijingTime(item.updated_at) : "-"}
                      </TableCell>

                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" className="h-8 w-8 p-0">
                              <span className="sr-only">打开菜单</span>
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem
                              onClick={() => router.push(`/evaluation/${evalId}/${item.id}`)}
                            >
                              <Eye className="mr-2 h-4 w-4" />
                              <span>查看详情</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-red-600 focus:bg-red-50 focus:text-red-700"
                              onClick={() => deleteExperiment(item.id)}
                            >
                              <Trash2Icon className="mr-2 h-4 w-4" />
                              <span>删除</span>
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>

        <CardFooter className="shrink-0 border-t pb-2">
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </CardFooter>
      </Card>
    </div>
  );
}