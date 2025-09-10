'use client';
import React from 'react';
import { useState, useEffect, use, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
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
    RefreshCw,
    CheckCircle,
    XCircle,
    Clock,
    Trash2Icon,
    Play,
} from "lucide-react";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { EvalConfig } from '@/app/evaluation/[evalId]/page';
import { formatBeijingTime } from '@/app/knowledgebases/utils/utils';
import { toast } from 'sonner';

export interface ExperimentItem {
  id: string;
  samples_count: number;
  name: string;
  description: string;
  status: string;
  run_config_id: string;
  avg_score: number;
  created_at: string;
  updated_at: string;
}

// 状态标签样式
export const getStatusBadge = (status: string) => {
    switch (status) {
        case "running":
            return <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200">
                <RefreshCw className="mr-1 h-3 w-3 animate-spin" /> 运行中
            </Badge>;
        case "success":
            return <Badge variant="secondary" className="bg-green-100 text-green-800 hover:bg-green-200">
                <CheckCircle className="mr-1 h-3 w-3" /> 成功
            </Badge>;
        case "failed":
            return <Badge variant="secondary" className="bg-red-100 text-red-800 hover:bg-red-200">
                <XCircle className="mr-1 h-3 w-3" /> 失败
            </Badge>;
        case "pending":
            return <Badge variant="secondary" className="bg-yellow-100 text-yellow-800 hover:bg-yellow-200">
                <Clock className="mr-1 h-3 w-3 animate-spin" /> 等待中
            </Badge>;
        default:
            return <Badge>{status}</Badge>;
    }
};

export default function EvalExperimentsDetailsPage(
    { params }: { params: Promise<{ evalId: string }> }
) {
    const { evalId } = use(params);
    let isRefreshing = false;
    const [evalConfig, setEvalConfig] = useState<EvalConfig>();
    const router = useRouter();
    const [experiments, setExperimentData] = useState<ExperimentItem[]>([]);
    const [page, setPage] = useState(1);
    const pageRef = useRef(page);
    const [totalPages, setTotalPages] = useState(1);
    const [isLoading, setIsLoading] = useState(true);
    const [dataseterror, setDatasetError] = useState(''); 
    const pageSize = 8;
    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("all");

    const fetchExperiments = useCallback(async () => {
        if (isRefreshing) {
          console.log("list already refreshing.")
          return;
        }
        console.log("Refreshing...");
        const url = `/api/config/evaluation/${evalId}/experiments?page=${pageRef.current}&size=${pageSize}`
    
        try {
          isRefreshing = true;
          const files_res = await fetch(url);
          if (!files_res.ok) throw new Error('获取实验列表失败');
    
          const exp_json_data = await files_res.json();
          console.log('获取实验reponse:', exp_json_data);
          const data = exp_json_data.data.items;
          setExperimentData(data || []);
          setTotalPages(exp_json_data.data.pages);
    
          const kb_files = data as ExperimentItem[];
          const files_unfinished = kb_files.some(
            (file) => file.status !== 'success' && file.status !== 'failed',
          );
    
          if (files_unfinished) {
            console.log('存在未完成的实验，继续检查状态。');
            setTimeout(() => {
              isRefreshing = false;
              fetchExperiments(); // 依赖 ref 获取最新 page
            }, 3000);
          } else {
            console.log('实验已完成。');
          }
          isRefreshing=false;
        } catch (err: any) {
          isRefreshing = false;
          toast.error(err.message);
        }
    }, [evalId]);

    useEffect(() => {
        pageRef.current = page;
      }, [page]);

    useEffect(() => {
        fetchExperiments();
    }, [fetchExperiments, page, experiments.length]);

    useEffect(() => {
          const fetchConfigs = async () => {
              setIsLoading(true);
              try {
                  const [evalRes] = await Promise.all([
                      fetch(`/api/config/evaluation/${evalId}`),
                    //   fetch(`/api/config/evaluation/${evalId}/experiments?page=${page}&size=${pageSize}`),
                  ]);
                  
                  const eval_data = await evalRes.json();
                  const evalData = eval_data.data;
                  console.log('evalData:', evalData);
                  setEvalConfig(evalData);
              } catch (err: any) {
                  setDatasetError(err || '加载数据集失败');
              } finally {
                  setIsLoading(false);
              }
          };
          fetchConfigs();
      }, []);

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
        return (score * 100).toFixed(0) + "%";
    };

    // 复制实验ID
    const copyExperimentId = (id: string) => {
        navigator.clipboard.writeText(id);
        // 这里可以添加一个toast通知
    };

    // 表格操作
    const handleAction = (action: string, id: string) => {
        console.log(`执行操作: ${action} - ${id}`);
        // 这里可以添加实际操作逻辑
        if (action === "view") {
            router.push(`/evaluation/${evalId}/${id}`)
        }
    };

    const handleDeleteAction = async (id: string) => {
        console.log(`Deleting experiment ${id}`);
        try {
            const res = await fetch(`/api/config/evaluation/${evalId}/experiments/${id}`, {
                method: 'DELETE',
                headers: {
                'Content-Type': 'application/json',
                },
            });

            if (!res.ok) {
                throw new Error('删除失败，请检查网络或配置');
            }

            // 删除成功后更新本地状态
            setExperimentData((prev) => prev.filter((experiment) => experiment.id !== id));
        } catch (err: any) {
            console.log('删除实验出错: ', err);
        }
    }

    return (
        <div className="flex flex-col h-screen py-4 space-y-6">
            {/* <div className="flex-none">
                <div className="p-2 space-y-2">
                    <div className="mb-2 flex items-center gap-2">
                        <Breadcrumb>
                            <BreadcrumbList>
                                <BreadcrumbItem>
                                    <BreadcrumbLink asChild>
                                        <Button
                                            variant="link"
                                            className="px-0"
                                            onClick={() => router.push('/evaluation')}
                                        >
                                            评估
                                        </Button>
                                    </BreadcrumbLink>
                                </BreadcrumbItem>
                                <BreadcrumbSeparator />
                                <BreadcrumbItem>
                                    <Button
                                        variant="link"
                                        className="px-0"
                                        onClick={() => router.push(`/evaluation/${evalId}`)}
                                    >
                                        {evalConfig?.name}
                                    </Button>
                                </BreadcrumbItem>
                                <BreadcrumbSeparator />
                                <BreadcrumbItem>
                                    <BreadcrumbPage>experiments</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                    <div className="flex justify-between items-center px-2">
                        <div>
                            <h1 className="text-2xl font-bold">实验</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                评估实验设置&查看
                            </div>
                        </div>
                    </div>
                </div>
            </div> */}

            <div className="w-full">
                    <Card className="w-full">
                        <CardHeader className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
                            <div>
                                <CardTitle>实验管理</CardTitle>
                                <p className="text-sm text-muted-foreground mt-1">
                                    管理您的AI评估实验
                                </p>
                            </div>

                            <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">
                                {/* <div className="relative flex-1">
                                    <Input
                                        placeholder="搜索实验ID或模型..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="pl-10"
                                    />
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        className="lucide lucide-search absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                                    >
                                        <circle cx="11" cy="11" r="8" />
                                        <path d="m21 21-4.3-4.3" />
                                    </svg>
                                </div> */}

                                <div className="flex gap-2">
                                    {/* <Select value={statusFilter} onValueChange={setStatusFilter}>
                                        <SelectTrigger className="w-[160px]">
                                            <SelectValue placeholder="状态筛选" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="all">全部状态</SelectItem>
                                            <SelectItem value="running">运行中</SelectItem>
                                            <SelectItem value="success">成功</SelectItem>
                                            <SelectItem value="failed">失败</SelectItem>
                                            <SelectItem value="pending">等待中</SelectItem>
                                        </SelectContent>
                                    </Select> */}

                                    <Button onClick={() => router.push(`/evaluation/${evalId}`)}>
                                        <Play className="mr-2 h-4 w-4" /> 新建实验
                                    </Button>
                                </div>
                            </div>
                        </CardHeader>

                        <CardContent>
                            <div className="rounded-md border">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead className="w-[180px]">实验ID</TableHead>
                                            <TableHead className="w-[180px]">实验名称</TableHead>
                                            <TableHead className="w-[180px]">实验描述</TableHead>
                                            <TableHead className="w-[100px]">样本数</TableHead>
                                            {/* <TableHead className="w-[200px]">模型设置</TableHead> */}
                                            <TableHead className="w-[120px]">状态</TableHead>
                                            <TableHead className="w-[100px]">平均得分</TableHead>
                                            <TableHead className="w-[160px]">创建时间</TableHead>
                                            <TableHead className="w-[160px]">完成时间</TableHead>
                                            <TableHead className="w-[50px] text-right">操作</TableHead>
                                        </TableRow>
                                    </TableHeader>

                                    <TableBody>
                                        {experiments.length === 0 ? (
                                            <TableRow>
                                                <TableCell colSpan={8} className="h-24 text-center">
                                                    暂无数据
                                                </TableCell>
                                            </TableRow>
                                        ) : (
                                            experiments.map((item) => (
                                                <TableRow key={item.id} className="hover:bg-muted/50 transition-colors">
                                                    <TableCell className="font-medium">
                                                        <div className="flex items-center">
                                                            <Button
                                                                variant="link"
                                                                className="truncate max-w-[120px] font-medium text-blue-600"
                                                                onClick={() =>
                                                                    router.push(
                                                                        `/evaluation/${evalId}/${item.id}`,
                                                                    )
                                                                }
                                                            >
                                                                {item.id}
                                                            </Button>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-6 w-6 ml-1"
                                                                onClick={() => copyExperimentId(item.id)}
                                                                title="复制实验ID"
                                                            >
                                                                <Copy className="h-3 w-3" />
                                                            </Button>
                                                        </div>
                                                    </TableCell>

                                                    <TableCell>
                                                        <Badge variant="outline" className="font-mono">
                                                            {item.name}
                                                        </Badge>
                                                    </TableCell>

                                                    <TableCell>
                                                        {item.description.substring(0, 20)}...
                                                    </TableCell>

                                                    <TableCell>
                                                        <Badge className="bg-blue-50 text-blue-700 hover:bg-blue-100">
                                                            {item.samples_count}
                                                        </Badge>
                                                    </TableCell>

                                                    <TableCell>
                                                        {getStatusBadge(item.status)}
                                                    </TableCell>

                                                    <TableCell>
                                                        <div className={`font-medium ${item.status === 'success' ? 'text-green-600' : 'text-muted-foreground'}`}>
                                                            {formatScore(item.avg_score, item.status)}
                                                        </div>
                                                    </TableCell>

                                                    <TableCell>
                                                        {formatBeijingTime(item.created_at)}
                                                    </TableCell>

                                                    <TableCell>
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
                                                                    onClick={() =>
                                                                        router.push(
                                                                            `/evaluation/${evalId}/${item.id}`,
                                                                        )
                                                                    }
                                                                >
                                                                    <Eye className="mr-2 h-4 w-4" />
                                                                    查看详情
                                                                </DropdownMenuItem>
                                                                <DropdownMenuItem onClick={() => handleAction('rerun', item.id)}>
                                                                    <RefreshCw className="mr-2 h-4 w-4" />
                                                                    重新运行
                                                                </DropdownMenuItem>
                                                                <DropdownMenuItem
                                                                    className="text-red-600 focus:bg-red-100"
                                                                    onClick={() => handleDeleteAction(item.id)}
                                                                >
                                                                    <Trash2Icon /> 删除
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

                        <CardFooter className="flex justify-center">
                            <div className="py-6">
                                <PaginationComponent
                                    currentPage={page}
                                    totalPages={totalPages}
                                    onPageChange={handlePageChange}
                                />
                            </div>
                        </CardFooter>
                    </Card>
            </div>
        </div>
    );
}
