'use client';
import React from 'react';
import { useState, useEffect, use, useCallback, useRef } from "react";
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
    RefreshCw,
    CheckCircle,
    XCircle,
    Clock,
    Trash2Icon,
    Play,
    BarChart2
} from "lucide-react";
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
    const pageSize = 10;

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
            isRefreshing = false;
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
        return score.toFixed(2);
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
                                {experiments.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-32 text-center">
                                            <div className="flex flex-col items-center gap-2 text-muted-foreground">
                                                <BarChart2 className="h-8 w-8" />
                                                <span>暂无实验记录，请前往样本页面下选中样本进行实验</span>
                                                {/* <Button
                                                            variant="outline"
                                                            size="sm"
                                                            onClick={() => router.push(`/evaluation/${evalId}`)}
                                                            className="hover:bg-blue-50"
                                                        >
                                                            <Play className="mr-1 h-3 w-3" /> 新建实验
                                                        </Button> */}
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
                                                {getStatusBadge(item.status)}
                                            </TableCell>

                                            <TableCell>
                                                <div className={`font-bold text-lg ${item.status === 'success'
                                                    ? item.avg_score && item.avg_score >= 0.8
                                                        ? 'text-green-600'
                                                        : item.avg_score && item.avg_score >= 0.6
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
                                                        {/* <DropdownMenuItem 
                            onClick={() => handleAction('rerun', item.id)}
                          >
                            <RefreshCw className="mr-2 h-4 w-4" />
                            <span>重新运行</span>
                          </DropdownMenuItem> */}
                                                        <DropdownMenuItem
                                                            className="text-red-600 focus:bg-red-50 focus:text-red-700"
                                                            onClick={() => handleDeleteAction(item.id)}
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
