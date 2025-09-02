'use client';
import React from 'react';
import { useState, useEffect, use, useMemo } from "react";
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


type ExperimentItem = {
  id: string
  samples_count: number
  name: string
  description: string
  status: string
  avg_score: number
  created_at: string
  updated_at: string
}

export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalId: string }> }
) {
    const { evalId } = use(params);
    const [evalConfig, setEvalConfig] = useState<EvalConfig>();
    const router = useRouter();
    const [experiments, setExperimentData] = useState<ExperimentItem[]>([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [isLoading, setIsLoading] = useState(true);
    const [dataseterror, setDatasetError] = useState(''); 
    const pageSize = 3;
    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("all");

    useEffect(() => {
          const fetchConfigs = async () => {
              setIsLoading(true);
              try {
                  const [evalRes, datasetRes] = await Promise.all([
                      fetch(`/api/config/evaluation/${evalId}`),
                      fetch(`/api/config/evaluation/${evalId}/experiments?page=${page}&size=${pageSize}`),
                  ]);
                  
                  const eval_data = await evalRes.json();
                  const evalData = eval_data.data;
                  console.log('evalData:', evalData);
                  setEvalConfig(evalData);
  
                  if (!datasetRes.ok) throw new Error('获取评估任务列表失败');
                  const json_data = await datasetRes.json();
                  console.log("evaluation dataset json_data", json_data)
                  const data = json_data.data.items;
                  
                  setExperimentData(data);
                  setTotalPages(json_data.data.pages);
              } catch (err: any) {
                  setDatasetError(err || '加载数据集失败');
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

    // // 过滤和搜索数据
    // const filteredData = useMemo(() => {
    //     return experiments.filter(item => {
    //         const matchesSearch = item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    //             item.settings.llm.toLowerCase().includes(searchTerm.toLowerCase());

    //         const matchesStatus = statusFilter === "all" || item.status === statusFilter;

    //         return matchesSearch && matchesStatus;
    //     });
    // }, [experiments, searchTerm, statusFilter]);

    // 状态标签样式
    const getStatusBadge = (status: string) => {
        switch (status) {
            case "running":
                return <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200">
                    <Clock className="mr-1 h-3 w-3" /> 运行中
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
                    <RefreshCw className="mr-1 h-3 w-3" /> 等待中
                </Badge>;
            default:
                return <Badge>{status}</Badge>;
        }
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

    // 格式化设置信息
    // const formatSettings = (settings: ExperimentItem["settings"]) => (
    //     <div className="space-y-1">
    //         <div className="flex">
    //             <span className="font-medium w-16">LLM:</span>
    //             <span className="text-muted-foreground">{settings.llm}</span>
    //         </div>
    //         <div className="flex">
    //             <span className="font-medium w-16">MCP:</span>
    //             <span className="text-muted-foreground">
    //                 {settings.mcp.join(", ")}
    //             </span>
    //         </div>
    //         <div className="flex">
    //             <span className="font-medium w-16">搜索:</span>
    //             <span className="text-muted-foreground">
    //                 {settings.search ? "启用" : "禁用"}
    //             </span>
    //         </div>
    //     </div>
    // );

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
            router.push(`/evaluation/${evalId}/experiments/${id}`)
        }
    };


    return (
        <div className="flex flex-col h-screen px-6 py-4 space-y-6">
            <div className="flex-none">
                <div className="p-2 space-y-2">
                    <div className="mb-2 flex items-center gap-2">
                        {/* 面包屑导航 */}
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
            </div>

            <div className="px-2 w-full">
                <div>
                    <Card className="w-full">
                        <CardHeader className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
                            <div>
                                <CardTitle>实验管理</CardTitle>
                                <p className="text-sm text-muted-foreground mt-1">
                                    管理您的AI评估实验
                                </p>
                            </div>

                            <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">
                                <div className="relative flex-1">
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
                                </div>

                                <div className="flex gap-2">
                                    <Select value={statusFilter} onValueChange={setStatusFilter}>
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
                                    </Select>

                                    {/* <Button>
                                <Play className="mr-2 h-4 w-4" /> 新建实验
                            </Button> */}
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
                                                            <div className="truncate max-w-[120px]" title={item.id}>
                                                                {item.id}
                                                            </div>
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
                                                                <DropdownMenuItem onClick={() => handleAction('view', item.id)}>
                                                                    <Eye className="mr-2 h-4 w-4" />
                                                                    查看详情
                                                                </DropdownMenuItem>
                                                                <DropdownMenuItem onClick={() => handleAction('rerun', item.id)}>
                                                                    <RefreshCw className="mr-2 h-4 w-4" />
                                                                    重新运行
                                                                </DropdownMenuItem>
                                                                <DropdownMenuItem
                                                                    className="text-red-600 focus:bg-red-100"
                                                                    onClick={() => handleAction('delete', item.id)}
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
        </div>
    );
}
