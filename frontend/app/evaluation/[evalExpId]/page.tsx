'use client';

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
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Button } from '@/components/ui/button';
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import { PlayCircleIcon, ChevronsRightIcon, UndoIcon } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { EvalConfigCard } from "../eval_config";

// 评估数据类型定义
interface EvaluationItem {
    id: string;
    query: string;
    level: number;
    agentResponse?: string;
    groundTruth: string;
    score?: number;
}

export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalExpId: string }> }
) {
    const { evalExpId } = use(params);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [evaluations, setEvaluations] = useState<EvaluationItem[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const pageSize = 6;
    const router = useRouter();

    useEffect(() => {
        const fetchConfigs = async () => {
            setIsLoading(true);

            // 模拟API调用获取评估数据
            const mockData: EvaluationItem[] = [
                {
                    id: "e1fc63a2-da7a-432f-be78-7c4a95598703",
                    query: "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.",
                    level: 1,
                    agentResponse: "17000",
                    groundTruth: "17",
                    score: 0.1
                },
                {
                    id: "46719c30-f4c3-4cad-be07-d5cb21eee6bb",
                    query: "Of the authors (First M. Last) that worked on the paper \"Pie Menus or Linear Menus, Which Is Better?\" in 2015, what was the title of the first paper authored by the one that had authored prior papers?",
                    level: 1,
                    agentResponse: "Mapping Human Oriented Information to Software Agents for Online Systems Usage",
                    groundTruth: "Mapping Human Oriented Information to Software Agents for Online Systems Usage",
                    score: 1.0
                },
                {
                    id: "b816bfce-3d80-4913-a07d-69b752ce6377",
                    query: "In Emily Midkiff's June 2014 article in a journal named for the one of Hreidmar's sons that guarded his house, what word was quoted from two different authors in distaste for the nature of dragon depictions?",
                    level: 2,
                    agentResponse: "The answer is fluffy.",
                    groundTruth: "fluffy",
                    score: 0.8
                },
                {
                    id: "72e110e7-464c-453c-a309-90a95aed6538",
                    query: "Under DDC 633 on Bielefeld University Library's BASE, as of 2020, from what country was the unknown language article with a flag unique from the others?",
                    level: 2,
                    groundTruth: "Guatemala",
                }
            ]

            // 计算分页 [[7]]
            const startIndex = (page - 1) * pageSize;
            const paginatedData = mockData.slice(startIndex, startIndex + pageSize);

            setEvaluations(paginatedData);
            setTotalPages(Math.ceil(mockData.length / pageSize));
            setIsLoading(false);
        };

        fetchConfigs();
    }, [page]);

    const handlePageChange = (newPage: number) => {
        if (newPage < 1 || newPage > totalPages) return;
        setPage(newPage);
    };

    // 对比结果的渲染逻辑
    const renderComparison = (item: EvaluationItem) => {
        const isMatch = item.score === 1.0 ? 1 : 0;

        return (
            <div className="space-y-2">
                {item.agentResponse && (
                    <div>
                        <div>
                            <span className="font-medium">Agent结果: </span>
                            <span className={isMatch ? "text-green-600" : "text-red-600"}>
                                {item.agentResponse}
                            </span>
                        </div>
                        <div>
                            <span className="font-medium">标准答案: </span>
                            {item.groundTruth}
                        </div>
                        <Badge variant={isMatch ? "default" : "destructive"}>
                            {isMatch ? "匹配" : "不匹配"}
                        </Badge>
                        {item.score && (
                            <div className="mt-1 text-sm">
                                评分: {(item.score * 100).toFixed(0)}/100
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
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
                                    <BreadcrumbPage>{evalExpId}</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold">GAIA</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                评估数据集: GAIA数据集 | 总计53个Level-1的测试用例
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="flex-1 overflow-y-auto px-2">
                <Tabs defaultValue="settings">
                    <TabsList className="py-4 bg-muted rounded-lg flex-none">
                        <TabsTrigger value="settings" className="p-4">
                            实验设置
                        </TabsTrigger>
                        <TabsTrigger value="datasets" className="p-4">
                            数据集
                        </TabsTrigger>
                        <TabsTrigger value="results" className="p-4">
                            评估实验
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="settings" className="py-4">
                        <EvalConfigCard eval_id={undefined} />
                    </TabsContent>
                    <TabsContent value="datasets" className="py-4">
                        <div className="overflow-y-auto">
                            <Card className="">
                                <CardHeader>
                                    <CardTitle>查询评估结果</CardTitle>
                                </CardHeader>
                                <CardContent className="">
                                    {isLoading ? (
                                        <div className="flex justify-center items-center h-64">
                                            <div className="text-muted-foreground">加载评估数据中...</div>
                                        </div>
                                    ) : (
                                        <Table className="w-full table-fixed border bg-white rounded-md overflow-hidden">
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead className="w-1/5">查询ID</TableHead>
                                                    <TableHead className="w-3/10">用户查询</TableHead>
                                                    <TableHead className="w-1/20">查询难度</TableHead>
                                                    <TableHead className="w-1/20">运行</TableHead>
                                                    <TableHead className="w-7/20">评估结果</TableHead>
                                                    <TableHead className="w-1/20">运行详情</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {evaluations.map((item) => (
                                                    <TableRow key={item.id}>
                                                        <TableCell className="font-medium">{item.id}</TableCell>
                                                        <TableCell
                                                            className="whitespace-normal break-words min-w-[150px] max-w-[300px]  py-2"
                                                        >
                                                            {item.query}
                                                        </TableCell>
                                                        <TableCell>{item.level}</TableCell>
                                                        <TableCell>
                                                            <Button variant="link"
                                                                size="icon"
                                                                className="size-8 text-blue-500">
                                                                <PlayCircleIcon />
                                                            </Button>
                                                        </TableCell>
                                                        <TableCell className="truncate">
                                                            {renderComparison(item)}
                                                        </TableCell>
                                                        <TableCell>
                                                            <Button variant="link"
                                                                size="icon"
                                                                className="size-8">
                                                                <ChevronsRightIcon />
                                                            </Button>
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    )}
                                </CardContent>
                            </Card>
                        </div>
                        <div className="py-6">
                            <PaginationComponent
                                currentPage={page}
                                totalPages={totalPages}
                                onPageChange={handlePageChange}
                            />
                        </div>
                    </TabsContent>
                    <TabsContent value="results" className="py-4">
                        
                    </TabsContent>
                </Tabs>
            </div>
        </div>
    );
}
