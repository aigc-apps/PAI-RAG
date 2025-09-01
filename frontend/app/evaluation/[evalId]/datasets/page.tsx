'use client';
import React from 'react';
import { useState, useEffect, use, useRef, useMemo } from "react";
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
    ChevronDown,
    ChevronUp,
    PlayIcon,
    CopyIcon,
    UploadIcon,
    Eye,
    Trash2Icon,
} from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';

interface SampleItem {
    id: string;
    question: string;
    answer: string;
    level?: number;
    tools?: string[];
}

export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalId: string }> }
) {
    const { evalId } = use(params);
    const router = useRouter();
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [datasets, setDatasets] = useState<SampleItem[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const pageSize = 3;
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [searchTerm, setSearchTerm] = useState("");
    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
    // 用于跟踪选中的行
    const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
    const [allItems, setAllItems] = useState<SampleItem[]>([]);
    const isAllSelected = selectedItems.size === allItems.length && allItems.length > 0;
    const [totalItems, setTotalItems] = useState(0);
    const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
    const [isDetailOpen, setIsDetailOpen] = useState(false);

    useEffect(() => {
        const fetchConfigs = async () => {
            setIsLoading(true);

            // 模拟API调用获取评估数据
            const mockData: SampleItem[] = [
                {
                    id: "e1fc63a2-da7a-432f-be78-7c4a95598703",
                    question: "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.",
                    answer: "17",
                    level: 1,
                    tools: ["Calculator", "Astronomy Database"]
                },
                {
                    id: "46719c30-f4c3-4cad-be07-d5cb21eee6bb",
                    question: "Of the authors (First M. Last) that worked on the paper \"Pie Menus or Linear Menus, Which Is Better?\" in 2015, what was the title of the first paper authored by the one that had authored prior papers?",
                    answer: "Mapping Human Oriented Information to Software Agents for Online Systems Usage",
                    level: 1,
                    tools: ["Search", "Browser Use", "Calculator"]
                },
                {
                    id: "b816bfce-3d80-4913-a07d-69b752ce6377",
                    question: "In Emily Midkiff's June 2014 article in a journal named for the one of Hreidmar's sons that guarded his house, what word was quoted from two different authors in distaste for the nature of dragon depictions?",
                    answer: "fluffy",
                    level: 2,
                    tools: ["Search", "Calculator"]
                },
                {
                    id: "72e110e7-464c-453c-a309-90a95aed6538",
                    question: "Under DDC 633 on Bielefeld University Library's BASE, as of 2020, from what country was the unknown language article with a flag unique from the others?",
                    answer: "Guatemala",
                    level: 2,
                    tools: ["Search"]
                },
                {
                    id: "9d5a0b4c-8f3a-4e7b-9c1d-3e6a2b1c0d9e",
                    question: "What is the capital of France and what is the square root of 144? Please provide both answers separated by a comma.",
                    answer: "Paris, 12",
                    level: 1,
                    tools: ["Browser Use", "Search"]
                },
                {
                    id: "a1b2c3d4-e5f6-4g7h-8i9j-0k1l2m3n4o5p",
                    question: "If a train leaves station A at 9:00 AM traveling at 60 km/h and another train leaves station B at 10:00 AM traveling at 80 km/h, when will they meet if the distance between stations is 420 km?",
                    answer: "12:00 PM",
                    level: 1,
                    tools: ["Browser Use"]
                }
            ];

            const startIndex = (page - 1) * pageSize;
            const paginatedData = mockData.slice(startIndex, startIndex + pageSize);

            setDatasets(paginatedData);
            setAllItems(mockData);
            setTotalItems(mockData.length);
            setTotalPages(Math.ceil(mockData.length / pageSize));
            setIsLoading(false);
        };

        fetchConfigs();
    }, [searchTerm, page]);

    const handlePageChange = (newPage: number) => {
        if (newPage < 1 || newPage > totalPages) return;
        setPage(newPage);
    };

    // 过滤和搜索数据
    const filteredData = useMemo(() => {
        return datasets.filter(item => {
            const matchesSearch = item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                item.question.toLowerCase().includes(searchTerm.toLowerCase()) ||
                item.answer.toLowerCase().includes(searchTerm.toLowerCase());

            return matchesSearch;
        });
    }, [datasets, searchTerm]);

    // 切换行展开状态
    const toggleRowExpansion = (id: string) => {
        setExpandedRows(prev => {
            const newSet = new Set(prev);
            if (newSet.has(id)) {
                newSet.delete(id);
            } else {
                newSet.add(id);
            }
            return newSet;
        });
    };

    // 复制ID
    const copyId = (id: string) => {
        navigator.clipboard.writeText(id);
        // 这里可以添加一个toast通知
    };

    // 运行单条数据
    const runSingleSample = (id: string) => {
        console.log(`正在运行样本: ${id}`);
        // 这里可以添加实际运行逻辑
    };

    // 检查项目是否被选中（考虑两种选择模式）
    const isItemSelected = (id: string) => {
        return selectedItems.has(id);
    };

    // 处理单个项目选择
    const handleSelectItem = (id: string) => {
        const newSelected = new Set(selectedItems);
        if (newSelected.has(id)) {
            newSelected.delete(id);
        } else {
            newSelected.add(id);
        }
        setSelectedItems(newSelected);
    };

    // 批量运行处理
    const handleBatchRun = () => {
        if (isAllSelected) {
            console.log(`正在运行所有 ${totalItems} 个匹配样本`);
            runAllMatchingSamples();
        } else {
            const selectedIds = Array.from(selectedItems);
            console.log(`正在运行 ${selectedIds.length} 个样本:`, selectedIds);
            runSelectedSamples(selectedIds);
        }
    };

    // 模拟运行所有匹配样本
    const runAllMatchingSamples = async () => {
        // 这里添加实际逻辑
        // 可能需要使用当前的搜索条件和过滤条件来获取所有匹配项
        console.log("正在运行所有匹配样本...");

        // 示例：调用API
        // await api.runSamples({ search: searchTerm, filter: currentFilter });
    };

    // 模拟运行选中样本
    const runSelectedSamples = async (sampleIds: string[]) => {
        // 这里添加实际逻辑
        console.log("正在运行选中样本:", sampleIds);

        // 示例：调用API
        // await api.runSamplesByIds(sampleIds);
    };

    const handleEyeClick = (sample: SampleItem) => {
        setSelectedSample(sample);
        setIsDetailOpen(true);
    };

    // 表格操作
    const handleAction = (action: string, id: string) => {
        console.log(`执行操作: ${action} - ${id}`);
        // 这里可以添加实际操作逻辑
        if (action === 'view') {
            const sample = datasets.find(s => s.id === id)
            if (sample) {
                setSelectedSample(sample)
                setIsDetailOpen(true)
            }
        } else if (action === 'delete') {
            // 处理删除操作
            // handleDelete(id)
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
                                        {evalId}
                                    </Button>
                                </BreadcrumbItem>
                                <BreadcrumbSeparator />
                                <BreadcrumbItem>
                                    <BreadcrumbPage>datasets</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                    <div className="flex justify-between items-center px-2">
                        <div>
                            <h1 className="text-2xl font-bold">数据集</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                数据集设置&上传
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
                                <CardTitle>数据集管理</CardTitle>
                                <p className="text-sm text-muted-foreground mt-1">
                                    管理您的问题/答案数据集
                                </p>
                            </div>

                            <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">
                                <div className="relative flex-1">
                                    <Input
                                        placeholder="搜索ID、问题或答案..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="pl-10"
                                    />
                                </div>

                                <div className="flex gap-2">
                                    <Button
                                        disabled={selectedItems.size === 0 && !isAllSelected}
                                        onClick={handleBatchRun}
                                    >
                                        <PlayIcon className="mr-2 h-4 w-4" />
                                        {isAllSelected ? `批量运行(所有${totalItems}项)` : `批量运行(${selectedItems.size}项)`}
                                    </Button>
                                    <Button variant="outline">
                                        <UploadIcon className="mr-2 h-4 w-4" /> 导入数据
                                    </Button>
                                </div>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="rounded-md border">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead className="w-[50px]">
                                                <Checkbox
                                                    checked={isAllSelected}
                                                    onCheckedChange={(checked) => {
                                                        if (checked) {
                                                            // When selecting all, populate selectedItems with all item IDs
                                                            setSelectedItems(new Set(allItems.map(item => item.id)));
                                                        } else {
                                                            // When deselecting all, clear selectedItems
                                                            setSelectedItems(new Set());
                                                        }
                                                    }}
                                                    aria-label="Select all"
                                                />
                                            </TableHead>
                                            <TableHead className="w-[15%]">样本ID</TableHead>
                                            <TableHead className="w-[40%]">问题</TableHead>
                                            <TableHead className="w-[30%]">答案</TableHead>
                                            <TableHead className="w-[100px] text-center">操作</TableHead>
                                        </TableRow>
                                    </TableHeader>

                                    <TableBody>
                                        {datasets.length === 0 ? (
                                            <TableRow>
                                                <TableCell colSpan={5} className="h-24 text-center">
                                                    暂无数据
                                                </TableCell>
                                            </TableRow>
                                        ) : (
                                            datasets.map((item) => (
                                                <TableRow key={item.id} className="hover:bg-muted/50 transition-colors">
                                                    <TableCell className="w-[50px]">
                                                        <Checkbox
                                                            checked={isItemSelected(item.id)}
                                                            onCheckedChange={() => handleSelectItem(item.id)}
                                                            aria-label="Select row"
                                                        />
                                                    </TableCell>
                                                    <TableCell className="font-medium">
                                                        <div className="flex items-center">
                                                            <span className="truncate max-w-[120px]" title={item.id}>
                                                                {item.id.substring(0, 20)}...
                                                            </span>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-6 w-6 ml-1"
                                                                onClick={() => copyId(item.id)}
                                                                title="复制样本ID"
                                                            >
                                                                <CopyIcon className="h-3 w-3" />
                                                            </Button>
                                                        </div>
                                                    </TableCell>

                                                    <TableCell
                                                        className="whitespace-normal break-words min-w-[250px] max-w-[400px] py-2"
                                                        style={{
                                                            whiteSpace: expandedRows.has(item.id) ? 'normal' : 'nowrap',
                                                            overflow: 'hidden',
                                                            textOverflow: 'ellipsis'
                                                        }}
                                                    >
                                                        {item.question}
                                                    </TableCell>

                                                    <TableCell className="whitespace-normal break-words min-w-[150px] max-w-[250px] py-2">
                                                        <Badge variant="secondary" className="bg-blue-50 text-blue-700 hover:bg-blue-100">
                                                            {item.answer}
                                                        </Badge>
                                                    </TableCell>
                                                    <TableCell className="text-right">
                                                        <Button
                                                            variant="link"
                                                            size="icon"
                                                            className="h-8 w-8"
                                                            onClick={() => toggleRowExpansion(item.id)}
                                                            title={expandedRows.has(item.id) ? "收起问题" : "展开问题"}
                                                        >
                                                            {expandedRows.has(item.id) ? (
                                                                <ChevronUp className="h-4 w-4" />
                                                            ) : (
                                                                <ChevronDown className="h-4 w-4" />
                                                            )}
                                                        </Button>
                                                        <Dialog open={isDetailOpen} onOpenChange={setIsDetailOpen}>
                                                            <DialogTrigger asChild>
                                                                <Button
                                                                    variant="link"
                                                                    size="icon"
                                                                    className="text-black-500 hover:text-black-700 px-1 py-1"
                                                                    onClick={() => handleEyeClick(item)}
                                                                    title="查看详情"
                                                                >
                                                                    <Eye className="mr-2 h-4 w-4" />
                                                                </Button>
                                                            </DialogTrigger>
                                                            <DialogContent className="max-w-2xl">
                                                                <DialogHeader>
                                                                    <DialogTitle>样本详情</DialogTitle>
                                                                </DialogHeader>
                                                                {selectedSample && (
                                                                    <div className="space-y-4">
                                                                        {/* Existing fields */}
                                                                        <div>
                                                                            <h4 className="text-sm font-medium text-muted-foreground">样本ID</h4>
                                                                            <p className="mt-1">{selectedSample.id}</p>
                                                                        </div>
                                                                        <div>
                                                                            <h4 className="text-sm font-medium text-muted-foreground">问题</h4>
                                                                            <p className="mt-1 whitespace-pre-wrap">{selectedSample.question}</p>
                                                                        </div>
                                                                        <div>
                                                                            <h4 className="text-sm font-medium text-muted-foreground">答案</h4>
                                                                            <Badge variant="secondary" className="mt-1 bg-blue-50 text-blue-700 hover:bg-blue-100 whitespace-pre-wrap">
                                                                                {selectedSample.answer}
                                                                            </Badge>
                                                                        </div>

                                                                        {/* New fields */}
                                                                        <div>
                                                                            <h4 className="text-sm font-medium text-muted-foreground">难度等级</h4>
                                                                            <p className="mt-1">
                                                                                {selectedSample.level ? (
                                                                                    <Badge>
                                                                                        {selectedSample.level}
                                                                                    </Badge>
                                                                                ) : "未指定"}
                                                                            </p>
                                                                        </div>

                                                                        <div>
                                                                            <h4 className="text-sm font-medium text-muted-foreground">使用工具</h4>
                                                                            <div className="mt-1 flex flex-wrap gap-2">
                                                                                {selectedSample.tools && selectedSample.tools.length > 0 ? (
                                                                                    selectedSample.tools.map((tool, index) => (
                                                                                        <Badge key={index} variant="outline">
                                                                                            {tool}
                                                                                        </Badge>
                                                                                    ))
                                                                                ) : (
                                                                                    <span className="text-muted-foreground">无</span>
                                                                                )}
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </DialogContent>
                                                        </Dialog>
                                                        <Button
                                                            variant="link"
                                                            size="icon"
                                                            className="h-8 w-8"
                                                            onClick={() => runSingleSample(item.id)}
                                                            title="运行单条"
                                                        >
                                                            <PlayIcon className="mr-2 h-4 w-4" />
                                                        </Button>
                                                        <Button
                                                            variant="link"
                                                            size="icon"
                                                            className="h-8 w-8"
                                                            title="删除"
                                                        >
                                                            <Trash2Icon className="mr-2 h-4 w-4" />
                                                        </Button>
                                                        {/* <DropdownMenu>
                                                    <DropdownMenuTrigger asChild>
                                                        <Button variant="ghost" className="h-8 w-8 p-0">
                                                            <span className="sr-only">打开菜单</span>
                                                            <MoreHorizontal className="h-4 w-4" />
                                                        </Button>
                                                    </DropdownMenuTrigger>
                                                    <DropdownMenuContent align="end">
                                                        <DropdownMenuItem onClick={() => runSingleSample(item.id)}>
                                                            <PlayIcon className="mr-2 h-4 w-4" />
                                                            运行单条
                                                        </DropdownMenuItem>
                                                        <DropdownMenuItem
                                                            className="text-red-600 focus:bg-red-100"
                                                            onClick={() => handleAction('delete', item.id)}
                                                        >
                                                            <Trash2Icon /> 删除
                                                        </DropdownMenuItem>
                                                    </DropdownMenuContent>
                                                </DropdownMenu> */}
                                                    </TableCell>
                                                </TableRow>
                                            ))
                                        )}
                                    </TableBody>
                                </Table>
                            </div>

                            {/* 选中项状态提示 */}
                            {(selectedItems.size > 0 || isAllSelected) && (
                                <div className="p-2 border-t bg-muted/50 mt-2 rounded-b-md">
                                    <div className="text-sm flex items-center justify-between">
                                        <span>
                                            {isAllSelected ? (
                                                `已选择所有匹配的 ${totalItems} 项`
                                            ) : (
                                                `已选择 ${selectedItems.size} 项`
                                            )}
                                        </span>
                                        <Button
                                            variant="link"
                                            className="p-0 h-auto font-normal text-muted-foreground hover:text-foreground"
                                            onClick={() => setSelectedItems(new Set())}
                                        >
                                            清除选择
                                        </Button>
                                    </div>
                                </div>
                            )}
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
