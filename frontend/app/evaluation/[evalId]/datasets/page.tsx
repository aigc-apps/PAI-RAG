'use client';
import React from 'react';
import { useState, useEffect, use, useRef } from "react";
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
    ChevronDown,
    ChevronUp,
    PlayIcon,
    CopyIcon,
    UploadIcon,
    Eye,
    Trash2Icon,
    Loader2,
} from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogDescription,
    DialogFooter
} from "@/components/ui/dialog";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import { EvalRunConfig } from '@/app/evaluation/[evalId]/settings/page';
export interface SampleItem {
    id: string;
    input: string;
    expected_output: string;
    eval_metadata?: {
        Steps?: string;
        Tools?: string;
    };
}

export default function EvalDatasetsDetailsPage(
    { params }: { params: Promise<{ evalId: string }> }
) {
    const { evalId } = use(params);
    const router = useRouter();
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [datasets, setDatasets] = useState<SampleItem[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const pageSize = 8;
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
    const [isRunSingleDetailOpen, setIsRunSingleDetailOpen] = useState(false);
    const [isRunBatchDetailOpen, setIsRunBatchDetailOpen] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [singleRuning, setSingleRuning] = useState(false);
    const [batchRuning, setBatchRuning] = useState(false);
    const [dataseterror, setDatasetError] = useState('');
    const [evalRunConfigs, setEvalRunConfigs] = useState<EvalRunConfig[]>([]);
    const [experimentName, setExperimentName] = useState("");
    const [experimentDescription, setExperimentDescription] = useState("");
    const [evalRunConfigId, setEvalRunConfigId] = useState<string>("");


    useEffect(() => {
        const fetchConfigs = async () => {
            setIsLoading(true);
            try {
                const [datasetRes, configDataRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${evalId}/dataset?page=${page}&size=${pageSize}`),
                    fetch(`/api/config/evaluation/${evalId}/configs?page=${page}&size=${pageSize}`),
                ]);


                if (!datasetRes.ok) throw new Error('获取评估任务列表失败');
                const json_data = await datasetRes.json();
                console.log("evaluation dataset json_data", json_data)
                const data = json_data.data.items;

                setDatasets(data);
                setTotalItems(json_data.data.total);
                setTotalPages(json_data.data.pages);

                const tmpAllItems = [];
                for (let curPage = 1; curPage <= json_data.data.pages; curPage++) {
                    console.log("start loading all items for page ", curPage)
                    const response = await fetch(`/api/config/evaluation/${evalId}/dataset?page=${curPage}&size=${pageSize}`);
                    const data = await response.json();
                    tmpAllItems.push(...data.data.items);
                }
                setAllItems(tmpAllItems);
                console.log("finish loading all items", tmpAllItems.length)


                if (!configDataRes.ok) throw new Error('获取评估任务列表失败');
                const config_json_data = await configDataRes.json();
                console.log("evaluation run_config json_data", config_json_data)
                setEvalRunConfigs(config_json_data.data.items);

            } catch (err: any) {
                setDatasetError(err || '加载数据集失败');
            } finally {
                setIsLoading(false);
            }
        };
        fetchConfigs();
    }, [searchTerm, page, datasets.length]);

    const handlePageChange = (newPage: number) => {
        if (newPage < 1 || newPage > totalPages) return;
        setPage(newPage);
    };

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

    const runSamples = async (ids: string[]) => {
        console.log(`Start running ${ids.length} samples`)
        const data = {
            name: experimentName,
            description: experimentDescription,
            dataset_ids: ids,
            run_config_id: evalRunConfigId
        };
        try {
            const res = await fetch(`/api/config/evaluation/${evalId}/experiments`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (!res.ok) {
                alert('实验创建失败');
                return;
            }
            const upload_result = await res.json();
            console.log('实验创建成功:', upload_result);
            router.push(`/evaluation/${evalId}/${upload_result.data.id}`);
        } catch (error) {
            console.error('实验创建失败:', error);
        } finally {
            setExperimentName("");
            setExperimentDescription("");
        }
    }
    // 运行单条数据
    const runSingleSample = async (id: string) => {
        console.log(`正在运行单条数据: ${id}`);
        setSingleRuning(true);
        runSamples([id]);
        setSingleRuning(false);
        setIsRunSingleDetailOpen(false);
    };

    // 批量运行处理
    const handleBatchRun = () => {
        if (isAllSelected) {
            console.log(`正在运行所有 ${totalItems} 个匹配样本`);
            const allSelectedIds = allItems.map(item => item.id);
            setBatchRuning(true);
            runSamples(allSelectedIds);
            setBatchRuning(false);
            setIsRunBatchDetailOpen(false);
        } else {
            const selectedIds = Array.from(selectedItems);
            console.log(`正在运行 ${selectedIds.length} 个样本:`);
            setBatchRuning(true);
            runSamples(selectedIds);
            setBatchRuning(false);
            setIsRunBatchDetailOpen(false);
        }
    };

    const handleEyeClick = (sample: SampleItem) => {
        setSelectedSample(sample);
        setIsDetailOpen(true);
    };

    const handleFileUpload = async (files: FileList | null) => {
        console.log('##handleFileUpload', files);
        if (!files) {
            alert('文件列表为空！');
            return;
        }
        setUploading(true);

        // 文件校验 (Demo功能，后续调整优化)
        const validFiles = Array.from(files).filter((file) => {
            const isValidSize = file.size <= 100 * 1024 * 1024;
            return isValidSize;
        });

        if (validFiles.length === 0) {
            alert("请选择有效的文件（如 PDF 或 Word，且小于 100MB）");
            setUploading(false);
            return;
        }

        // 上传文件
        const formData = new FormData();
        formData.append('file', validFiles[0]); // 目前仅上传第一个文件，后续支持多文件

        try {
            const res = await fetch(
                `/api/config/evaluation/${evalId}/dataset`,
                {
                    method: 'POST',
                    body: formData,
                },
            );
            if (!res.ok) {
                alert('上传失败');
                return;
            }
            const upload_result = await res.json();
            console.log('上传成功:', upload_result);
            setDatasets((prev) => [...prev, ...upload_result.data]); // 追加新配置
        } catch (error) {
            console.error('上传失败:', error);
        } finally {
            setUploading(false);
            // 清空文件选择框
            if (fileInputRef.current) {
                fileInputRef.current.value = ''; // 清空 input 的值
            }
        }
    };


    const modifyEvalRunConfig = (selected_ids: Set<string>) => {
        return (
            <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="name" className="text-right">
                        名称
                    </Label>
                    <Input
                        id="name"
                        value={experimentName}
                        onChange={(e) => setExperimentName(e.target.value)}
                        className="col-span-3"
                        placeholder="请输入实验名称"
                    />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="description" className="text-right">
                        描述
                    </Label>
                    <Textarea
                        id="description"
                        value={experimentDescription}
                        onChange={(e) => setExperimentDescription(e.target.value)}
                        className="col-span-3"
                        placeholder="请输入实验描述"
                        rows={3}
                    />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="description" className="text-right">
                        数据样本ID
                    </Label>
                    <div className="col-span-2">
                        <div className="max-h-40 overflow-y-auto rounded-md border p-2 bg-muted/20">
                            <div className="flex flex-wrap gap-1.5">
                                {[...selected_ids].map((select_id: string) => (
                                    <Badge key={select_id} variant="secondary" className="bg-green-50 text-green-700 hover:bg-green-100 whitespace-pre-wrap mr-1 mb-1">
                                        {select_id}
                                    </Badge>
                                ))}
                            </div>
                        </div>

                    </div>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="description" className="text-right">
                        实验设置
                    </Label>
                    <div className="grid gap-4 py-1">
                        {evalRunConfigs.length > 0 ? (
                            <Select
                                onValueChange={(value) =>
                                    setEvalRunConfigId(value)
                                }>
                                <SelectTrigger>
                                    <SelectValue placeholder="请选择实验设置" />
                                </SelectTrigger>
                                <SelectContent>
                                    {evalRunConfigs.map((config) => (
                                        <SelectItem key={config.id} value={config.id}>
                                            {config.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        ) : (
                            <div className="flex flex-col gap-2">
                                <p className="text-sm text-muted-foreground">尚未进行实验配置</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col py-4 space-y-6">
            <div className="w-full">
                <Card className="w-full">
                    <CardHeader className="flex flex-col md:flex-row md:items-center md:justify-between md:space-y-0">
                        <div>
                            <CardTitle>数据集管理</CardTitle>
                            <p className="text-sm text-muted-foreground mt-1">
                                管理您的问题/答案数据集
                            </p>
                        </div>

                        <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">
                            <div className="flex flex-col gap-2">
                                <div className="flex gap-2 justify-end">
                                    <Dialog open={isRunBatchDetailOpen} onOpenChange={setIsRunBatchDetailOpen}>
                                        <DialogTrigger asChild>
                                            <Button
                                                disabled={selectedItems.size === 0 && !isAllSelected}
                                            >
                                                <PlayIcon className="mr-2 h-4 w-4" />
                                                {isAllSelected ? `运行实验(所有${totalItems}项)` : `运行实验(${selectedItems.size}项)`}
                                            </Button>
                                        </DialogTrigger>
                                        <DialogContent className="sm:max-w-[750px]">
                                            <DialogHeader>
                                                <DialogTitle>创建新实验（批量）</DialogTitle>
                                                <DialogDescription>
                                                    请输入此次实验名称和描述，然后运行试验。
                                                </DialogDescription>
                                            </DialogHeader>
                                            {modifyEvalRunConfig(selectedItems)}
                                            <DialogFooter>
                                                <Button
                                                    variant="outline"
                                                    onClick={() => setIsRunBatchDetailOpen(false)}
                                                    disabled={batchRuning}
                                                >
                                                    取消
                                                </Button>
                                                <Button
                                                    onClick={handleBatchRun}
                                                    disabled={!experimentName.trim()}
                                                >
                                                    运行 {batchRuning && <Loader2 className="ml-2 h-4 w-4 animate-spin" />}
                                                </Button>
                                            </DialogFooter>
                                        </DialogContent>
                                    </Dialog>
                                    <Button
                                        onClick={() =>
                                            document.getElementById('file-upload')?.click()
                                        }
                                        disabled={uploading} // 上传时禁用按钮
                                    >
                                        {uploading ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                上传中...
                                            </>
                                        ) : (
                                            <>
                                                <UploadIcon className="mr-2 h-4 w-4" /> 导入数据
                                            </>
                                        )}
                                    </Button>
                                    <div className="flex gap-2 items-center">
                                        <input
                                            id="file-upload"
                                            type="file"
                                            className="hidden"
                                            ref={fileInputRef}
                                            onChange={(e) => handleFileUpload(e.target.files)}
                                        />
                                    </div>
                                </div>
                                <p className="text-xs text-muted-foreground ml-2">
                                    数据要求为JSONL文件，每行需要包含 input(string), expected_output(string) 和 metadata(dict, 可选)
                                </p>
                            </div>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="rounded-md border overflow-y-auto">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead className="w-[50px]">
                                            <Checkbox
                                                checked={isAllSelected}
                                                onCheckedChange={(checked) => {
                                                    if (checked) {
                                                        setSelectedItems(new Set(allItems.map(item => item.id)));
                                                    } else {
                                                        setSelectedItems(new Set());
                                                    }
                                                }}
                                                aria-label="Select all"
                                            />
                                        </TableHead>
                                        <TableHead className="w-[10%]">样本ID</TableHead>
                                        <TableHead className="w-[45%]">问题</TableHead>
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
                                                        <Button
                                                            variant="link"
                                                            className="truncate max-w-[120px] text-blue-600"
                                                            onClick={() => handleEyeClick(item)}
                                                        >
                                                            {item.id.substring(0, 20)}...
                                                        </Button>
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
                                                    {item.input}
                                                </TableCell>

                                                <TableCell className="whitespace-normal break-words min-w-[150px] max-w-[250px] py-2">
                                                    <Badge variant="secondary" className="bg-green-50 text-green-700 hover:bg-green-100 whitespace-pre-wrap">
                                                        {item.expected_output}
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
                                                                        <p className="mt-1 whitespace-pre-wrap">{selectedSample.input}</p>
                                                                    </div>
                                                                    <div>
                                                                        <h4 className="text-sm font-medium text-muted-foreground">答案</h4>
                                                                        <Badge variant="secondary" className="mt-1 bg-blue-50 text-blue-700 hover:bg-blue-100 whitespace-pre-wrap">
                                                                            {selectedSample.expected_output}
                                                                        </Badge>
                                                                    </div>

                                                                    {/* New fields */}
                                                                    <div>
                                                                        <h4 className="text-sm font-medium text-muted-foreground">步骤</h4>
                                                                        <p className="mt-1">
                                                                            {selectedSample.eval_metadata?.Steps ? (
                                                                                <Badge className='bg-green-50 text-green-700 hover:bg-green-100 whitespace-pre-wrap'>
                                                                                    {selectedSample.eval_metadata.Steps}
                                                                                </Badge>
                                                                            ) : "未指定"}
                                                                        </p>
                                                                    </div>

                                                                    <div>
                                                                        <h4 className="text-sm font-medium text-muted-foreground">使用工具</h4>
                                                                        <div className="mt-1 flex flex-wrap gap-2">
                                                                            {selectedSample.eval_metadata?.Tools ? (
                                                                                <Badge className='bg-yellow-50 text-yellow-700 hover:bg-yellow-100 whitespace-pre-wrap'>
                                                                                    {selectedSample.eval_metadata?.Tools}
                                                                                </Badge>
                                                                            ) : (
                                                                                <span className="text-muted-foreground">无</span>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </DialogContent>
                                                    </Dialog>
                                                    <Dialog open={isRunSingleDetailOpen} onOpenChange={setIsRunSingleDetailOpen}>
                                                        <DialogTrigger asChild>
                                                            <Button
                                                                variant="link"
                                                                size="icon"
                                                                className="text-black-500 hover:text-black-700 px-1 py-1"
                                                                title="运行单条"
                                                            >
                                                                <PlayIcon className="mr-2 h-4 w-4" />
                                                            </Button>
                                                        </DialogTrigger>
                                                        <DialogContent className="sm:max-w-[750px]">
                                                            <DialogHeader>
                                                                <DialogTitle>创建新实验（单条）</DialogTitle>
                                                                <DialogDescription>
                                                                    请输入此次实验名称和描述，然后运行试验。
                                                                </DialogDescription>
                                                            </DialogHeader>
                                                            {modifyEvalRunConfig(new Set([item.id]))}
                                                            <DialogFooter>
                                                                <Button
                                                                    variant="outline"
                                                                    onClick={() => setIsRunSingleDetailOpen(false)}
                                                                    disabled={singleRuning}
                                                                >
                                                                    取消
                                                                </Button>
                                                                <Button
                                                                    onClick={() => runSingleSample(item.id)}
                                                                    disabled={!experimentName.trim()}
                                                                >
                                                                    运行 {singleRuning && <Loader2 className="ml-2 h-4 w-4 animate-spin" />}
                                                                </Button>
                                                            </DialogFooter>
                                                        </DialogContent>
                                                    </Dialog>
                                                    <Button
                                                        variant="link"
                                                        size="icon"
                                                        className="h-8 w-8"
                                                        title="删除"
                                                    >
                                                        <Trash2Icon className="mr-2 h-4 w-4" />
                                                    </Button>
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
                        <div className="">
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
