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
    Pencil,
    FileText,
    Info,
    MessageSquare,
    CheckCircle,
    Tag,
    Plus
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
import { toast } from 'sonner';

export interface SampleItem {
    id: string;
    input: string;
    expected_output: string;
    eval_metadata?: Record<string, any>;
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
    const pageSize = 10;
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
    // 用于跟踪选中的行
    const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
    const [allItems, setAllItems] = useState<SampleItem[]>([]);
    const isAllSelected = selectedItems.size === allItems.length && allItems.length > 0;
    const [totalItems, setTotalItems] = useState(0);
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
    const [dialogMode, setDialogMode] = useState<'view' | 'edit'>('view'); // 新增状态
    const [isEditOpen, setIsEditOpen] = useState(false);
    const [editingSample, setEditingSample] = useState<SampleItem | null>(null);
    const [editedInput, setEditedInput] = useState("");
    const [editedOutput, setEditedOutput] = useState("");
    const [editedMetadata, setEditedMetadata] = useState<Record<string, any>>({});

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

        const fetchAllItems = async () => {
            setIsLoading(true);
            try {
                const tmpPageSize = 1000;
                const [datasetRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${evalId}/dataset?page=1&size=${tmpPageSize}`),
                ]);


                if (!datasetRes.ok) throw new Error('获取评估任务列表失败');
                const json_data = await datasetRes.json();
                console.log("evaluation dataset json_data", json_data)

                const tmpAllItems = [];
                for (let curPage = 1; curPage <= json_data.data.pages; curPage++) {
                    console.log("start loading all items for page ", curPage)
                    const tmpPageSize = 1000;
                    const response = await fetch(`/api/config/evaluation/${evalId}/dataset?page=${curPage}&size=${tmpPageSize}`);
                    const data = await response.json();
                    tmpAllItems.push(...data.data.items);
                }
                setAllItems(tmpAllItems);
                console.log("finish loading all items", tmpAllItems.length)

            } catch (err: any) {
                setDatasetError(err || '加载数据集失败');
            } finally {
                setIsLoading(false);
            }
        };

        fetchConfigs();
        fetchAllItems();
    }, [page, datasets.length]);

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
            toast.success('实验创建成功');
            router.push(`/evaluation/${evalId}/${upload_result.data.id}`);
        } catch (error) {
            console.error('实验创建失败:', error);
            toast.error('实验创建失败');
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
            setDatasets((prev) => [...prev, ...upload_result.data]);
            toast.success('文件上传成功');
        } catch (error) {
            console.error('上传失败:', error);
            toast.error(`上传失败 ${error}`);
        } finally {
            setUploading(false);
            // 清空文件选择框
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    // 删除样本 
    const handleDeleteClick = async (item: SampleItem) => {
        try {
            const res = await fetch(`/api/config/evaluation/${evalId}/dataset/${item.id}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!res.ok) {
                throw new Error('删除样本失败，请检查网络或配置');
            }
            // 删除成功后更新本地状态
            setDatasets((prev) => prev.filter((dataset) => dataset.id !== item.id));
            toast.success('删除样本成功');
        } catch (err: any) {
            console.log('删除评估任务出错: ', err);
            toast.error('删除样本失败');
        }
    };
    // 查看样本
    const handleEyeClick = (item: SampleItem) => {
        setEditingSample(item);
        setEditedInput(item.input);
        setEditedOutput(item.expected_output);
        setEditedMetadata(item.eval_metadata || {}); // 初始化 metadata
        setDialogMode('view');
        setIsEditOpen(true);
    };
    // 编辑样本
    const handleEditClick = (item: SampleItem) => {
        setEditingSample(item);
        setEditedInput(item.input);
        setEditedOutput(item.expected_output);
        setEditedMetadata(item.eval_metadata || {}); // 初始化 metadata
        setDialogMode('edit');
        setIsEditOpen(true);
    };

    // 保存编辑
    const handleSaveEdit = async () => {
        if (!editingSample) return;

        try {
            const updatedSample = {
                ...editingSample,
                input: editedInput,
                expected_output: editedOutput,
                eval_metadata: Object.keys(editedMetadata).length > 0 ? editedMetadata : undefined,
            };

            // 调用 API 更新样本
            const response = await fetch(`/api/config/evaluation/${evalId}/dataset/${editingSample.id}`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(updatedSample),
            });

            if (!response.ok) {
                throw new Error("更新失败");
            }

            // 更新本地状态
            setDatasets(prev =>
                prev.map(item =>
                    item.id === editingSample.id ? updatedSample : item
                )
            );
            toast.success('样本信息更新成功');
            setIsEditOpen(false);
        } catch (error) {
            console.error("更新失败:", error);
            toast.error(`样本信息更新失败 ${error}`);
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
        <div className="flex flex-col h-full min-h-0">
            <Card className="flex flex-col h-full min-h-0 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                <CardHeader className="shrink-0 flex md:items-center md:justify-between">
                    <div>
                        <CardTitle className="text-2xl font-bold flex items-center gap-2">
                            <FileText className="h-5 w-5" /> 样本管理
                        </CardTitle>
                        <p className="text-sm text-muted-foreground mt-1">
                            管理您的问题/答案数据集。 请选中样本运行实验。
                        </p>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
                        <div className="flex flex-col gap-2">
                            <div className="flex gap-2 justify-end">
                                <Dialog open={isRunBatchDetailOpen} onOpenChange={setIsRunBatchDetailOpen}>
                                    <DialogTrigger asChild>
                                        <Button
                                            disabled={selectedItems.size === 0 && !isAllSelected}
                                            className="text-white shadow-md hover:shadow-lg transition-all"
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
                                                className="text-white"
                                            >
                                                运行 {batchRuning && <Loader2 className="ml-2 h-4 w-4 animate-spin" />}
                                            </Button>
                                        </DialogFooter>
                                    </DialogContent>
                                </Dialog>
                                <Button
                                    onClick={() => document.getElementById('file-upload')?.click()}
                                    disabled={uploading}
                                    className="text-white shadow-md hover:shadow-lg transition-all"
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
                            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                                <div className="flex items-start gap-2">
                                    <Info className="h-4 w-4 text-gray-600 mt-0.5 flex-shrink-0" />
                                    <p className="text-xs text-gray-800">
                                        <span className="font-medium">数据要求：</span>
                                        JSONL文件，每行需包含 input(string), expected_output(string) 和 metadata(dict, 可选)
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 overflow-y-auto p-0">
                    <div className="rounded-md h-full min-h-0">
                        <Table className='rounded-md border'>
                            <TableHeader>
                                <TableRow className="transition-colors">
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
                                        <TableCell colSpan={5} className="h-32 text-center">
                                            <div className="flex flex-col items-center gap-2 text-muted-foreground">
                                                <FileText className="h-8 w-8" />
                                                <span>暂无数据</span>
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    onClick={() => document.getElementById('file-upload')?.click()}
                                                    className="mt-2"
                                                >
                                                    <UploadIcon className="mr-1 h-3 w-3" /> 导入数据
                                                </Button>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    datasets.map((item) => (
                                        <TableRow key={item.id} className="transition-colors group">
                                            <TableCell className="w-[50px]">
                                                <Checkbox
                                                    checked={isItemSelected(item.id)}
                                                    onCheckedChange={() => handleSelectItem(item.id)}
                                                    aria-label="Select row"
                                                />
                                            </TableCell>
                                            <TableCell className="font-medium">
                                                <div className="flex">
                                                    <Button
                                                        variant="link"
                                                        className="truncate max-w-[120px] font-medium group-hover:underline"
                                                        onClick={() => handleEyeClick(item)}
                                                    >
                                                        {item.id.substring(0, 12)}...
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-6 w-6"
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
                                                <Badge variant="secondary" className="bg-green-50 text-green-700 hover:bg-green-100 whitespace-pre-wrap border-green-200">
                                                    {item.expected_output}
                                                </Badge>
                                            </TableCell>
                                            <TableCell className="text-right">
                                                <div className="flex items-center justify-end gap-1">
                                                    <Button
                                                        variant="ghost"
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
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleEyeClick(item);
                                                        }}
                                                        title="查看详情"
                                                    >
                                                        <Eye className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-8 w-8 text-blue-500 hover:text-blue-700 hover:bg-blue-100"
                                                        title="编辑"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleEditClick(item);
                                                        }}
                                                    >
                                                        <Pencil className="h-4 w-4" />
                                                    </Button>

                                                    <Dialog open={isRunSingleDetailOpen} onOpenChange={setIsRunSingleDetailOpen}>
                                                        <DialogTrigger asChild>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                title="运行单条"
                                                            >
                                                                <PlayIcon className="h-4 w-4" />
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
                                                                    className="text-white"
                                                                >
                                                                    运行 {singleRuning && <Loader2 className="ml-2 h-4 w-4 animate-spin" />}
                                                                </Button>
                                                            </DialogFooter>
                                                        </DialogContent>
                                                    </Dialog>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-8 w-8 text-red-500 hover:text-red-700 hover:bg-red-100"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            handleDeleteClick(item);
                                                        }}
                                                        title="删除"
                                                    >
                                                        <Trash2Icon className="h-4 w-4" />
                                                    </Button>
                                                    <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
                                                        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                                                            <DialogHeader>
                                                                <DialogTitle className="flex items-center gap-2">
                                                                    {dialogMode === 'view' ? (
                                                                        <>
                                                                            <Eye className="h-5 w-5 text-blue-500" /> 查看样本
                                                                        </>
                                                                    ) : (
                                                                        <>
                                                                            <Pencil className="h-5 w-5 text-green-500" /> 编辑样本
                                                                        </>
                                                                    )}
                                                                </DialogTitle>
                                                            </DialogHeader>

                                                            {editingSample && (
                                                                <div className="space-y-4 py-4">
                                                                    {/* 问题 */}
                                                                    <div className="space-y-2">
                                                                        <Label htmlFor="edit-input" className="flex items-center gap-1">
                                                                            <MessageSquare className="h-4 w-4 text-blue-500" />
                                                                            问题
                                                                        </Label>
                                                                        {dialogMode === 'view' ? (
                                                                            <div className="p-3 bg-muted rounded-md border">
                                                                                <p className="whitespace-pre-wrap">{editedInput}</p>
                                                                            </div>
                                                                        ) : (
                                                                            <Textarea
                                                                                id="edit-input"
                                                                                value={editedInput}
                                                                                onChange={(e) => setEditedInput(e.target.value)}
                                                                                placeholder="请输入问题"
                                                                                className="min-h-[80px]"
                                                                            />
                                                                        )}
                                                                    </div>

                                                                    {/* 答案 */}
                                                                    <div className="space-y-2">
                                                                        <Label htmlFor="edit-output" className="flex items-center gap-1">
                                                                            <CheckCircle className="h-4 w-4 text-green-500" />
                                                                            答案
                                                                        </Label>
                                                                        {dialogMode === 'view' ? (
                                                                            <div className="p-3 bg-green-50 rounded-md border border-green-200">
                                                                                <p className="whitespace-pre-wrap text-green-800">{editedOutput}</p>
                                                                            </div>
                                                                        ) : (
                                                                            <Textarea
                                                                                id="edit-output"
                                                                                value={editedOutput}
                                                                                onChange={(e) => setEditedOutput(e.target.value)}
                                                                                placeholder="请输入预期答案"
                                                                                className="min-h-[80px]"
                                                                            />
                                                                        )}
                                                                    </div>

                                                                    {/* 动态 Metadata */}
                                                                    <div className="space-y-4">
                                                                        <div className="flex items-center gap-2">
                                                                            <Tag className="h-4 w-4 text-purple-500" />
                                                                            <h3 className="text-sm font-medium">元数据</h3>
                                                                            {dialogMode === 'edit' && (
                                                                                <Button
                                                                                    type="button"
                                                                                    variant="outline"
                                                                                    size="sm"
                                                                                    onClick={() => {
                                                                                        // 添加新字段
                                                                                        setEditedMetadata(prev => ({
                                                                                            ...prev,
                                                                                            [`新字段${Object.keys(prev).length + 1}`]: ""
                                                                                        }));
                                                                                    }}
                                                                                    className="ml-auto"
                                                                                >
                                                                                    <Plus className="h-3 w-3 mr-1" /> 添加字段
                                                                                </Button>
                                                                            )}
                                                                        </div>

                                                                        {dialogMode === 'view' ? (
                                                                            // 查看模式：展示所有 metadata
                                                                            editingSample.eval_metadata && Object.keys(editingSample.eval_metadata).length > 0 ? (
                                                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                                                                    {Object.entries(editingSample.eval_metadata).map(([key, value]) => (
                                                                                        <div key={key} className="p-3 bg-purple-50 rounded-md border border-purple-200">
                                                                                            <div className="text-xs font-medium text-purple-600 mb-1">{key}</div>
                                                                                            <div className="text-sm text-purple-800 break-words">
                                                                                                {value !== null && value !== undefined ? String(value) : '空值'}
                                                                                            </div>
                                                                                        </div>
                                                                                    ))}
                                                                                </div>
                                                                            ) : (
                                                                                <div className="p-3 bg-muted rounded-md border text-center text-muted-foreground">
                                                                                    无元数据
                                                                                </div>
                                                                            )
                                                                        ) : (
                                                                            // 编辑模式：可编辑所有 metadata
                                                                            editedMetadata && Object.keys(editedMetadata).length > 0 ? (
                                                                                <div className="space-y-3">
                                                                                    {Object.entries(editedMetadata).map(([key, value]) => (
                                                                                        <div key={key} className="flex gap-2 items-start p-2 bg-muted/50 rounded-md">
                                                                                            <Input
                                                                                                value={key}
                                                                                                onChange={(e) => {
                                                                                                    const newMetadata = { ...editedMetadata };
                                                                                                    delete newMetadata[key];
                                                                                                    newMetadata[e.target.value] = value;
                                                                                                    setEditedMetadata(newMetadata);
                                                                                                }}
                                                                                                placeholder="字段名"
                                                                                                className="w-1/3 text-sm"
                                                                                            />
                                                                                            <Input
                                                                                                value={value !== null && value !== undefined ? String(value) : ''}
                                                                                                onChange={(e) => {
                                                                                                    const newMetadata = { ...editedMetadata };
                                                                                                    newMetadata[key] = e.target.value;
                                                                                                    setEditedMetadata(newMetadata);
                                                                                                }}
                                                                                                placeholder="字段值"
                                                                                                className="flex-1 text-sm"
                                                                                            />
                                                                                            <Button
                                                                                                type="button"
                                                                                                variant="ghost"
                                                                                                size="icon"
                                                                                                className="h-8 w-8 text-red-500 hover:text-red-700 hover:bg-red-100"
                                                                                                onClick={() => {
                                                                                                    const newMetadata = { ...editedMetadata };
                                                                                                    delete newMetadata[key];
                                                                                                    setEditedMetadata(newMetadata);
                                                                                                }}
                                                                                            >
                                                                                                <Trash2Icon className="h-4 w-4" />
                                                                                            </Button>
                                                                                        </div>
                                                                                    ))}
                                                                                </div>
                                                                            ) : (
                                                                                <div className="p-3 bg-muted/50 rounded-md border-dashed border text-center text-muted-foreground">
                                                                                    点击“添加字段”按钮添加元数据
                                                                                </div>
                                                                            )
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            )}

                                                            <DialogFooter>
                                                                <Button
                                                                    variant="outline"
                                                                    onClick={() => setIsEditOpen(false)}
                                                                >
                                                                    关闭
                                                                </Button>
                                                                {dialogMode === 'edit' && (
                                                                    <Button
                                                                        onClick={handleSaveEdit}
                                                                        className="bg-green-600 hover:bg-green-700 text-white"
                                                                    >
                                                                        保存
                                                                    </Button>
                                                                )}
                                                            </DialogFooter>
                                                        </DialogContent>
                                                    </Dialog>
                                                </div>
                                            </TableCell>
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </div>

                    {/* 选中项状态提示 */}
                    {(selectedItems.size > 0 || isAllSelected) && (
                        <div className="p-3 border-t mt-2 rounded-b-md">
                            <div className="text-sm flex items-center justify-between">
                                <span className="font-medium">
                                    {isAllSelected ? (
                                        `已选择所有匹配的 ${totalItems} 项`
                                    ) : (
                                        `已选择 ${selectedItems.size} 项`
                                    )}
                                </span>
                                <Button
                                    variant="link"
                                    className="p-0 h-auto font-normal"
                                    onClick={() => setSelectedItems(new Set())}
                                >
                                    清除选择
                                </Button>
                            </div>
                        </div>
                    )}
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