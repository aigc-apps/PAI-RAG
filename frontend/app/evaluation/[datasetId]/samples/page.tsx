'use client';

import React, { useState, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
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
    BookAIcon,
} from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogDescription,
    DialogFooter,
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
import { RunConfig } from '@/app/evaluation/[datasetId]/types';
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';

import { toast } from 'sonner';

import { SampleDetailDialog } from '@/app/evaluation/components/sample-detail-dialog';
import { useDatasetActions } from '@/app/evaluation/[datasetId]/samples/useDatasetActions';
import { SampleItem } from '@/app/evaluation/[datasetId]/types';

export default function EvalDatasetsDetailsPage({
    params,
}: {
    params: Promise<{ datasetId: string }>;
}) {
    const { datasetId } = use(params);
    const router = useRouter();

    // === 状态管理 ===
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [datasets, setDatasets] = useState<SampleItem[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const pageSize = 10;

    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
    const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
    const [allItems, setAllItems] = useState<SampleItem[]>([]);
    const [totalItems, setTotalItems] = useState(0);

    // 对话框状态
    const [isRunSingleDetailOpen, setIsRunSingleDetailOpen] = useState(false);
    const [isRunBatchDetailOpen, setIsRunBatchDetailOpen] = useState(false);
    const [isEditOpen, setIsEditOpen] = useState(false);
    const [editingSample, setEditingSample] = useState<SampleItem | null>(null);
    const [dialogMode, setDialogMode] = useState<'view' | 'edit'>('view');

    // 实验配置相关
    const [experimentName, setExperimentName] = useState("");
    const [experimentDescription, setExperimentDescription] = useState("");
    const [runConfigId, setRunConfigId] = useState<string>("");
    const [runConfigs, setRunConfigs] = useState<RunConfig[]>([]);
    const [evaluatorConfigId, setEvaluatorConfigId] = useState<string>("");
    const [evaluatorConfigs, setEvaluatorConfigs] = useState<EvaluatorConfig[]>([]);


    // 上传状态
    const [uploading, setUploading] = useState(false);
    const fileInputRef = React.useRef<HTMLInputElement>(null);

    const { runSamples, deleteSample, uploadFile } = useDatasetActions({ datasetId });

    // === 数据加载 ===
    useEffect(() => {
        const loadData = async () => {
            setIsLoading(true);
            try {
                const [datasetRes, runConfigsRes, evalConfigsRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${datasetId}/samples?page=${page}&size=${pageSize}`),
                    fetch(`/api/config/evaluation/${datasetId}/runconfigs`),
                    fetch(`/api/config/evaluation/${datasetId}/evalconfigs`),
                ]);

                // 加载分页数据
                if (datasetRes.ok) {
                    const data = await datasetRes.json();
                    setDatasets(data.data.items);
                    setTotalItems(data.data.total);
                    setTotalPages(data.data.pages);
                }

                // 加载配置
                if (runConfigsRes.ok) {
                    const configData = await runConfigsRes.json();
                    setRunConfigs(configData.data.items);
                }

                if (evalConfigsRes.ok) {
                    const evalConfigData = await evalConfigsRes.json();
                    setEvaluatorConfigs(evalConfigData.data.items);
                }

                // 加载所有数据（用于全选）
                const tmpPageSize = 1000;
                const firstPageRes = await fetch(`/api/config/evaluation/${datasetId}/samples?page=1&size=${tmpPageSize}`);
                if (!firstPageRes.ok) throw new Error('获取数据样本列表失败');
                const json_data = await firstPageRes.json();
                const tmpAllItems: SampleItem[] = [];

                for (let curPage = 1; curPage <= json_data.data.pages; curPage++) {
                    console.log("加载所有数据，第", curPage, "页");
                    const response = await fetch(`/api/config/evaluation/${datasetId}/samples?page=${curPage}&size=${tmpPageSize}`);
                    const data = await response.json();
                    tmpAllItems.push(...data.data.items);
                }
                setAllItems(tmpAllItems);


            } catch (err) {
                console.error(err);
                toast.error('加载数据失败');
            } finally {
                setIsLoading(false);
            }
        };

        loadData();
    }, [page, datasetId, datasets.length]);

    // === 交互函数 ===
    const handlePageChange = (newPage: number) => {
        if (newPage < 1 || newPage > totalPages) return;
        setPage(newPage);
    };

    const toggleRowExpansion = (id: string) => {
        setExpandedRows(prev => {
            const newSet = new Set(prev);
            newSet.has(id) ? newSet.delete(id) : newSet.add(id);
            return newSet;
        });
    };

    const copyId = (id: string) => {
        navigator.clipboard.writeText(id);
        toast.success('复制成功');
    };

    const isItemSelected = (id: string) => selectedItems.has(id);

    const handleSelectItem = (id: string) => {
        setSelectedItems(prev => {
            const newSet = new Set(prev);
            newSet.has(id) ? newSet.delete(id) : newSet.add(id);
            return newSet;
        });
    };

    const isAllSelected = selectedItems.size === allItems.length && allItems.length > 0;

    // === 运行逻辑 ===
    const handleBatchRun = async () => {
        const ids = isAllSelected
            ? allItems.map(item => item.id)
            : Array.from(selectedItems);

        await runSamples({
            name: experimentName,
            description: experimentDescription,
            sample_ids: ids,
            run_config_id: runConfigId,
            evaluator_config_id: evaluatorConfigId
        });

        setIsRunBatchDetailOpen(false);
        setExperimentName("");
        setExperimentDescription("");
    };

    const runSingleSample = async (id: string) => {
        await runSamples({
            name: experimentName,
            description: experimentDescription,
            sample_ids: [id],
            run_config_id: runConfigId,
            evaluator_config_id: evaluatorConfigId
        });
        setIsRunSingleDetailOpen(false);
        setExperimentName("");
        setExperimentDescription("");
    };

    // === 上传逻辑 ===
    const handleFileUpload = async (files: FileList | null) => {
        if (!files?.length) return;

        const file = files[0];
        if (file.size > 100 * 1024 * 1024) {
            toast.error("文件大小超过 100MB");
            return;
        }

        setUploading(true);
        try {
            const uploadedItems = await uploadFile(file);
            console.log('##handleFileUpload', uploadedItems);
            setDatasets(prev => [...prev, ...uploadedItems]);
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    // === 编辑/查看逻辑 ===
    const handleEyeClick = (item: SampleItem) => {
        setEditingSample(item);
        setDialogMode('view');
        setIsEditOpen(true);
    };

    const handleEditClick = (item: SampleItem) => {
        setEditingSample(item);
        setDialogMode('edit');
        setIsEditOpen(true);
    };

    const handleSaveEdit = async (updatedSample: SampleItem) => {
        try {
            const response = await fetch(`/api/config/evaluation/${datasetId}/dataset/${updatedSample.id}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(updatedSample),
            });

            if (!response.ok) throw new Error("更新失败");

            setDatasets(prev => prev.map(item => item.id === updatedSample.id ? updatedSample : item));
            toast.success('更新成功');
            setIsEditOpen(false);
        } catch (error) {
            toast.error('更新失败');
        }
    };

    // === 渲染辅助函数 ===
    const modifyRunConfig = (selected_ids: Set<string>) => (
        <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">名称</Label>
                <Input
                    id="name"
                    value={experimentName}
                    onChange={(e) => setExperimentName(e.target.value)}
                    className="col-span-3"
                    placeholder="请输入实验名称"
                />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="description" className="text-right">描述</Label>
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
                <Label className="text-right">数据样本ID</Label>
                <div className="col-span-2">
                    <div className="max-h-40 overflow-y-auto rounded-md border p-2 bg-muted/20">
                        <div className="flex flex-wrap gap-1.5">
                            {[...selected_ids].map(id => (
                                <Badge key={id} variant="secondary" className="bg-green-50 text-green-700">
                                    {id.substring(0, 8)}...
                                </Badge>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
                <Label className="text-right">运行设置</Label>
                <div className="grid gap-4 py-1">
                    {runConfigs.length > 0 ? (
                        <Select onValueChange={setRunConfigId}>
                            <SelectTrigger>
                                <SelectValue placeholder="请选择运行设置" />
                            </SelectTrigger>
                            <SelectContent>
                                {runConfigs.map(config => (
                                    <SelectItem key={config.id} value={config.id}>
                                        {config.name}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    ) : (
                        <p className="text-sm text-muted-foreground">尚未进行运行配置</p>
                    )}
                </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
                <Label className="text-right">评估器设置</Label>
                <div className="grid gap-4 py-1">
                    {evaluatorConfigs.length > 0 ? (
                        <Select onValueChange={setEvaluatorConfigId}>
                            <SelectTrigger>
                                <SelectValue placeholder="请选择评估设置" />
                            </SelectTrigger>
                            <SelectContent>
                                {evaluatorConfigs.map(config => (
                                    <SelectItem key={config.id} value={config.id}>
                                        {config.name}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    ) : (
                        <p className="text-sm text-muted-foreground">尚未进行评估器配置</p>
                    )}
                </div>
            </div>
        </div>
    );

    // // === UI 渲染 ===
    // if (isLoading) {
    //     return (
    //         <div className="flex items-center justify-center h-64">
    //             <Loader2 className="h-8 w-8 animate-spin" />
    //         </div>
    //     );
    // }

    return (
        <div className="flex flex-col h-full min-h-0">
            {/* 🆕 可复用对话框 */}
            <SampleDetailDialog
                open={isEditOpen}
                onOpenChange={setIsEditOpen}
                sample={editingSample}
                mode={dialogMode}
                onSave={dialogMode === 'edit' ? handleSaveEdit : undefined}
            />

            <Card className="flex flex-col h-full min-h-0 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                <CardHeader className="shrink-0 flex md:items-center md:justify-between">
                    <div>
                        <CardTitle className="text-2xl font-bold flex items-center gap-2">
                            <FileText className="h-5 w-5" /> 样本管理
                        </CardTitle>
                        <p className="text-sm text-muted-foreground mt-1">
                            管理您的问题/答案数据集。请选中样本运行实验。
                        </p>
                    </div>
                    <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
                        <div className="flex flex-col gap-2">
                            <div className="flex gap-2 justify-end">
                                {/* 批量运行 */}
                                <Dialog open={isRunBatchDetailOpen} onOpenChange={setIsRunBatchDetailOpen}>
                                    <DialogTrigger asChild>
                                        <Button disabled={selectedItems.size === 0 && !isAllSelected} className="text-white">
                                            <PlayIcon className="mr-2 h-4 w-4" />
                                            {isAllSelected ? `运行实验(所有${totalItems}项)` : `运行实验(${selectedItems.size}项)`}
                                        </Button>
                                    </DialogTrigger>
                                    <DialogContent className="sm:max-w-[750px]">
                                        <DialogHeader>
                                            <DialogTitle>创建新实验（批量）</DialogTitle>
                                            <DialogDescription>请输入此次实验名称和描述，然后运行试验。</DialogDescription>
                                        </DialogHeader>
                                        {modifyRunConfig(selectedItems)}
                                        <DialogFooter>
                                            <Button variant="outline" onClick={() => setIsRunBatchDetailOpen(false)}>
                                                取消
                                            </Button>
                                            <Button onClick={handleBatchRun} disabled={!experimentName.trim()} className="text-white">
                                                运行
                                            </Button>
                                        </DialogFooter>
                                    </DialogContent>
                                </Dialog>

                                {/* 上传 */}
                                <Button
                                    onClick={() => document.getElementById('file-upload')?.click()}
                                    disabled={uploading}
                                    className="text-white"
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
                                <input
                                    id="file-upload"
                                    type="file"
                                    className="hidden"
                                    ref={fileInputRef}
                                    onChange={(e) => handleFileUpload(e.target.files)}
                                />
                            </div>

                            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                                <div className="flex items-start gap-2">
                                    <Info className="h-4 w-4 text-gray-600 mt-0.5" />
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
                                <TableRow>
                                    <TableHead className="w-[50px]">
                                        <Checkbox
                                            checked={isAllSelected}
                                            onCheckedChange={(checked) => {
                                                setSelectedItems(checked ? new Set(allItems.map(item => item.id)) : new Set());
                                            }}
                                        />
                                    </TableHead>
                                    <TableHead className="w-[10%]">样本ID</TableHead>
                                    <TableHead className="w-[45%]">问题</TableHead>
                                    <TableHead className="w-[30%]">答案</TableHead>
                                    <TableHead className="w-[100px] text-center">操作</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-32 text-center">
                                            <div className="flex items-center justify-center space-x-4">
                                                <Loader2 className="h-6 w-6 animate-spin" />
                                                <h4 className="font-bold">Loading Datasets</h4>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : datasets.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-32 text-center">
                                            <div className="flex flex-col items-center gap-2 text-muted-foreground">
                                                <BookAIcon className="h-8 w-8" />
                                                <span>暂无数据样本，请上传文件</span>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    datasets.map((item) => (
                                        <TableRow key={item.id}>
                                            <TableCell>
                                                <Checkbox
                                                    checked={isItemSelected(item.id)}
                                                    onCheckedChange={() => handleSelectItem(item.id)}
                                                />
                                            </TableCell>
                                            <TableCell className="font-medium">
                                                <div className="flex">
                                                    <Button
                                                        variant="link"
                                                        className="truncate max-w-[120px] font-medium"
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
                                            <TableCell>
                                                <Badge variant="secondary" className="bg-green-50 text-green-700 whitespace-pre-wrap">
                                                    {item.expected_output}
                                                </Badge>
                                            </TableCell>
                                            <TableCell className="text-right">
                                                <div className="flex items-center justify-end gap-1">
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => toggleRowExpansion(item.id)}
                                                        title={expandedRows.has(item.id) ? "收起" : "展开"}
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
                                                        onClick={() => handleEyeClick(item)}
                                                        title="查看详情"
                                                    >
                                                        <Eye className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="text-blue-500"
                                                        onClick={() => handleEditClick(item)}
                                                        title="编辑"
                                                    >
                                                        <Pencil className="h-4 w-4" />
                                                    </Button>
                                                    {/* 单条运行 */}
                                                    <Dialog open={isRunSingleDetailOpen} onOpenChange={setIsRunSingleDetailOpen}>
                                                        <DialogTrigger asChild>
                                                            <Button variant="ghost" size="icon" title="运行单条">
                                                                <PlayIcon className="h-4 w-4" />
                                                            </Button>
                                                        </DialogTrigger>
                                                        <DialogContent className="sm:max-w-[750px]">
                                                            <DialogHeader>
                                                                <DialogTitle>创建新实验（单条）</DialogTitle>
                                                                <DialogDescription>请输入此次实验名称和描述，然后运行试验。</DialogDescription>
                                                            </DialogHeader>
                                                            {modifyRunConfig(new Set([item.id]))}
                                                            <DialogFooter>
                                                                <Button variant="outline" onClick={() => setIsRunSingleDetailOpen(false)}>
                                                                    取消
                                                                </Button>
                                                                <Button onClick={() => runSingleSample(item.id)} disabled={!experimentName.trim()} className="text-white">
                                                                    运行
                                                                </Button>
                                                            </DialogFooter>
                                                        </DialogContent>
                                                    </Dialog>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="text-red-500"
                                                        onClick={() => deleteSample(item.id)}
                                                        title="删除"
                                                    >
                                                        <Trash2Icon className="h-4 w-4" />
                                                    </Button>
                                                </div>
                                            </TableCell>
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </div>

                    {(selectedItems.size > 0 || isAllSelected) && (
                        <div className="p-3 border-t">
                            <div className="text-sm flex items-center justify-between">
                                <span className="font-medium">
                                    {isAllSelected ? `已选择所有 ${totalItems} 项` : `已选择 ${selectedItems.size} 项`}
                                </span>
                                <Button variant="link" className="p-0 h-auto" onClick={() => setSelectedItems(new Set())}>
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