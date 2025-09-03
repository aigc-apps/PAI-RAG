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
    Loader2,
    ChevronDownIcon
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
import {
    DropdownMenu,
    DropdownMenuCheckboxItem,
    DropdownMenuContent,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Switch } from '@/components/ui/switch';
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import { EvalConfig } from '@/app/evaluation/[evalId]/page';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';

export interface SampleItem {
    id: string;
    input: string;
    expected_output: string;
    eval_metadata?: {
        Steps?: string;
        Tools?: string;
    };
}

interface EvalRunConfig {
    model_id: string;
    mcp_ids: string[];
    kb_ids: string[];
    enable_search: boolean;
    enable_vision: boolean;
    enable_agent: boolean;
    enable_input_guardrail?: boolean;
    enable_output_guardrail?: boolean;
    guardrail_hint?: string;
}

const default_eval_run_config = {
    model_id: "",
    mcp_ids: [],
    kb_ids: [],
    enable_search: false,
    enable_vision: false,
    enable_agent: false,
    enable_input_guardrail: false,
    enable_output_guardrail: false,
    guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
};

export default function EvalExpDetailsPage(
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
    const [evalConfig, setEvalConfig] = useState<EvalConfig>();
    const [evalRunConfig, setEvalRunConfig] = useState<EvalRunConfig>(default_eval_run_config);
    const [experimentName, setExperimentName] = useState("");
    const [experimentDescription, setExperimentDescription] = useState("");
    const [llms, setLlms] = useState<LlmConfig[]>([]);
    const [mcps, setMcps] = useState<McpConfig[]>([]);
    const [kbs, setKbs] = useState<KbConfig[]>([]);
    const [selectedKbNames, setSelectedKbNames] = useState<string[]>([]);
    const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>([]);

    useEffect(() => {
        const fetchBasicConfigs = async () => {
            setIsLoading(true);
            try {
                const [llmRes, mcpRes, kbRes] = await Promise.all([
                    fetch(`/api/config/llms`),
                    fetch(`/api/config/mcps`),
                    fetch(`/api/config/knowledgebases`),
                ]);

                const llmData = (await llmRes.json())?.data.items || [];
                console.log('llmData', llmData);
                setLlms([...llmData]);

                const mcpData =
                    ((await mcpRes.json())?.data.items as McpConfig[]) || [];
                console.log('mcpData', mcpData);
                setMcps([...mcpData]);

                const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
                console.log('kbData', kbData);
                setKbs([...kbData]);
            } catch (err: any) {
                setDatasetError(err || '加载数据失败');
            } finally {
                setIsLoading(false);
            }
        };

        fetchBasicConfigs();
    }, []);

    useEffect(() => {
        const fetchConfigs = async () => {
            setIsLoading(true);
            try {
                const [evalRes, datasetRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${evalId}`),
                    fetch(`/api/config/evaluation/${evalId}/dataset?page=${page}&size=${pageSize}`),
                ]);

                const eval_data = await evalRes.json();
                const evalData = eval_data.data;
                console.log('evalData:', evalData);
                setEvalConfig(evalData);
                setEvalRunConfig(evalData.default_run_config);

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

            } catch (err: any) {
                setDatasetError(err || '加载数据集失败');
            } finally {
                setIsLoading(false);
            }
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
                item.input.toLowerCase().includes(searchTerm.toLowerCase()) ||
                item.expected_output.toLowerCase().includes(searchTerm.toLowerCase());

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
            run_config: evalRunConfig
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
            router.push(`/evaluation/${evalId}/experiments`);
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
        } catch (error) {
            console.error('上传失败:', error);
        } finally {
            setUploading(false);
            // 清空文件选择框
            if (fileInputRef.current) {
                fileInputRef.current.value = ''; // 清空 input 的值
            }
            setPage(1);
        }
    };

    const handleKbSelect = (kb_id: string, kb_name: string, checked: boolean) => {
        console.log('handleKbSelect', kb_id, kb_name, checked);
        if (checked) {
            const kb_ids = evalRunConfig.kb_ids.includes(kb_id)
                ? evalRunConfig.kb_ids
                : [...evalRunConfig.kb_ids, kb_id];
            setEvalRunConfig((prev) => ({
                ...prev,
                kb_ids: kb_ids,
            }));
            if (!selectedKbNames.includes(kb_name)) {
                setSelectedKbNames((prev) => [...prev, kb_name]);
            }
        } else {
            const kb_ids = evalRunConfig.kb_ids.filter((id) => id !== kb_id);
            setEvalRunConfig((prev) => ({
                ...prev,
                kb_ids: kb_ids,
            }));
            if (selectedKbNames.includes(kb_name)) {
                setSelectedKbNames((prev) => prev.filter((name) => name !== kb_name));
            }
        }
    };
    const handleMcpSelect = (
        mcp_id: string,
        mcp_name: string,
        checked: boolean,
    ) => {
        if (checked) {
            const mcp_ids = evalRunConfig.mcp_ids.includes(mcp_id)
                ? evalRunConfig.mcp_ids
                : [...evalRunConfig.mcp_ids, mcp_id];
            setEvalRunConfig((prev) => ({
                ...prev,
                mcp_ids: mcp_ids,
            }));

            if (!selectedMcpNames.includes(mcp_name)) {
                setSelectedMcpNames((prev) => [...prev, mcp_name]);
            }
        } else {
            const mcp_ids = evalRunConfig.mcp_ids.filter((id) => id !== mcp_id);
            setEvalRunConfig((prev) => ({
                ...prev,
                mcp_ids: mcp_ids,
            }));
            if (selectedMcpNames.includes(mcp_name)) {
                setSelectedMcpNames((prev) => prev.filter((name) => name !== mcp_name));
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
                        <div className="flex">
                            <Label htmlFor="basemodel" className="w-[90px]">
                                基模型选择 <span className="text-destructive">*</span>{' '}
                            </Label>
                            <div className="px-6">
                                {llms.length > 0 ? (
                                    <Select
                                        value={evalRunConfig?.model_id}
                                        onValueChange={(value) =>
                                            setEvalRunConfig((prev) => ({
                                                ...prev,
                                                model_id: value,
                                            }))
                                        }
                                    >
                                        <SelectTrigger>
                                            <SelectValue placeholder="请选择基模型" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {llms.map((llm) => (
                                                <SelectItem key={llm.id} value={llm.model_id}>
                                                    {llm.model_id}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                ) : (
                                    <div>
                                        <p className="text-sm text-muted-foreground">尚未配置大模型</p>
                                        <Button
                                            variant="outline"
                                            onClick={() => {
                                                router.push('/config/model/llm');
                                            }}
                                        >
                                            前往添加
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </div>
                        <div className="flex gap-6">
                            <Label htmlFor="enable_search" className="w-[90px]">
                                启用联网搜索
                            </Label>
                            <Switch
                                id="enable_search"
                                checked={evalRunConfig?.enable_search}
                                onCheckedChange={(checked) => {
                                    setEvalRunConfig((prev) => ({
                                        ...prev,
                                        enable_search: checked,
                                    }));
                                }}
                            />
                        </div>
                        <div className="flex gap-6">
                            <Label htmlFor="enable_agent" className="w-[90px]">
                                Agentic模式
                            </Label>
                            <Switch
                                id="enable_agent"
                                checked={evalRunConfig?.enable_agent}
                                onCheckedChange={(checked) => {
                                    setEvalRunConfig((prev) => ({
                                        ...prev,
                                        enable_agent: checked,
                                    }));
                                }}
                            />
                        </div>
                        <div className="flex">
                            <Label htmlFor="kb_selection" className="w-[90px]">
                                知识库选择
                            </Label>
                            <div className="pl-6 pr-6">
                                {kbs.length > 0 ? (
                                    <DropdownMenu modal={true}>
                                        <DropdownMenuTrigger asChild>
                                            <Button
                                                variant="outline"
                                                className="text-sm text-muted-foreground"
                                            >
                                                已选{evalRunConfig?.kb_ids?.length || 0}个，可多选 <ChevronDownIcon />
                                            </Button>
                                        </DropdownMenuTrigger>
                                        <DropdownMenuContent className="w-56">
                                            <DropdownMenuLabel>知识库</DropdownMenuLabel>
                                            <DropdownMenuSeparator />
                                            {kbs.map((kb) => (
                                                <DropdownMenuCheckboxItem
                                                    key={kb.id}
                                                    checked={evalRunConfig?.kb_ids?.includes(kb.id)}
                                                    onCheckedChange={(checked) =>
                                                        handleKbSelect(kb.id, kb.name, checked)
                                                    }
                                                    onSelect={(e) => e.preventDefault()}
                                                >
                                                    {kb.name}
                                                </DropdownMenuCheckboxItem>
                                            ))}
                                        </DropdownMenuContent>
                                    </DropdownMenu>
                                ) : (
                                    <div>
                                        <p className="text-sm text-muted-foreground">尚未配置知识库</p>
                                    </div>
                                )}
                            </div>
                            {selectedKbNames.length > 0 && (
                                <div className="flex gap-1.5 items-center">
                                    {selectedKbNames.map((name) => (
                                        <Badge variant="secondary" className="h-6" key={name}>
                                            {name}
                                        </Badge>
                                    ))}
                                </div>
                            )}
                        </div>
                        <div className="flex">
                            <Label htmlFor="mcp_selection" className="w-[90px]">
                                MCP选择
                            </Label>
                            <div className="pl-6 pr-6">
                                {mcps.length > 0 ? (
                                    <DropdownMenu modal={true}>
                                        <DropdownMenuTrigger asChild>
                                            <Button
                                                variant="outline"
                                                className="text-sm text-muted-foreground"
                                            >
                                                已选{evalRunConfig?.mcp_ids?.length}个，可多选 <ChevronDownIcon />
                                            </Button>
                                        </DropdownMenuTrigger>
                                        <DropdownMenuContent className="w-56">
                                            <DropdownMenuLabel>MCP</DropdownMenuLabel>
                                            <DropdownMenuSeparator />
                                            {mcps.map((mcp) => (
                                                <DropdownMenuCheckboxItem
                                                    key={mcp.id}
                                                    checked={evalRunConfig?.mcp_ids?.includes(mcp.id)}
                                                    onCheckedChange={(checked) =>
                                                        handleMcpSelect(mcp.id, mcp.name, checked)
                                                    }
                                                    onSelect={(e) => e.preventDefault()}
                                                >
                                                    {mcp.name}
                                                </DropdownMenuCheckboxItem>
                                            ))}
                                        </DropdownMenuContent>
                                    </DropdownMenu>
                                ) : (
                                    <div>
                                        <p className="text-sm text-muted-foreground">尚未配置MCP</p>
                                    </div>
                                )}
                            </div>
                            {selectedMcpNames.length > 0 && (
                                <div className="flex gap-1.5 items-center">
                                    {selectedMcpNames.map((name) => (
                                        <Badge variant="secondary" className="h-6" key={name}>
                                            {name}
                                        </Badge>
                                    ))}
                                </div>
                            )}
                        </div>
                        <div className="flex items-center">
                            <Label htmlFor="ai_guardrail" className="w-[90px]">
                                AI安全护栏
                            </Label>

                            <div className="gap-4 pl-6 text-sm items-center space-y-2">
                                <div className="flex space-y-4">
                                    <Label htmlFor="input_guardrail" className="w-[120px]">
                                        输入护栏
                                    </Label>
                                    <Switch
                                        id="enable_input_check"
                                        checked={evalRunConfig?.enable_input_guardrail || false}
                                        onCheckedChange={(checked) => {
                                            setEvalRunConfig((prev) => ({
                                                ...prev,
                                                enable_input_guardrail: checked,
                                            }));
                                        }}
                                    />
                                </div>
                                <div className="flex space-y-4">
                                    <Label htmlFor="output_guardrail" className="w-[120px]">
                                        输出护栏
                                    </Label>
                                    <Switch
                                        id="enable_output_check"
                                        checked={evalRunConfig?.enable_output_guardrail || false}
                                        onCheckedChange={(checked) => {
                                            setEvalRunConfig((prev) => ({
                                                ...prev,
                                                enable_output_guardrail: checked,
                                            }));
                                        }}
                                    />
                                    
                                </div>

                                <div className="space-y-4">
                                    <Label htmlFor="guardrail_hint" className="w-[100px]">
                                        默认护栏提示
                                    </Label>
                                    <Input
                                        className="w-100"
                                        value={evalRunConfig?.guardrail_hint || "作为人工智能助手，我无法回应包含不当或敏感信息的内容。"}
                                        onChange={(e) => {
                                            setEvalRunConfig((prev) => ({
                                                ...prev,
                                                guardrail_hint: e.target.value,
                                            }));

                                        }}
                                    />
                                    
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
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
                                    {/* <Button
                                        disabled={selectedItems.size === 0 && !isAllSelected}
                                        onClick={handleBatchRun}
                                    >
                                        <PlayIcon className="mr-2 h-4 w-4" />
                                        {isAllSelected ? `批量运行(所有${totalItems}项)` : `批量运行(${selectedItems.size}项)`}
                                    </Button> */}
                                    <Dialog open={isRunBatchDetailOpen} onOpenChange={setIsRunBatchDetailOpen}>
                                        <DialogTrigger asChild>
                                            <Button
                                                disabled={selectedItems.size === 0 && !isAllSelected}
                                            >
                                                <PlayIcon className="mr-2 h-4 w-4" />
                                                {isAllSelected ? `批量运行(所有${totalItems}项)` : `批量运行(${selectedItems.size}项)`}
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
                                    <Button onClick={() => router.push(`/evaluation/${evalId}/experiments`)}>
                                        <Eye className="mr-2 h-4 w-4" /> 查看实验
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
