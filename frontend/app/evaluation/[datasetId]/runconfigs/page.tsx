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
    Dialog,
    DialogTrigger,
} from "@/components/ui/dialog";
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { Settings, Pencil, Trash2, Settings2, BarChart2, Loader2 } from "lucide-react";
import { EvalConfigFormDialog } from "@/app/evaluation/components/runconfig-form-dialog";
import { RunConfig } from '@/app/evaluation/[datasetId]/types';

const default_eval_run_config = {
    id: "",
    name: "",
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



export default function RunConfigsPage(
    { params }: { params: Promise<{ datasetId: string }> }
) {
    const { datasetId } = use(params);
    const router = useRouter();
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [evalRunConfigs, setRunConfigs] = useState<RunConfig[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const pageSize = 10;
    // 用于跟踪选中的行
    const [totalItems, setTotalItems] = useState(0);
    const [dataseterror, setDatasetError] = useState('');
    const [llms, setLlms] = useState<LlmConfig[]>([]);
    const [mcps, setMcps] = useState<McpConfig[]>([]);
    const [kbs, setKbs] = useState<KbConfig[]>([]);

    const [isNewSettingsOpen, setIsNewSettingsOpen] = useState(false);
    const [isCreateLoading, setIsCreateLoading] = useState(false);
    const [isEditSetting, setIsEditSetting] = useState(false);
    const [editConfig, setEditConfig] = useState<RunConfig>(default_eval_run_config);

    useEffect(() => {
        const fetchConfigs = async () => {
            setIsLoading(true);
            try {
                const [evalRes, datasetRes, llmRes, mcpRes, kbRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${datasetId}`),
                    fetch(`/api/config/evaluation/${datasetId}/runconfigs?page=${page}&size=${pageSize}`),
                    fetch(`/api/config/llms`),
                    fetch(`/api/config/mcps`),
                    fetch(`/api/config/knowledgebases`),
                ]);

                const eval_data = await evalRes.json();
                const evalData = eval_data.data;
                console.log('evalData:', evalData);

                if (!datasetRes.ok) throw new Error('获取评估任务列表失败');
                const json_data = await datasetRes.json();
                console.log("evaluation dataset json_data", json_data)
                const data = json_data.data.items;

                setRunConfigs(data);
                setTotalItems(json_data.data.total);
                setTotalPages(json_data.data.pages);

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


    const createNewRunConfig = async (data: RunConfig) => {
        console.log("createNewRunConfig", data)
        try {
            if (!isEditSetting) {
                const res = await fetch(
                    `/api/config/evaluation/${datasetId}/runconfigs`,
                    {
                        method: "POST",
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data),
                    },
                );
                if (!res.ok) {
                    alert('创建失败');
                    return;
                }
                const result = await res.json();
                console.log('创建成功:', result);
                setRunConfigs((prev) => [...prev, result.data]); // 追加新配置
            } else {
                const res = await fetch(
                    `/api/config/evaluation/${datasetId}/runconfigs/${data.id}`,
                    {
                        method: "PUT",
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data),
                    },
                );
                if (!res.ok) {
                    alert('更新失败');
                    return;
                }
                const result = await res.json();
                setRunConfigs((prev) =>
                    prev.map((config) => (config.id === result.data.id ? result.data : config)),
                );
                console.log('更新成功:', result.data);
            }
        } catch (error) {
            console.error('创建失败:', error);
        } finally {
            setIsCreateLoading(false);
            setIsNewSettingsOpen(false);
        }
    }

    const renderBadges = (ids: string[], configs: McpConfig[] | KbConfig[], maxShow = 2) => {
        if (ids.length === 0) return <span className="text-muted-foreground">—</span>;

        // 创建 id → name 映射
        const idToNameMap = Object.fromEntries(
            configs.map(config => [config.id, config.name])
        );

        // 获取所有名称（保留原始顺序）
        const names = ids.map(id => idToNameMap[id] || id); // 如果没找到，fallback 到 ID

        const visible = names.slice(0, maxShow);
        const hidden = names.slice(maxShow);


        return (
            <div className="flex flex-wrap items-center gap-1">
                {visible.map((name, idx) => (
                    <Badge key={idx} variant="secondary" className="text-xs">
                        {name}
                    </Badge>
                ))}
                {hidden.length > 0 && (
                    <TooltipProvider>
                        <Tooltip>
                            <TooltipTrigger>
                                <Badge variant="outline" className="text-xs">
                                    +{hidden.length}
                                </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="top" className="max-w-xs">
                                <div className="space-y-1">
                                    {hidden.map((name, i) => (
                                        <div key={i} className="text-sm">
                                            {name}
                                        </div>
                                    ))}
                                </div>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                )}
            </div>
        );
    };

    const renderGuardrailStatus = (config: RunConfig) => {
        const hasInput = config.enable_input_guardrail;
        const hasOutput = config.enable_output_guardrail;
        const hint = config.guardrail_hint;

        if (!hasInput && !hasOutput && !hint) {
            return <span className="text-muted-foreground">未启用</span>;
        }

        return (
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-auto p-1">
                            <span className="text-xs text-blue-600">详情</span>
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-sm p-3">
                        <div className="space-y-1 text-sm">
                            <div>
                                <strong>输入护栏：</strong>
                                {hasInput ? "✅ 启用" : "❌ 未启用"}
                            </div>
                            <div>
                                <strong>输出护栏：</strong>
                                {hasOutput ? "✅ 启用" : "❌ 未启用"}
                            </div>
                            {hint && (
                                <div>
                                    <strong>提示语：</strong>
                                    <div className="mt-1 text-xs bg-muted p-2 rounded break-all text-black">
                                        {hint}
                                    </div>
                                </div>
                            )}
                        </div>
                    </TooltipContent>
                </Tooltip>
            </TooltipProvider>
        );
    };


    const onDelete = async (config_id: string) => {
        try {
            const res = await fetch(`/api/config/evaluation/${datasetId}/runconfigs/${config_id}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!res.ok) {
                throw new Error('删除失败，请检查网络或配置');
            }
            setRunConfigs((prev) => prev.filter((config) => config.id !== config_id));
        }
        catch (err: any) { console.log('删除实验设置任务出错: ', err); }
    }



    return (
        <div className="flex flex-col h-full min-h-0">
            <Card className="flex flex-col h-full min-h-0 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                <CardHeader className="shrink-0 flex md:items-center md:justify-between">
                    <div>
                        <CardTitle className="text-2xl font-bold flex items-center gap-2">
                            <Settings2 className="h-5 w-5" /> 运行设置
                        </CardTitle>
                        <p className="text-sm text-muted-foreground mt-1">
                            应用运行时参数、模型、插件等设置
                        </p>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">

                        <div className="flex gap-2">
                            <EvalConfigFormDialog
                                mode={isEditSetting ? "edit" : "new"}
                                config={isEditSetting ? editConfig : undefined}
                                llms={llms}
                                mcps={mcps}
                                kbs={kbs}
                                datasetId={datasetId}
                                isOpen={isNewSettingsOpen}
                                onOpenChange={setIsNewSettingsOpen}
                                onSave={createNewRunConfig}
                                isSaving={isCreateLoading}
                            />
                            <Dialog open={isNewSettingsOpen} onOpenChange={setIsNewSettingsOpen}>
                                <DialogTrigger asChild>
                                    <Button>
                                        <Settings className="mr-2 h-4 w-4" /> 新建配置
                                    </Button>
                                </DialogTrigger>
                            </Dialog>
                        </div>
                    </div>
                </CardHeader>
                <CardContent className="flex-1 min-h-0 overflow-y-auto p-0">
                    <div className="rounded-md h-full min-h-0">
                        <Table className='rounded-md border'>
                            <TableHeader>
                                <TableRow>
                                    <TableHead className='border-r border-r-border'>名称</TableHead>
                                    <TableHead>基模型</TableHead>
                                    <TableHead>MCP</TableHead>
                                    <TableHead>知识库</TableHead>
                                    <TableHead className="text-center">联网搜索</TableHead>
                                    <TableHead className="text-center">Agentic</TableHead>
                                    <TableHead className="text-center border-r border-r-border">护栏状态</TableHead>
                                    <TableHead className="text-center">操作</TableHead>
                                </TableRow>
                            </TableHeader>

                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-32 text-center">
                                            <div className="flex items-center justify-center space-x-4">
                                                <Loader2 className="h-6 w-6 animate-spin" />
                                                <h4 className="font-bold">Loading Configs</h4>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : evalRunConfigs.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-24 text-center">
                                            暂无数据
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    evalRunConfigs.map((config) => (
                                        <TableRow key={config.id} className="hover:bg-muted/50">
                                            <TableCell className='border-r border-r-border'>
                                                <div className="flex flex-col">
                                                    <span className="font-medium">{config.name}</span>
                                                    <span className="text-xs text-muted-foreground">ID: {config.id.slice(0, 8)}...</span>
                                                </div>
                                            </TableCell>
                                            <TableCell className="font-mono text-sm">{config.model_id || "—"}</TableCell>
                                            <TableCell>{renderBadges(config.mcp_ids, mcps)}</TableCell>
                                            <TableCell>{renderBadges(config.kb_ids, kbs)}</TableCell>
                                            <TableCell className="text-center">
                                                <Switch checked={config.enable_search} disabled />
                                            </TableCell>
                                            <TableCell className="text-center">
                                                <Switch checked={config.enable_agent} disabled />
                                            </TableCell>
                                            <TableCell className="text-center border-r border-r-border">
                                                {renderGuardrailStatus(config)}
                                            </TableCell>

                                            <TableCell className="text-right">
                                                <div className="flex justify-end gap-1">
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => {
                                                            setEditConfig(config);
                                                            setIsEditSetting(true);
                                                            setIsNewSettingsOpen(true);
                                                        }}
                                                    >
                                                        <Pencil className="h-4 w-4" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        onClick={() => onDelete(config.id)}
                                                    >
                                                        <Trash2 className="h-4 w-4 text-destructive" />
                                                    </Button>
                                                </div>
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
