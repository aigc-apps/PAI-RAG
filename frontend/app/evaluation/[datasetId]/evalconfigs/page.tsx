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
import { EvalConfigFormDialog } from "@/app/evaluation/components/evalconfig-form-dialog";
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';

const default_evaluator_config = {
    id: "",
    name: "",
    type: "",
    model_id: "",
    case_sensitive: false,
    ignore_punctuation: false
};



export default function EvaluatorConfigsPage(
    { params }: { params: Promise<{ datasetId: string }> }
) {
    const { datasetId } = use(params);
    const router = useRouter();
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [evaluatorConfigs, setEvaluatorConfigs] = useState<EvaluatorConfig[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const pageSize = 10;
    // 用于跟踪选中的行
    const [totalItems, setTotalItems] = useState(0);
    const [dataseterror, setDatasetError] = useState('');
    const [llms, setLlms] = useState<LlmConfig[]>([]);

    const [isNewSettingsOpen, setIsNewSettingsOpen] = useState(false);
    const [isCreateLoading, setIsCreateLoading] = useState(false);
    const [isEditSetting, setIsEditSetting] = useState(false);
    const [editConfig, setEditConfig] = useState<EvaluatorConfig>(default_evaluator_config);

    useEffect(() => {
        const fetchConfigs = async () => {
            setIsLoading(true);
            try {
                const [evalRes, datasetRes, llmRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${datasetId}`),
                    fetch(`/api/config/evaluation/${datasetId}/evalconfigs?page=${page}&size=${pageSize}`),
                    fetch(`/api/config/llms`),
                ]);

                const eval_data = await evalRes.json();
                const evalData = eval_data.data;
                console.log('evalData:', evalData);

                if (!datasetRes.ok) throw new Error('获取评估任务列表失败');
                const json_data = await datasetRes.json();
                console.log("evaluation dataset json_data", json_data)
                const data = json_data.data.items;

                setEvaluatorConfigs(data);
                setTotalItems(json_data.data.total);
                setTotalPages(json_data.data.pages);

                const llmData = (await llmRes.json())?.data.items || [];
                console.log('llmData', llmData);
                setLlms([...llmData]);

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


    const createNewEvaluatorConfig = async (data: EvaluatorConfig) => {
        console.log("createNewEvaluatorConfig", data)
        try {
            if (!isEditSetting) {
                const res = await fetch(
                    `/api/config/evaluation/${datasetId}/evalconfigs`,
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
                setEvaluatorConfigs((prev) => [...prev, result.data]); // 追加新配置
            } else {
                const res = await fetch(
                    `/api/config/evaluation/${datasetId}/evalconfigs/${data.id}`,
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
                setEvaluatorConfigs((prev) =>
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


    const onDelete = async (config_id: string) => {
        try {
            const res = await fetch(`/api/config/evaluation/${datasetId}/evalconfigs/${config_id}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!res.ok) {
                throw new Error('删除失败，请检查网络或配置');
            }
            setEvaluatorConfigs((prev) => prev.filter((config) => config.id !== config_id));
        }
        catch (err: any) { console.log('删除实验设置任务出错: ', err); }
    }



    return (
        <div className="flex flex-col h-full min-h-0">
            <Card className="flex flex-col h-full min-h-0 shadow-sm hover:shadow-md transition-shadow overflow-hidden">
                <CardHeader className="shrink-0 flex md:items-center md:justify-between">
                    <div>
                        <CardTitle className="text-lg font-medium flex items-center gap-2">
                            <BarChart2 className="h-5 w-5" /> 评估器设置
                        </CardTitle>
                        <p className="text-sm text-muted-foreground mt-1">
                            进行评估实验的评分标准、评估器等配置
                        </p>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-2 w-full md:w-auto">

                        <div className="flex gap-2">
                            <EvalConfigFormDialog
                                mode={isEditSetting ? "edit" : "new"}
                                config={isEditSetting ? editConfig : undefined}
                                llms={llms}
                                datasetId={datasetId}
                                isOpen={isNewSettingsOpen}
                                onOpenChange={setIsNewSettingsOpen}
                                onSave={createNewEvaluatorConfig}
                                isSaving={isCreateLoading}
                            />
                            <Dialog open={isNewSettingsOpen} onOpenChange={setIsNewSettingsOpen}>
                                <DialogTrigger asChild>
                                    <Button onClick={() => setIsEditSetting(false)}>
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

                                    {/* 评估设置列 */}
                                    <TableHead className='border-r border-r-border'>评估器类型</TableHead>
                                    <TableHead className='border-r border-r-border'>评估设置</TableHead>
                                    <TableHead className="text-center">操作</TableHead>
                                </TableRow>
                            </TableHeader>

                            <TableBody>
                                {isLoading ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-32 text-center">
                                            <div className="flex items-center justify-center space-x-4">
                                                <Loader2 className="h-6 w-6 animate-spin" />
                                                <h4 className="font-medium">Loading Configs</h4>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ) : evaluatorConfigs.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={9} className="h-24 text-center">
                                            暂无数据
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    evaluatorConfigs.map((config) => (
                                        <TableRow key={config.id} className="hover:bg-muted/50">
                                            <TableCell className='border-r border-r-border'>
                                                <div className="flex flex-col">
                                                    <span className="font-medium">{config.name}</span>
                                                    <span className="text-xs text-muted-foreground">ID: {config.id.slice(0, 8)}...</span>
                                                </div>
                                            </TableCell>
                                            
                                            <TableCell className='border-r border-r-border'>
                                                <div className="space-y-1">
                                                    <div className="font-medium text-sm">
                                                        <Badge variant={config.type === "ExactMatch" ? "secondary" : "outline"}>
                                                            {config.type === "ExactMatch" ? "精确匹配" : "LLM 评判"}
                                                        </Badge>
                                                    </div>
                                                </div>
                                            </TableCell>

                                            <TableCell className='border-r border-r-border'>
                                                <div className="space-y-1">
                                                    {config.type === "ExactMatch" && (
                                                        <div className="text-xs text-muted-foreground space-y-0.5">
                                                            <div>区分大小写: {config.case_sensitive ? "是" : "否"}</div>
                                                            <div>忽略标点: {config.ignore_punctuation ? "是" : "否"}</div>
                                                        </div>
                                                    )}
                                                    {config.type === "LLMJudge" && (
                                                        <div className="text-xs text-muted-foreground">
                                                            模型: {config.model_id || "未指定"}
                                                        </div>
                                                    )}
                                                </div>
                                            </TableCell>


                                            <TableCell className="text-right">
                                                <div className="flex justify-center gap-1">
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
