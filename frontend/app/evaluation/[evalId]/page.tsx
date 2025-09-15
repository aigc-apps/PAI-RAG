'use client';

import { useState, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Button } from '@/components/ui/button';
import React from 'react';
import { toast } from 'sonner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { BookOpen, Settings, FlaskConical } from "lucide-react";
import EvalDatasetsDetailsPage from '@/app/evaluation/[evalId]/datasets/page';
import EvalExperimentsDetailsPage from '@/app/evaluation/[evalId]/experiments/page';
import EvalSettingsDetailsPage from '@/app/evaluation/[evalId]/settings/page';
import { EvalConfig } from "@/app/evaluation/[evalId]/types";
export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalId: string }> }
) {
    const { evalId } = use(params);
    const router = useRouter();
    const [evaluation, setEvaluationConfig] = useState<EvalConfig>(); // 知识库列表

    useEffect(() => {
        const fetchKbConfigs = async () => {
            try {
                const [evalRes, allDatasetRes, experimentRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${evalId}`),
                    fetch(`/api/config/evaluation/${evalId}/dataset`),
                    fetch(`/api/config/evaluation/${evalId}/experiments`),
                ]);

                if (!evalRes.ok) throw new Error('获取评估任务配置失败');
                const json_data = await evalRes.json();
                const kb_data = json_data.data;
                console.log('评估任务详情数据:', kb_data);
                setEvaluationConfig(kb_data); // 更新状态
            } catch (err: any) {
                toast.error(err.message);
            }
        };
        fetchKbConfigs();
    }, []);

    if (!evaluation) {
        return <div className="p-6">加载中...</div>;
    }

    return (
        <div className="flex flex-col h-screen px-6">
            <div className="h-[150px] shrink-0">
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
                                    <BreadcrumbPage>{evaluation.name}</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold">概览: {evaluation.name}</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                {evaluation.description}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="px-2 w-full flex-1 flex flex-col min-h-0">
                <Tabs defaultValue="experiments" className="w-full flex-1 flex flex-col min-h-0">
                    <TabsList className="p-1 shrink-0">
                        <TabsTrigger
                            key="datasets"
                            value="datasets"
                            className="px-4"
                        >
                            <BookOpen className="h-3.5 w-3.5" /> 样本
                        </TabsTrigger>
                        <TabsTrigger
                            key="experiments"
                            value="experiments"
                            className="px-4"
                        >
                            <FlaskConical className="h-3.5 w-3.5" />运行历史
                        </TabsTrigger>
                        <TabsTrigger
                            key="settings"
                            value="settings"
                            className="px-4"
                        >
                            <Settings className="h-3.5 w-3.5" />运行设置
                        </TabsTrigger>
                    </TabsList>
                    <div className="flex-1 min-h-0 overflow-hidden">
                        <TabsContent key="datasets" value="datasets" className="h-full flex flex-col min-h-0">
                            <EvalDatasetsDetailsPage params={params} />
                        </TabsContent>
                        <TabsContent key="experiments" value="experiments" className="h-full flex flex-col min-h-0">
                            <EvalExperimentsDetailsPage params={params} />
                        </TabsContent>
                        <TabsContent key="settings" value="settings" className="h-full flex flex-col min-h-0">
                            <EvalSettingsDetailsPage params={params} />
                        </TabsContent>
                    </div>
                </Tabs>

            </div>
        </div>
    );
}
