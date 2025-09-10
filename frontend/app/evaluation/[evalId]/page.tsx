'use client';

import { useState, useEffect, use, useRef } from "react";
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
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from 'sonner';
import { CheckCircle, CircleXIcon} from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { BookOpen, Settings, FlaskConical } from "lucide-react";
import EvalDatasetsDetailsPage from '@/app/evaluation/[evalId]/datasets/page';
import EvalExperimentsDetailsPage from '@/app/evaluation/[evalId]/experiments/page';
import EvalSettingsDetailsPage from '@/app/evaluation/[evalId]/settings/page';
export interface EvalConfig {
  id: string;
  name: string;
  description: string;
  type: string;
}

export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalId : string }> }
) {
    const { evalId } = use(params);
    const router = useRouter();
    const [evaluation, setEvaluationConfig] = useState<EvalConfig>(); // 知识库列表
    const [datasetLen, setDatasetLen] = useState<number>(0); // 数据集条目数量
    const [experimentLen, setExperimentLen] = useState<number>(0); // 实验条目数量

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
    
            setEvaluationConfig(kb_data); // 更新状态
            console.log('评估任务详情数据:', kb_data);

            setDatasetLen(allDatasetRes.ok ? await allDatasetRes.json().then(res => res.data.total) : 0);
            setExperimentLen(experimentRes.ok ? await experimentRes.json().then(res => res.data.total) : 0);

          } catch (err: any) {
            toast.error(err.message);
          }
        };
        fetchKbConfigs();
      }, []);

    // const totalExperiments = evaluation.experiments.length;
    // const completedExperiments = evaluation.experiments.filter(e => e.status === 'completed').length;
     
    if (!evaluation) {
        return <div className="p-6">加载中...</div>;
    }

    return (
        <div className="flex flex-col h-screen px-6">
            <div className="h-[150px]">
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
            <div className="px-2 w-full flex-1">
                <Tabs defaultValue="datasets" className="w-full">
                    <TabsList className="p-1">
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
                            <FlaskConical className="h-3.5 w-3.5" />实验
                        </TabsTrigger>
                        <TabsTrigger
                            key="settings"
                            value="settings"
                            className="px-4"
                        >
                            <Settings className="h-3.5 w-3.5"/>实验设置
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent key="datasets" value="datasets">
                        <EvalDatasetsDetailsPage params={params} />
                    </TabsContent>
                    <TabsContent key="experiments" value="experiments">
                        <EvalExperimentsDetailsPage params={params} />
                    </TabsContent>
                    <TabsContent key="settings" value="settings">
                        <EvalSettingsDetailsPage params={params} />
                    </TabsContent>
                    </Tabs>
                
                </div>
        </div>
    );
}
