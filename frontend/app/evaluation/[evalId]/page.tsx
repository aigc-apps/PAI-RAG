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

export interface EvalConfig {
  id: string;
  name: string;
  description: string;
  run_type: string;
  chatbot_id: string;
  default_run_config: {
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
                                    <BreadcrumbPage>{evaluation.name}</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold">任务概览: {evaluation.name}</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                {evaluation.description}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="px-2 max-w-6xl">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <Card className="flex flex-col h-full">
                    <CardHeader>
                        <CardTitle>任务设置</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow">
                        {
                            (evaluation?.chatbot_id || evaluation?.default_run_config?.model_id) ? (
                                <div className="space-y-3">
                                    <div className="flex items-center">
                                        <h3 className="font-semibold">任务类型</h3>
                                        {evaluation.run_type === "chatbot" ? (
                                            <Badge className="ml-2">{evaluation?.chatbot_id}</Badge>
                                        ) : (
                                            <Badge className="ml-2">自定义评估</Badge>
                                        )}
                                    </div>
                                    <div className="flex items-center">
                                        <h3 className="font-semibold mb-1">基模型</h3>
                                        <Badge variant="outline" className="ml-2">{evaluation.default_run_config.model_id}</Badge>
                                    </div>
                                    <div className="flex items-center">
                                        <h3 className="font-semibold">联网搜索</h3>
                                        {evaluation.default_run_config.enable_search ? (
                                            <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                                        ) : (
                                            <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                                        )}
                                    </div>
                                    <div className="flex items-center">
                                        <h3 className="font-semibold">Agentic模式</h3>
                                        {evaluation.default_run_config.enable_agent ? (
                                            <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                                        ) : (
                                            <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                                        )}
                                    </div>
                                    <div className="flex items-center">
                                        <h3 className="font-semibold mb-1">MCP Server</h3>
                                        {Array.isArray(evaluation.default_run_config.mcp_ids) && evaluation.default_run_config.mcp_ids.length === 0 ? (
                                            <p className="text-muted-foreground pl-2">尚未配置MCP</p>
                                        ) : (
                                            Array.isArray(evaluation.default_run_config.mcp_ids) && evaluation.default_run_config.mcp_ids.map((mcp, idx) => (
                                                <Badge key={mcp || idx}>{mcp}</Badge>
                                            ))
                                        )}
                                    </div>
                                    <div className="flex items-center">
                                        <h3 className="font-semibold mb-1">知识库</h3>
                                        {Array.isArray(evaluation.default_run_config.kb_ids) && evaluation.default_run_config.kb_ids.length === 0 ? (
                                            <p className="text-muted-foreground pl-2">尚未配置知识库</p>
                                        ) : (
                                            Array.isArray(evaluation.default_run_config.kb_ids) && evaluation.default_run_config.kb_ids.map((kb, idx) => (
                                                <Badge key={kb || idx}>{kb}</Badge>
                                            ))
                                        )}
                                    </div>
                                    <div className="flex items-center">
                                        <h3 className="font-semibold">安全护栏</h3>
                                        <div className="pl-4">
                                            {evaluation.default_run_config.enable_input_guardrail ? (<Badge>输入护栏已启用</Badge>) : (<Badge variant="outline">输入护栏未启用</Badge>)}
                                            {evaluation.default_run_config.enable_output_guardrail ? (<Badge>输出护栏已启用</Badge>) : (<Badge variant="outline">输出护栏未启用</Badge>)}
                                        </div>
                                    </div>

                                </div>
                            ): (
                                <div>
                                    <h3 className="font-semibold mb-1">未进行任务设置，请先设置</h3>
                                </div>
                            )
                        }
                        
                    </CardContent>
                    <CardFooter>
                        <Button variant="outline" className="w-full" onClick={() => router.push(`/evaluation/${evaluation.id}/settings`)}>
                            {(evaluation.chatbot_id || evaluation.default_run_config.model_id) ? ("查看/修改任务设置"):("进行任务设置")}
                        </Button>
                    </CardFooter>
                    </Card>

                    {/* 数据集卡片 */}
                    <Card className="flex flex-col h-full">
                    <CardHeader>
                        <CardTitle>数据集信息</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow">
                        {datasetLen > 0 ? (
                            <div className="space-y-3">
                                <div className="text-center p-3 bg-muted rounded-lg">
                                    <div className="text-2xl font-bold">{datasetLen}</div>
                                    <div className="text-sm text-muted-foreground">数据集条数</div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-sm text-muted-foreground">
                                当前任务还未关联数据集，请前往数据集页面添加数据。
                            </div>
                        )}
                        
                    </CardContent>
                    <CardFooter>
                        <Button variant="outline" className="w-full" onClick={() => router.push(`/evaluation/${evaluation.id}/datasets`)}>
                            {datasetLen > 0 ? ("查看数据集详情") : ("添加数据") } 
                        </Button>
                    </CardFooter>
                    </Card>

                    {/* 实验卡片 */}
                    <Card className="flex flex-col h-full">
                    <CardHeader>
                        <CardTitle>实验统计</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow">
                        <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="text-center p-3 bg-muted rounded-lg">
                            <div className="text-2xl font-bold">{experimentLen}</div>
                            <div className="text-sm text-muted-foreground">总实验数</div>
                            </div>
                            <div className="text-center p-3 bg-muted rounded-lg">
                            {/* <div className="text-2xl font-bold">{completedExperiments}</div> */}
                            <div className="text-sm text-muted-foreground">已完成</div>
                            </div>
                        </div>
                        
                        <div>
                            <h3 className="font-semibold mb-2">实验状态</h3>
                            <div className="flex flex-wrap gap-2">
                            {/* {evaluation.experiments.map(exp => (
                                <Badge 
              
                                key={exp.id} 
                                    variant={
                                        exp.status === 'completed' ? 'default' : 
                                        exp.status === 'running' ? 'destructive' : 
                                        'secondary'
                                    }
                                >
                                {exp.name}: {exp.status === 'completed' ? '已完成' : 
                                            exp.status === 'running' ? '进行中' : '待处理'}
                                </Badge>
                            ))} */}
                            </div>
                        </div>
                        </div>
                    </CardContent>
                    <CardFooter>
                        <Button variant="outline" className="w-full" onClick={() => router.push(`/evaluation/${evaluation.id}/experiments`)}>
                            查看所有实验
                        </Button>
                    </CardFooter>
                    </Card>
                </div>
                </div>
        </div>
    );
}
