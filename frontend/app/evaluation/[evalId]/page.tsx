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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { EvalConfigCard } from "../eval_config";
import { DatasetTableCard } from "../dataset-table-card";
import { ExperimentTableCard } from "../experiment-table-card";
import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Link, useParams } from "react-router-dom";


export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalId : string }> }
) {
    const { evalId } = use(params);
    const router = useRouter();

    const task = {
        id: evalId || 'task-001',
        model: 'BERT-large',
        mcp: 'F1 Score',
        dataset: {
            name: 'GLUE Benchmark',
            entries: 11000
        },
        experiments: [
            { id: 'exp-001', status: 'completed', name: 'Baseline' },
            { id: 'exp-002', status: 'running', name: 'Optimized' },
            { id: 'exp-003', status: 'pending', name: 'Ablation' }
        ]
    };
    const totalExperiments = task.experiments.length;
    const completedExperiments = task.experiments.filter(e => e.status === 'completed').length;

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
                                    <BreadcrumbPage>{evalId}</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                    {/* <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold">GAIA</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                评估数据集: GAIA数据集 | 总计53个Level-1的测试用例
                            </div>
                        </div>
                    </div> */}
                </div>
            </div>
            {/* <div className="flex-1 overflow-y-auto px-2">
                <Tabs defaultValue="settings">
                    <TabsList className="py-4 bg-muted rounded-lg flex-none">
                        <TabsTrigger value="settings" className="p-4">
                            设置
                        </TabsTrigger>
                        <TabsTrigger value="datasets" className="p-4">
                            数据集
                        </TabsTrigger>
                        <TabsTrigger value="results" className="p-4">
                            实验
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="settings" className="py-4">
                        <EvalConfigCard eval_id={undefined} />
                    </TabsContent>
                    <TabsContent value="datasets" className="py-4">
                        <div className="overflow-y-auto">
                            <DatasetTableCard evalId={evalId} />             
                        </div>
                    </TabsContent>
                    <TabsContent value="results" className="py-4">
                        <div className="overflow-y-auto">
                            <ExperimentTableCard evalId={evalId} />             
                        </div>
                    </TabsContent>
                </Tabs>
            </div> */}
            <div className="px-2 max-w-6xl">
                <h1 className="text-2xl font-bold mb-6">任务概览: {task.id}</h1>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {/* 任务设置卡片 */}
                    <Card className="flex flex-col h-full">
                    <CardHeader>
                        <CardTitle>任务设置</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow">
                        <div className="space-y-3">
                        <div>
                            <h3 className="font-semibold mb-1">模型</h3>
                            <p className="text-muted-foreground">{task.model}</p>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-1">评估指标 (MCP)</h3>
                            <p className="text-muted-foreground">{task.mcp}</p>
                        </div>
                        </div>
                    </CardContent>
                    <CardFooter>
                        <Button variant="outline" className="w-full" onClick={() => router.push(`/evaluation/${task.id}/settings`)}>
                            {/* <Link to={`/tasks/${task.id}/settings`}></Link> */}
                            查看设置详情
                        </Button>
                    </CardFooter>
                    </Card>

                    {/* 数据集卡片 */}
                    <Card className="flex flex-col h-full">
                    <CardHeader>
                        <CardTitle>数据集信息</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow">
                        <div className="space-y-3">
                        <div>
                            <h3 className="font-semibold mb-1">名称</h3>
                            <p className="text-muted-foreground">{task.dataset.name}</p>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-1">条目数量</h3>
                            <p className="text-muted-foreground">{task.dataset.entries.toLocaleString()} 条</p>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-1">数据状态</h3>
                            <Badge variant="secondary">已验证</Badge>
                        </div>
                        </div>
                    </CardContent>
                    <CardFooter>
                        <Button variant="outline" className="w-full" onClick={() => router.push(`/evaluation/${task.id}/datasets`)}>
                            {/* <Link to={`/tasks/${task.id}/dataset`}>查看数据集详情</Link> */}
                            查看数据集详情
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
                            <div className="text-2xl font-bold">{totalExperiments}</div>
                            <div className="text-sm text-muted-foreground">总实验数</div>
                            </div>
                            <div className="text-center p-3 bg-muted rounded-lg">
                            <div className="text-2xl font-bold">{completedExperiments}</div>
                            <div className="text-sm text-muted-foreground">已完成</div>
                            </div>
                        </div>
                        
                        <div>
                            <h3 className="font-semibold mb-2">实验状态</h3>
                            <div className="flex flex-wrap gap-2">
                            {task.experiments.map(exp => (
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
                            ))}
                            </div>
                        </div>
                        </div>
                    </CardContent>
                    <CardFooter>
                        <Button variant="outline" className="w-full" onClick={() => router.push(`/evaluation/${task.id}/experiments`)}>
                            {/* <Link to={`/tasks/${task.id}/experiments`}>查看所有实验</Link> */}
                            查看所有实验
                        </Button>
                    </CardFooter>
                    </Card>
                </div>
                </div>
        </div>
    );
}
