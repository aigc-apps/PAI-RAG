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
// 评估数据类型定义
interface SampleItem {
    id: string;
    question: string;
    answer: string;
}

export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalId : string }> }
) {
    const { evalId } = use(params);
    const router = useRouter();

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
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold">GAIA</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                评估数据集: GAIA数据集 | 总计53个Level-1的测试用例
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="flex-1 overflow-y-auto px-2">
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
            </div>
        </div>
    );
}
