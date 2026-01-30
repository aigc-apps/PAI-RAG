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
import { useI18n } from '@/app/providers/i18n';
import { BookOpen, Settings, FlaskConical } from "lucide-react";
import EvalDatasetsDetailsPage from '@/app/evaluation/[datasetId]/samples/page';
import EvalExperimentsDetailsPage from '@/app/evaluation/[datasetId]/experiments/page';
import RunConfigsPage from '@/app/evaluation/[datasetId]/runconfigs/page';
import EvaluatorConfigsPage from '@/app/evaluation/[datasetId]/evalconfigs/page';
import { EvalConfig } from "@/app/evaluation/[datasetId]/types";

export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ datasetId: string }> }
) {
    const { t } = useI18n();
  
    const { datasetId } = use(params);
    const router = useRouter();
    const [evaluation, setEvaluationConfig] = useState<EvalConfig>(); // Evaluation configuration

    useEffect(() => {
        const fetchKbConfigs = async () => {
            try {
                const [evalRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${datasetId}`)
                ]);

                if (!evalRes.ok) throw new Error(t('evaluation.fetchError'));
                const json_data = await evalRes.json();
                const kb_data = json_data.data;
                console.log('Evaluation task details data:', kb_data);
                setEvaluationConfig(kb_data); // Update state
            } catch (err: any) {
                toast.error(err.message);
            }
        };
        fetchKbConfigs();
    }, [datasetId, t]);

    if (!evaluation) {
        return <div className="p-6">{t('evaluation.loading')}</div>;
    }

    return (
        <div className="flex flex-col h-screen px-6">
            <div className="p-0 space-y-2">
                <div className="mb-2 flex items-center gap-2">
                    {/* Breadcrumb navigation */}
                    <Breadcrumb>
                        <BreadcrumbList>
                            <BreadcrumbItem>
                                <BreadcrumbLink asChild>
                                    <Button
                                        variant="link"
                                        className="px-0"
                                        onClick={() => router.push('/evaluation')}
                                    >
                                        {t('sidebar.evaluation')}
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
                        <h1 className="text-xl font-medium">{t('evaluation.overview')}: {evaluation.name}</h1>
                        <div className="text-sm text-muted-foreground mt-1">
                            {evaluation.description}
                        </div>
                    </div>
                </div>
            </div>
            <div className="px-2 w-full flex-1 flex flex-col min-h-0 pt-4">
                <Tabs defaultValue="experiments" className="w-full flex-1 flex flex-col min-h-0">
                    <TabsList className="p-1 shrink-0">
                        <TabsTrigger
                            key="datasets"
                            value="datasets"
                            className="px-4"
                        >
                            <BookOpen className="h-3.5 w-3.5" /> {t('evaluation.samples')}
                        </TabsTrigger>
                        <TabsTrigger
                            key="experiments"
                            value="experiments"
                            className="px-4"
                        >
                            <FlaskConical className="h-3.5 w-3.5" />{t('evaluation.runHistory')}
                        </TabsTrigger>
                        <TabsTrigger
                            key="runconfigs"
                            value="runconfigs"
                            className="px-4"
                        >
                            <Settings className="h-3.5 w-3.5" />{t('evaluation.runSettings')}
                        </TabsTrigger>
                        <TabsTrigger
                            key="evalconfigs"
                            value="evalconfigs"
                            className="px-4"
                        >
                            <Settings className="h-3.5 w-3.5" />{t('evaluation.evaluatorSettings')}
                        </TabsTrigger>
                    </TabsList>
                    <div className="flex-1 min-h-0 overflow-hidden">
                        <TabsContent key="datasets" value="datasets" className="h-full flex flex-col min-h-0">
                            <EvalDatasetsDetailsPage params={params} />
                        </TabsContent>
                        <TabsContent key="experiments" value="experiments" className="h-full flex flex-col min-h-0">
                            <EvalExperimentsDetailsPage params={params} />
                        </TabsContent>
                        <TabsContent key="runconfigs" value="runconfigs" className="h-full flex flex-col min-h-0">
                            <RunConfigsPage params={params} />
                        </TabsContent>
                        <TabsContent key="evalconfigs" value="evalconfigs" className="h-full flex flex-col min-h-0">
                            <EvaluatorConfigsPage params={params} />
                        </TabsContent>
                    </div>
                </Tabs>

            </div>
        </div>
    );
}
