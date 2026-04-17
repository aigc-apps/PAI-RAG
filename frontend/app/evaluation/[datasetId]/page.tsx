'use client';

import { useState, useEffect, use } from 'react';
import { useRouter } from 'next/navigation';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import React from 'react';
import { toast } from 'sonner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useI18n } from '@/app/providers/i18n';
import { BookOpen, Settings, FlaskConical, ClipboardCheck } from 'lucide-react';
import EvalDatasetsDetailsPage from '@/app/evaluation/[datasetId]/samples/page';
import EvalExperimentsDetailsPage from '@/app/evaluation/[datasetId]/experiments/page';
import RunConfigsPage from '@/app/evaluation/[datasetId]/runconfigs/page';
import EvaluatorConfigsPage from '@/app/evaluation/[datasetId]/evalconfigs/page';
import { EvalConfig } from '@/app/evaluation/[datasetId]/types';
import { HeaderPortal } from '@/components/header-portal';
import { PageLoading } from '@/components/ui/loading';

type EvalView = 'datasets' | 'experiments' | 'runconfigs' | 'evalconfigs';

const TAB_ITEMS: Array<{ value: EvalView; labelKey: string; icon: React.ReactNode }> = [
  { value: 'experiments', labelKey: 'evaluation.runHistory', icon: <FlaskConical className="w-3.5 h-3.5" /> },
  { value: 'datasets', labelKey: 'evaluation.samples', icon: <BookOpen className="w-3.5 h-3.5" /> },
  { value: 'runconfigs', labelKey: 'evaluation.runSettings', icon: <Settings className="w-3.5 h-3.5" /> },
  { value: 'evalconfigs', labelKey: 'evaluation.evaluatorSettings', icon: <ClipboardCheck className="w-3.5 h-3.5" /> },
];

export default function EvalExpDetailsPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { t } = useI18n();
  const { datasetId } = use(params);
  const router = useRouter();
  const [evaluation, setEvaluationConfig] = useState<EvalConfig>();
  const [view, setView] = useState<EvalView>('experiments');

  useEffect(() => {
    const fetchKbConfigs = async () => {
      try {
        const evalRes = await fetch(`/api/config/evaluation/${datasetId}`);
        if (!evalRes.ok) throw new Error(t('evaluation.fetchError'));
        const json_data = await evalRes.json();
        setEvaluationConfig(json_data.data);
      } catch (err: any) {
        toast.error(err.message);
      }
    };
    fetchKbConfigs();
  }, [datasetId, t]);

  if (!evaluation) {
    return <PageLoading className="h-full" label={t('evaluation.loading')} />;
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button
                  variant="link"
                  className="px-0 h-auto"
                  onClick={() => router.push('/evaluation')}
                >
                  <span suppressHydrationWarning>{t('sidebar.evaluation')}</span>
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage className="font-semibold">{evaluation.name}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        {evaluation?.id && (
          <Badge variant="secondary" className="text-[10px] font-mono bg-muted text-muted-foreground">
            ID: {evaluation.id}
          </Badge>
        )}
        {evaluation.description && (
          <span className="text-xs text-muted-foreground truncate max-w-[280px]">
            {evaluation.description}
          </span>
        )}
      </HeaderPortal>

      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {/* Inline underline tabs */}
        <div className="flex-none flex items-center gap-5 px-6 pt-3 border-b border-border">
          {TAB_ITEMS.map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => setView(item.value)}
              className={`flex items-center gap-1.5 text-xs py-2 border-b-2 -mb-[1px] transition-colors ${
                view === item.value
                  ? 'border-primary text-foreground font-medium'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              {item.icon}
              <span suppressHydrationWarning>{t(item.labelKey)}</span>
            </button>
          ))}
        </div>

        <Tabs value={view} onValueChange={(v) => setView(v as EvalView)} className="flex-1 min-h-0 flex flex-col">
          <TabsList className="sr-only" aria-hidden="true">
            {TAB_ITEMS.map((item) => (
              <TabsTrigger key={item.value} value={item.value}>
                {t(item.labelKey)}
              </TabsTrigger>
            ))}
          </TabsList>
          <div className="flex-1 min-h-0 overflow-hidden">
            <TabsContent value="datasets" className="h-full flex flex-col min-h-0 m-0">
              <EvalDatasetsDetailsPage params={params} />
            </TabsContent>
            <TabsContent value="experiments" className="h-full flex flex-col min-h-0 m-0">
              <EvalExperimentsDetailsPage params={params} />
            </TabsContent>
            <TabsContent value="runconfigs" className="h-full flex flex-col min-h-0 m-0">
              <RunConfigsPage params={params} />
            </TabsContent>
            <TabsContent value="evalconfigs" className="h-full flex flex-col min-h-0 m-0">
              <EvaluatorConfigsPage params={params} />
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </div>
  );
}
