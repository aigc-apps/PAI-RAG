'use client';
import React from 'react';
import { useState, useEffect, use } from 'react';
import { useI18n } from '@/app/providers/i18n';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Dialog, DialogTrigger } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { LlmConfig } from '@/app/config/model/llm/page';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Plus,
  Pencil,
  Trash2,
  ClipboardCheck,
  MoreHorizontal,
} from 'lucide-react';
import { PageLoading } from '@/components/ui/loading';
import { EvalConfigFormDialog } from '@/app/evaluation/components/evalconfig-form-dialog';
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

const default_evaluator_config: EvaluatorConfig = {
  id: '',
  name: '',
  type: '',
  model_id: '',
  case_sensitive: false,
  ignore_punctuation: false,
};

export default function EvaluatorConfigsPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { t } = useI18n();

  const { datasetId } = use(params);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [evaluatorConfigs, setEvaluatorConfigs] = useState<EvaluatorConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const pageSize = 10;
  const [llms, setLlms] = useState<LlmConfig[]>([]);

  const [isNewSettingsOpen, setIsNewSettingsOpen] = useState(false);
  const [isCreateLoading, setIsCreateLoading] = useState(false);
  const [isEditSetting, setIsEditSetting] = useState(false);
  const [editConfig, setEditConfig] = useState<EvaluatorConfig>(default_evaluator_config);
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchConfigs = async () => {
      setIsLoading(true);
      try {
        const [datasetRes, llmRes] = await Promise.all([
          tenantFetch(
            `/api/config/evaluation/${datasetId}/evalconfigs?page=${page}&size=${pageSize}`,
          ),
          tenantFetch(`/api/config/llms`),
        ]);

        if (!datasetRes.ok) throw new Error(t('evaluation.fetchEvalTasksFailed'));
        const json_data = await datasetRes.json();
        setEvaluatorConfigs(json_data.data.items);
        setTotalPages(json_data.data.pages || 1);

        const llmData = (await llmRes.json())?.data.items || [];
        setLlms([...llmData]);
      } catch (err: any) {
        console.log(err || t('evaluation.loadDatasetFailed'));
      } finally {
        setIsLoading(false);
      }
    };
    fetchConfigs();
  }, [page, datasetId, t, tenantFetch]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const createNewEvaluatorConfig = async (data: EvaluatorConfig) => {
    try {
      if (!isEditSetting) {
        const res = await tenantFetch(
          `/api/config/evaluation/${datasetId}/evalconfigs`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
          },
        );
        if (!res.ok) {
          alert(t('evaluation.createFailed'));
          return;
        }
        const result = await res.json();
        setEvaluatorConfigs((prev) => [...prev, result.data]);
      } else {
        const res = await tenantFetch(
          `/api/config/evaluation/${datasetId}/evalconfigs/${data.id}`,
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
          },
        );
        if (!res.ok) {
          alert(t('evaluation.updateFailed'));
          return;
        }
        const result = await res.json();
        setEvaluatorConfigs((prev) =>
          prev.map((config) =>
            config.id === result.data.id ? result.data : config,
          ),
        );
      }
    } catch (error) {
      console.error(t('evaluation.createFailed'), error);
    } finally {
      setIsCreateLoading(false);
      setIsNewSettingsOpen(false);
    }
  };

  const onDelete = async (config_id: string) => {
    try {
      const res = await tenantFetch(
        `/api/config/evaluation/${datasetId}/evalconfigs/${config_id}`,
        {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
        },
      );
      if (!res.ok) throw new Error(t('evaluation.deleteFailed'));
      setEvaluatorConfigs((prev) => prev.filter((config) => config.id !== config_id));
    } catch (err: any) {
      console.log(t('evaluation.deleteTaskError'), err);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto p-6">
        {/* Toolbar */}
        <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
          <p className="text-[11px] text-muted-foreground">
            {t('evaluation.evaluatorSettingsDesc')}
          </p>
          <EvalConfigFormDialog
            mode={isEditSetting ? 'edit' : 'new'}
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
              <Button size="sm" className="text-xs h-7" onClick={() => setIsEditSetting(false)}>
                <Plus className="mr-1 h-3 w-3" />
                {t('evaluation.newEvaluator')}
              </Button>
            </DialogTrigger>
          </Dialog>
        </div>

        {isLoading ? (
          <PageLoading label={t('evaluation.loadingConfigs')} />
        ) : evaluatorConfigs.length === 0 ? (
          <div className="empty-state mt-4">
            <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
              <ClipboardCheck className="w-7 h-7" />
            </div>
            <p className="text-base font-semibold mb-1">{t('evaluation.noData')}</p>
            <p className="text-sm text-muted-foreground text-center max-w-md mb-4">
              {t('evaluation.evaluatorSettingsDesc')}
            </p>
            <Button size="sm" onClick={() => setIsNewSettingsOpen(true)}>
              <Plus className="w-4 h-4 mr-1" />
              {t('evaluation.newEvaluator')}
            </Button>
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden bg-background">
            <Table className="table-modern">
              <TableHeader>
                <TableRow className="h-8">
                  <TableHead className="w-[220px] text-xs text-muted-foreground pl-4 py-1">
                    {t('evaluation.name')}
                  </TableHead>
                  <TableHead className="w-[140px] text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.evaluatorType')}
                  </TableHead>
                  <TableHead className="text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.evaluationSettings')}
                  </TableHead>
                  <TableHead className="w-[60px] text-xs text-muted-foreground px-2 py-1 text-right pr-3">
                    {t('evaluation.operations')}
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {evaluatorConfigs.map((config) => (
                  <TableRow key={config.id} className="group h-10">
                    <TableCell className="pl-4 px-2 py-0.5">
                      <div className="flex flex-col">
                        <span className="text-xs font-medium truncate">{config.name}</span>
                        <span className="text-[10px] font-mono text-muted-foreground">
                          {config.id.slice(0, 8)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="px-2 py-0.5">
                      <Badge
                        variant="outline"
                        className={`h-5 px-1.5 text-[10px] ${
                          config.type === 'ExactMatch'
                            ? 'bg-blue-500/10 text-blue-700 border-blue-500/30'
                            : 'bg-purple-500/10 text-purple-700 border-purple-500/30'
                        }`}
                      >
                        {config.type === 'ExactMatch'
                          ? t('evaluation.exactMatch')
                          : t('evaluation.llmJudge')}
                      </Badge>
                    </TableCell>
                    <TableCell className="px-2 py-0.5">
                      {config.type === 'ExactMatch' ? (
                        <div className="flex flex-wrap gap-1 text-[10px] text-muted-foreground">
                          <Badge variant="secondary" className="h-5 px-1.5 text-[10px] font-normal">
                            {t('evaluation.caseSensitive')}:{' '}
                            {config.case_sensitive ? t('common.yes') : t('common.no')}
                          </Badge>
                          <Badge variant="secondary" className="h-5 px-1.5 text-[10px] font-normal">
                            {t('evaluation.ignorePunctuation')}:{' '}
                            {config.ignore_punctuation ? t('common.yes') : t('common.no')}
                          </Badge>
                        </div>
                      ) : config.type === 'LLMJudge' ? (
                        <span className="text-[11px] text-muted-foreground">
                          {t('evaluation.model')}:{' '}
                          <span className="font-mono text-foreground/80">
                            {config.model_id || t('evaluation.notSpecified')}
                          </span>
                        </span>
                      ) : (
                        <span className="text-[11px] text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right pr-3 px-2 py-0.5">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="menu-compact">
                          <DropdownMenuItem
                            onSelect={() => {
                              setEditConfig(config);
                              setIsEditSetting(true);
                              setIsNewSettingsOpen(true);
                            }}
                          >
                            <Pencil />
                            {t('common.edit')}
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            onSelect={() => onDelete(config.id)}
                            className="text-destructive focus:text-destructive"
                          >
                            <Trash2 />
                            {t('common.delete')}
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      {!isLoading && evaluatorConfigs.length > 0 && (
        <div className="flex-none border-t border-border bg-background/60 backdrop-blur-sm py-2">
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      )}
    </div>
  );
}
