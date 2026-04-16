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
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
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
  Settings2,
  Loader2,
  MoreHorizontal,
  ShieldCheck,
  Check,
  X,
} from 'lucide-react';
import { RunConfigFormDialog } from '@/app/evaluation/components/runconfig-form-dialog';
import { RunConfig } from '@/app/evaluation/[datasetId]/types';
import { REACT_PROMPT } from '@/app/common/prompts';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

export default function RunConfigsPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { t } = useI18n();
  const default_eval_run_config: RunConfig = {
    id: '',
    name: '',
    model_id: '',
    mcp_ids: [],
    kb_ids: [],
    enable_search: false,
    enable_vision: false,
    enable_agent: false,
    enable_input_guardrail: false,
    enable_output_guardrail: false,
    guardrail_hint: t('apps.guardrailHint'),
    prompts: {
      react: REACT_PROMPT,
    },
  };

  const { datasetId } = use(params);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [evalRunConfigs, setRunConfigs] = useState<RunConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const pageSize = 10;
  const [llms, setLlms] = useState<LlmConfig[]>([]);
  const [mcps, setMcps] = useState<McpConfig[]>([]);
  const [kbs, setKbs] = useState<KbConfig[]>([]);

  const [isNewSettingsOpen, setIsNewSettingsOpen] = useState(false);
  const [isCreateLoading, setIsCreateLoading] = useState(false);
  const [isEditSetting, setIsEditSetting] = useState(false);
  const [editConfig, setEditConfig] = useState<RunConfig>(default_eval_run_config);

  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchConfigs = async () => {
      setIsLoading(true);
      try {
        const [datasetRes, llmRes, mcpRes, kbRes] = await Promise.all([
          tenantFetch(
            `/api/config/evaluation/${datasetId}/runconfigs?page=${page}&size=${pageSize}`,
          ),
          tenantFetch(`/api/config/llms`),
          tenantFetch(`/api/config/mcps`),
          tenantFetch(`/api/config/knowledgebases`),
        ]);

        if (!datasetRes.ok) throw new Error(t('evaluation.fetchEvalTaskListFailed'));
        const json_data = await datasetRes.json();
        setRunConfigs(json_data.data.items);
        setTotalPages(json_data.data.pages || 1);

        const llmData = (await llmRes.json())?.data.items || [];
        setLlms([...llmData]);
        const mcpData = ((await mcpRes.json())?.data.items as McpConfig[]) || [];
        setMcps([...mcpData]);
        const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
        setKbs([...kbData]);
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

  const createNewRunConfig = async (data: RunConfig) => {
    try {
      if (!isEditSetting) {
        const res = await tenantFetch(`/api/config/evaluation/${datasetId}/runconfigs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        if (!res.ok) {
          alert(t('evaluation.createFailed'));
          return;
        }
        const result = await res.json();
        setRunConfigs((prev) => [...prev, result.data]);
      } else {
        const res = await tenantFetch(
          `/api/config/evaluation/${datasetId}/runconfigs/${data.id}`,
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
        setRunConfigs((prev) =>
          prev.map((config) =>
            config.id === result.data.id ? result.data : config,
          ),
        );
      }
    } catch (error) {
      console.error('Create/Update failed:', error);
    } finally {
      setIsCreateLoading(false);
      setIsNewSettingsOpen(false);
    }
  };

  const renderBadges = (
    ids: string[],
    configs: McpConfig[] | KbConfig[],
    maxShow = 2,
  ) => {
    if (!ids || ids.length === 0)
      return <span className="text-[11px] text-muted-foreground">—</span>;

    const idToNameMap = Object.fromEntries(configs.map((config) => [config.id, config.name]));
    const names = ids.map((id) => idToNameMap[id] || id);
    const visible = names.slice(0, maxShow);
    const hidden = names.slice(maxShow);

    return (
      <div className="flex flex-wrap items-center gap-1">
        {visible.map((name, idx) => (
          <Badge
            key={idx}
            variant="secondary"
            className="h-5 px-1.5 text-[10px] font-normal max-w-[120px] truncate"
          >
            {name}
          </Badge>
        ))}
        {hidden.length > 0 && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                  +{hidden.length}
                </Badge>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                <div className="space-y-0.5">
                  {hidden.map((name, i) => (
                    <div key={i} className="text-xs">
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

  const renderBoolBadge = (value: boolean) =>
    value ? (
      <Badge className="h-5 px-1.5 text-[10px] bg-green-500/10 text-green-700 border-green-500/30 hover:bg-green-500/10">
        <Check className="w-2.5 h-2.5 mr-0.5" />
        {t('evaluation.enabled')}
      </Badge>
    ) : (
      <span className="text-[11px] text-muted-foreground">—</span>
    );

  const renderGuardrailStatus = (config: RunConfig) => {
    const hasInput = config.enable_input_guardrail;
    const hasOutput = config.enable_output_guardrail;
    const hint = config.guardrail_hint;

    if (!hasInput && !hasOutput) {
      return <span className="text-[11px] text-muted-foreground">—</span>;
    }

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge
              variant="outline"
              className="h-5 px-1.5 text-[10px] bg-blue-500/10 text-blue-700 border-blue-500/30 cursor-help"
            >
              <ShieldCheck className="w-2.5 h-2.5 mr-0.5" />
              {hasInput && hasOutput ? t('evaluation.enabled') : t('evaluation.details')}
            </Badge>
          </TooltipTrigger>
          <TooltipContent className="max-w-sm p-3">
            <div className="space-y-1 text-xs">
              <div>
                <strong>{t('evaluation.inputGuardrail')}</strong>
                {hasInput ? t('evaluation.enabled') : t('evaluation.disabled')}
              </div>
              <div>
                <strong>{t('evaluation.outputGuardrail')}</strong>
                {hasOutput ? t('evaluation.enabled') : t('evaluation.disabled')}
              </div>
              {hint && (
                <div>
                  <strong>{t('evaluation.hint')}</strong>
                  <div className="mt-1 text-[10px] bg-muted p-2 rounded break-all">
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
      const res = await tenantFetch(
        `/api/config/evaluation/${datasetId}/runconfigs/${config_id}`,
        {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
        },
      );
      if (!res.ok) throw new Error(t('evaluation.deleteFailed'));
      setRunConfigs((prev) => prev.filter((config) => config.id !== config_id));
    } catch (err: any) {
      console.log(t('evaluation.deleteRunConfigError'), err);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-3">
        {/* Toolbar */}
        <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
          <p className="text-[11px] text-muted-foreground">
            {t('evaluation.runSettingsDesc')}
          </p>
          <RunConfigFormDialog
            mode={isEditSetting ? 'edit' : 'new'}
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
              <Button size="sm" className="text-xs h-7" onClick={() => setIsEditSetting(false)}>
                <Plus className="mr-1 h-3 w-3" />
                {t('evaluation.newRun')}
              </Button>
            </DialogTrigger>
          </Dialog>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t('evaluation.loadingConfigs')}
          </div>
        ) : evalRunConfigs.length === 0 ? (
          <div className="empty-state mt-4">
            <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
              <Settings2 className="w-7 h-7" />
            </div>
            <p className="text-base font-semibold mb-1">{t('common.noData')}</p>
            <p className="text-sm text-muted-foreground text-center max-w-md mb-4">
              {t('evaluation.runSettingsDesc')}
            </p>
            <Button size="sm" onClick={() => setIsNewSettingsOpen(true)}>
              <Plus className="w-4 h-4 mr-1" />
              {t('evaluation.newRun')}
            </Button>
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden bg-background">
            <Table className="table-modern">
              <TableHeader>
                <TableRow className="h-8">
                  <TableHead className="w-[180px] text-xs text-muted-foreground pl-4 py-1">
                    {t('evaluation.name')}
                  </TableHead>
                  <TableHead className="w-[160px] text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.baseModel')}
                  </TableHead>
                  <TableHead className="text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.mcp')}
                  </TableHead>
                  <TableHead className="text-xs text-muted-foreground px-2 py-1">
                    {t('knowledgebase.title')}
                  </TableHead>
                  <TableHead className="w-[80px] text-xs text-muted-foreground px-2 py-1 text-center">
                    {t('evaluation.webSearch')}
                  </TableHead>
                  <TableHead className="w-[80px] text-xs text-muted-foreground px-2 py-1 text-center">
                    {t('evaluation.agentic')}
                  </TableHead>
                  <TableHead className="w-[100px] text-xs text-muted-foreground px-2 py-1 text-center">
                    {t('evaluation.guardrailStatus')}
                  </TableHead>
                  <TableHead className="w-[60px] text-xs text-muted-foreground px-2 py-1 text-right pr-3">
                    {t('common.operations')}
                  </TableHead>
                </TableRow>
              </TableHeader>

              <TableBody>
                {evalRunConfigs.map((config) => (
                  <TableRow key={config.id} className="group h-10">
                    <TableCell className="pl-4 px-2 py-0.5">
                      <div className="flex flex-col">
                        <span className="text-xs font-medium truncate">{config.name}</span>
                        <span className="text-[10px] font-mono text-muted-foreground">
                          {config.id.slice(0, 8)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-[11px] px-2 py-0.5 truncate">
                      {config.model_id || (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell className="px-2 py-0.5">
                      {renderBadges(config.mcp_ids, mcps)}
                    </TableCell>
                    <TableCell className="px-2 py-0.5">
                      {renderBadges(config.kb_ids, kbs)}
                    </TableCell>
                    <TableCell className="px-2 py-0.5 text-center">
                      {renderBoolBadge(config.enable_search)}
                    </TableCell>
                    <TableCell className="px-2 py-0.5 text-center">
                      {renderBoolBadge(config.enable_agent)}
                    </TableCell>
                    <TableCell className="px-2 py-0.5 text-center">
                      {renderGuardrailStatus(config)}
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

      {!isLoading && evalRunConfigs.length > 0 && (
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
