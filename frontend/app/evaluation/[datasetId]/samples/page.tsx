'use client';

import React, { useState, useEffect, use } from 'react';
import { useRouter } from 'next/navigation';
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
import {
  ChevronDown,
  ChevronUp,
  PlayIcon,
  CopyIcon,
  UploadIcon,
  Eye,
  Trash2,
  Pencil,
  FileText,
  Info,
  BookOpen,
  MoreHorizontal,
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { RunConfig } from '@/app/evaluation/[datasetId]/types';
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';

import { toast } from 'sonner';

import { SampleDetailDialog } from '@/app/evaluation/components/sample-detail-dialog';
import { useDatasetActions } from '@/app/evaluation/[datasetId]/samples/useDatasetActions';
import { SampleItem } from '@/app/evaluation/[datasetId]/types';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { PageLoading, Spinner } from '@/components/ui/loading';

export default function EvalDatasetsDetailsPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { datasetId } = use(params);
  const { t } = useI18n();

  // === State management ===
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [datasets, setDatasets] = useState<SampleItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const pageSize = 10;

  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [allItems, setAllItems] = useState<SampleItem[]>([]);
  const [totalItems, setTotalItems] = useState(0);

  // Dialog state
  const [isRunSingleDetailOpen, setIsRunSingleDetailOpen] = useState(false);
  const [isRunBatchDetailOpen, setIsRunBatchDetailOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [editingSample, setEditingSample] = useState<SampleItem | null>(null);
  const [dialogMode, setDialogMode] = useState<'view' | 'edit'>('view');

  // Experiment configuration related
  const [experimentName, setExperimentName] = useState('');
  const [experimentDescription, setExperimentDescription] = useState('');
  const [runConfigId, setRunConfigId] = useState<string>('');
  const [runConfigs, setRunConfigs] = useState<RunConfig[]>([]);
  const [evaluatorConfigId, setEvaluatorConfigId] = useState<string>('');
  const [evaluatorConfigs, setEvaluatorConfigs] = useState<EvaluatorConfig[]>([]);

  // Upload state
  const [uploading, setUploading] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const { tenantFetch } = useTenantFetch();
  const { runSamples, deleteSample, uploadFile } = useDatasetActions({ datasetId });

  // === Data loading ===
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      try {
        const [datasetRes, runConfigsRes, evalConfigsRes] = await Promise.all([
          tenantFetch(
            `/api/config/evaluation/${datasetId}/samples?page=${page}&size=${pageSize}`,
          ),
          tenantFetch(`/api/config/evaluation/${datasetId}/runconfigs`),
          tenantFetch(`/api/config/evaluation/${datasetId}/evalconfigs`),
        ]);

        if (datasetRes.ok) {
          const data = await datasetRes.json();
          setDatasets(data.data.items);
          setTotalItems(data.data.total);
          setTotalPages(data.data.pages);
        }

        if (runConfigsRes.ok) {
          const configData = await runConfigsRes.json();
          setRunConfigs(configData.data.items);
        }

        if (evalConfigsRes.ok) {
          const evalConfigData = await evalConfigsRes.json();
          setEvaluatorConfigs(evalConfigData.data.items);
        }

        // Load all data for select-all
        const tmpPageSize = 1000;
        const firstPageRes = await tenantFetch(
          `/api/config/evaluation/${datasetId}/samples?page=1&size=${tmpPageSize}`,
        );
        if (!firstPageRes.ok) throw new Error(t('evaluation.fetchSampleListFailed'));
        const json_data = await firstPageRes.json();
        const tmpAllItems: SampleItem[] = [];

        for (let curPage = 1; curPage <= json_data.data.pages; curPage++) {
          const response = await tenantFetch(
            `/api/config/evaluation/${datasetId}/samples?page=${curPage}&size=${tmpPageSize}`,
          );
          const data = await response.json();
          tmpAllItems.push(...data.data.items);
        }
        setAllItems(tmpAllItems);
      } catch (err) {
        console.error(err);
        toast.error(t('evaluation.loadDataFailed'));
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [page, datasetId, datasets.length, t, tenantFetch]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const toggleRowExpansion = (id: string) => {
    setExpandedRows((prev) => {
      const newSet = new Set(prev);
      newSet.has(id) ? newSet.delete(id) : newSet.add(id);
      return newSet;
    });
  };

  const copyId = (id: string) => {
    navigator.clipboard.writeText(id);
    toast.success(t('evaluation.copySuccess'));
  };

  const isItemSelected = (id: string) => selectedItems.has(id);

  const handleSelectItem = (id: string) => {
    setSelectedItems((prev) => {
      const newSet = new Set(prev);
      newSet.has(id) ? newSet.delete(id) : newSet.add(id);
      return newSet;
    });
  };

  const isAllSelected = selectedItems.size === allItems.length && allItems.length > 0;

  const handleBatchRun = async () => {
    const ids = isAllSelected ? allItems.map((item) => item.id) : Array.from(selectedItems);

    await runSamples({
      name: experimentName,
      description: experimentDescription,
      sample_ids: ids,
      run_config_id: runConfigId,
      evaluator_config_id: evaluatorConfigId,
    });

    setIsRunBatchDetailOpen(false);
    setExperimentName('');
    setExperimentDescription('');
  };

  const runSingleSample = async (id: string) => {
    await runSamples({
      name: experimentName,
      description: experimentDescription,
      sample_ids: [id],
      run_config_id: runConfigId,
      evaluator_config_id: evaluatorConfigId,
    });
    setIsRunSingleDetailOpen(false);
    setExperimentName('');
    setExperimentDescription('');
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (!files?.length) return;

    const file = files[0];
    if (file.size > 1000 * 1024 * 1024) {
      toast.error(t('evaluation.fileSizeExceeded'));
      return;
    }

    setUploading(true);
    try {
      const uploadedItems = await uploadFile(file);
      setDatasets((prev) => [...prev, ...uploadedItems]);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleEyeClick = (item: SampleItem) => {
    setEditingSample(item);
    setDialogMode('view');
    setIsEditOpen(true);
  };

  const handleEditClick = (item: SampleItem) => {
    setEditingSample(item);
    setDialogMode('edit');
    setIsEditOpen(true);
  };

  const handleSaveEdit = async (updatedSample: SampleItem) => {
    try {
      const response = await tenantFetch(
        `/api/config/evaluation/${datasetId}/samples/${updatedSample.id}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updatedSample),
        },
      );

      if (!response.ok) throw new Error(t('evaluation.updateFailed'));

      setDatasets((prev) =>
        prev.map((item) => (item.id === updatedSample.id ? updatedSample : item)),
      );
      toast.success(t('evaluation.updateSuccess'));
      setIsEditOpen(false);
    } catch (error) {
      toast.error(t('evaluation.updateFailed'));
    }
  };

  // Shared run-config form content (used by both single and batch run dialogs)
  const modifyRunConfig = (selected_ids: Set<string>) => (
    <div className="space-y-3 py-2">
      <div className="grid grid-cols-4 items-center gap-3">
        <Label htmlFor="name" className="text-right text-xs">
          {t('evaluation.experimentNameLabel')}
        </Label>
        <Input
          id="name"
          value={experimentName}
          onChange={(e) => setExperimentName(e.target.value)}
          className="col-span-3 h-8 text-xs"
          placeholder={t('evaluation.experimentNamePlaceholder')}
        />
      </div>
      <div className="grid grid-cols-4 items-start gap-3">
        <Label htmlFor="description" className="text-right text-xs pt-2">
          {t('evaluation.descriptionLabel')}
        </Label>
        <Textarea
          id="description"
          value={experimentDescription}
          onChange={(e) => setExperimentDescription(e.target.value)}
          className="col-span-3 text-xs resize-none"
          placeholder={t('evaluation.descriptionPlaceholder')}
          rows={2}
        />
      </div>
      <div className="grid grid-cols-4 items-start gap-3">
        <Label className="text-right text-xs pt-1">{t('evaluation.dataSampleIds')}</Label>
        <div className="col-span-3">
          <div className="max-h-24 overflow-y-auto rounded-md border border-border p-2 bg-muted/30">
            <div className="flex flex-wrap gap-1">
              {[...selected_ids].map((id) => (
                <Badge
                  key={id}
                  variant="secondary"
                  className="bg-green-500/10 text-green-700 border-green-500/30 font-mono text-[10px] h-5 px-1.5"
                >
                  {id.substring(0, 8)}
                </Badge>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-4 items-center gap-3">
        <Label className="text-right text-xs">{t('evaluation.runSettingsLabel')}</Label>
        <div className="col-span-3">
          {runConfigs.length > 0 ? (
            <Select onValueChange={setRunConfigId}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder={t('evaluation.selectRunSettings')} />
              </SelectTrigger>
              <SelectContent>
                {runConfigs.map((config) => (
                  <SelectItem key={config.id} value={config.id} className="text-xs">
                    {config.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <p className="text-xs text-muted-foreground">{t('evaluation.noRunConfig')}</p>
          )}
        </div>
      </div>
      <div className="grid grid-cols-4 items-center gap-3">
        <Label className="text-right text-xs">{t('evaluation.evaluatorSettingsLabel')}</Label>
        <div className="col-span-3">
          {evaluatorConfigs.length > 0 ? (
            <Select onValueChange={setEvaluatorConfigId}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder={t('evaluation.selectEvaluatorSettings')} />
              </SelectTrigger>
              <SelectContent>
                {evaluatorConfigs.map((config) => (
                  <SelectItem key={config.id} value={config.id} className="text-xs">
                    {config.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <p className="text-xs text-muted-foreground">{t('evaluation.noEvaluatorConfig')}</p>
          )}
        </div>
      </div>
    </div>
  );

  const selectedCount = isAllSelected ? totalItems : selectedItems.size;

  return (
    <div className="flex flex-col h-full min-h-0">
      <SampleDetailDialog
        open={isEditOpen}
        onOpenChange={setIsEditOpen}
        sample={editingSample}
        mode={dialogMode}
        onSave={dialogMode === 'edit' ? handleSaveEdit : undefined}
      />

      <div className="flex-1 min-h-0 overflow-y-auto p-6">
        {/* Toolbar */}
        <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <Info className="h-3 w-3" />
            <span className="font-medium">{t('evaluation.dataRequirements')}</span>
            <span>{t('evaluation.dataRequirementsDesc')}</span>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {selectedCount > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="text-xs h-7"
                onClick={() => setSelectedItems(new Set())}
              >
                {t('evaluation.clearSelection')}
              </Button>
            )}
            <Dialog open={isRunBatchDetailOpen} onOpenChange={setIsRunBatchDetailOpen}>
              <DialogTrigger asChild>
                <Button
                  size="sm"
                  disabled={selectedItems.size === 0 && !isAllSelected}
                  className="text-xs h-7"
                >
                  <PlayIcon className="mr-1 h-3 w-3" />
                  {isAllSelected
                    ? t('evaluation.runExperimentAll', { count: totalItems })
                    : t('evaluation.runExperimentSelected', { count: selectedItems.size })}
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[640px]">
                <DialogHeader>
                  <DialogTitle className="text-base">
                    {t('evaluation.createNewExperimentBatch')}
                  </DialogTitle>
                  <DialogDescription className="text-xs">
                    {t('evaluation.experimentNameDesc')}
                  </DialogDescription>
                </DialogHeader>
                {modifyRunConfig(selectedItems)}
                <DialogFooter>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsRunBatchDetailOpen(false)}
                  >
                    {t('common.cancel')}
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleBatchRun}
                    disabled={!experimentName.trim()}
                  >
                    {t('evaluation.run')}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Button
              size="sm"
              variant="outline"
              className="text-xs h-7"
              onClick={() => document.getElementById('file-upload')?.click()}
              disabled={uploading}
            >
              {uploading ? (
                <>
                  <Spinner size="sm" className="mr-1" />
                  {t('evaluation.uploading')}
                </>
              ) : (
                <>
                  <UploadIcon className="mr-1 h-3 w-3" />
                  {t('evaluation.importData')}
                </>
              )}
            </Button>
            <input
              id="file-upload"
              type="file"
              className="hidden"
              ref={fileInputRef}
              onChange={(e) => handleFileUpload(e.target.files)}
            />
          </div>
        </div>

        {/* Content */}
        {isLoading ? (
          <PageLoading />
        ) : datasets.length === 0 ? (
          <div className="empty-state mt-4">
            <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
              <BookOpen className="w-7 h-7" />
            </div>
            <p className="text-base font-semibold mb-1">{t('evaluation.noDataSamples')}</p>
            <p className="text-sm text-muted-foreground text-center max-w-md">
              {t('evaluation.dataRequirementsDesc')}
            </p>
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden bg-background">
            <Table className="table-modern">
              <TableHeader>
                <TableRow className="h-8">
                  <TableHead className="w-[40px] pl-3 py-1">
                    <Checkbox
                      checked={isAllSelected}
                      onCheckedChange={(checked) => {
                        setSelectedItems(
                          checked ? new Set(allItems.map((item) => item.id)) : new Set(),
                        );
                      }}
                    />
                  </TableHead>
                  <TableHead className="w-[140px] text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.sampleId')}
                  </TableHead>
                  <TableHead className="text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.question')}
                  </TableHead>
                  <TableHead className="w-[28%] text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.answer')}
                  </TableHead>
                  <TableHead className="w-[140px] text-xs text-muted-foreground px-2 py-1">
                    {t('evaluation.attachment')}
                  </TableHead>
                  <TableHead className="w-[60px] text-xs text-muted-foreground px-2 py-1 text-right pr-3">
                    {t('evaluation.operations')}
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasets.map((item) => {
                  const expanded = expandedRows.has(item.id);
                  return (
                    <TableRow key={item.id} className="group h-9">
                      <TableCell className="pl-3 px-2 py-0.5">
                        <Checkbox
                          checked={isItemSelected(item.id)}
                          onCheckedChange={() => handleSelectItem(item.id)}
                        />
                      </TableCell>
                      <TableCell className="px-2 py-0.5">
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            className="text-xs font-mono text-primary hover:underline truncate max-w-[100px]"
                            onClick={() => handleEyeClick(item)}
                          >
                            {item.id.substring(0, 8)}
                          </button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => copyId(item.id)}
                            title={t('evaluation.copySampleId')}
                          >
                            <CopyIcon className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>
                      <TableCell
                        className={`text-xs px-2 py-0.5 ${
                          expanded ? 'whitespace-normal break-words' : 'truncate max-w-0'
                        }`}
                        style={
                          !expanded
                            ? {
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                              }
                            : undefined
                        }
                      >
                        {item.input}
                      </TableCell>
                      <TableCell
                        className={`text-xs text-muted-foreground px-2 py-0.5 ${
                          expanded ? 'whitespace-normal break-words' : ''
                        }`}
                        style={
                          !expanded
                            ? {
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                maxWidth: 0,
                              }
                            : undefined
                        }
                      >
                        {item.expected_output}
                      </TableCell>
                      <TableCell className="px-2 py-0.5">
                        {item.eval_metadata?.file_name ? (
                          <Badge
                            variant="secondary"
                            className="h-5 px-1.5 text-[10px] font-normal max-w-[130px] truncate"
                          >
                            <FileText className="h-2.5 w-2.5 mr-1" />
                            {item.eval_metadata.file_name}
                          </Badge>
                        ) : (
                          <span className="text-[11px] text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell className="px-2 py-0.5 text-right pr-3">
                        <div className="flex items-center justify-end gap-0.5">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => toggleRowExpansion(item.id)}
                            title={
                              expanded ? t('evaluation.collapse') : t('evaluation.expand')
                            }
                          >
                            {expanded ? (
                              <ChevronUp className="h-3.5 w-3.5" />
                            ) : (
                              <ChevronDown className="h-3.5 w-3.5" />
                            )}
                          </Button>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <MoreHorizontal className="h-3.5 w-3.5" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" className="menu-compact">
                              <DropdownMenuItem
                                onSelect={(e) => {
                                  e.preventDefault();
                                  setTimeout(() => handleEyeClick(item), 0);
                                }}
                              >
                                <Eye />
                                {t('evaluation.viewDetails')}
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onSelect={(e) => {
                                  e.preventDefault();
                                  setTimeout(() => handleEditClick(item), 0);
                                }}
                              >
                                <Pencil />
                                {t('evaluation.edit')}
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onSelect={(e) => {
                                  e.preventDefault();
                                  setTimeout(() => {
                                    setEditingSample(item);
                                    setIsRunSingleDetailOpen(true);
                                  }, 0);
                                }}
                              >
                                <PlayIcon />
                                {t('evaluation.runSingle')}
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                onSelect={(e) => {
                                  e.preventDefault();
                                  setTimeout(() => deleteSample(item.id), 0);
                                }}
                                className="text-destructive focus:text-destructive"
                              >
                                <Trash2 />
                                {t('common.delete')}
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}

        {/* Single run dialog (triggered via dropdown) */}
        <Dialog open={isRunSingleDetailOpen} onOpenChange={setIsRunSingleDetailOpen}>
          <DialogContent className="sm:max-w-[640px]">
            <DialogHeader>
              <DialogTitle className="text-base">
                {t('evaluation.createNewExperimentSingle')}
              </DialogTitle>
              <DialogDescription className="text-xs">
                {t('evaluation.experimentNameDesc')}
              </DialogDescription>
            </DialogHeader>
            {editingSample && modifyRunConfig(new Set([editingSample.id]))}
            <DialogFooter>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsRunSingleDetailOpen(false)}
              >
                {t('common.cancel')}
              </Button>
              <Button
                size="sm"
                onClick={() => editingSample && runSingleSample(editingSample.id)}
                disabled={!experimentName.trim()}
              >
                {t('evaluation.run')}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Sticky footer: selection count + pagination */}
      {!isLoading && datasets.length > 0 && (
        <div className="flex-none border-t border-border bg-background/60 backdrop-blur-sm">
          {selectedCount > 0 && (
            <div className="px-6 py-1.5 border-b border-border text-xs flex items-center justify-between">
              <span className="font-medium">
                {isAllSelected
                  ? t('evaluation.selectedAll', { count: totalItems })
                  : t('evaluation.selectedCount', { count: selectedItems.size })}
              </span>
            </div>
          )}
          <div className="py-2">
            <PaginationComponent
              currentPage={page}
              totalPages={totalPages}
              onPageChange={handlePageChange}
            />
          </div>
        </div>
      )}
    </div>
  );
}
