'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Plus, Trash2, Database, MoreHorizontal, Pencil, Clock, FlaskConical, FileStack } from 'lucide-react';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { formatBeijingTime } from '@/app/knowledgebases/utils/utils';
import { Badge } from '@/components/ui/badge';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';
import { HeaderPortal } from '@/components/header-portal';
import { PageLoading } from '@/components/ui/loading';

interface Dataset {
  id: string;
  name: string;
  description: string;
  created_at: string;
  dataset_count: number;
  experiments_count: number;
}

const EvaluationPage = () => {
  const { t } = useI18n();
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [datasets, setDatasets] = useState(Array<Dataset>);
  const [loading, setLoading] = useState(true);
  const pageSize = 12;
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isCreateLoading, setIsCreateLoading] = useState(false);
  const [datasetName, setDatasetName] = useState('');
  const [datasetDesc, setDatasetDesc] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null);
  const router = useRouter();
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchConfigs = async () => {
      setLoading(true);
      try {
        const res = await tenantFetch(
          `/api/config/evaluation?page=${page}&size=${pageSize}`,
        );
        if (!res.ok) throw new Error(t('evaluation.fetchError'));
        const json_data = await res.json();
        const data = json_data.data.items;
        setDatasets(data);
        setTotalPages(json_data.data.pages || 1);
      } catch (err: any) {
        console.log(err || t('evaluation.loadError'));
      } finally {
        setLoading(false);
      }
    };

    fetchConfigs();
  }, [page, tenantFetch, t]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const createNewEvalDataset = async () => {
    const data = {
      name: datasetName,
      description: datasetDesc || t('evaluation.defaultEvalDesc'),
      type: 'custom',
    };

    try {
      setIsCreateLoading(true);
      const res = await tenantFetch(`/api/config/evaluation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!res.ok) {
        alert(t('evaluation.createFailed'));
        return;
      }
      const upload_result = await res.json();
      setDatasets((prev) => [...prev, upload_result.data]);
      setDatasetName('');
      setDatasetDesc('');
    } catch (error) {
      console.error('创建失败:', error);
    } finally {
      setIsCreateLoading(false);
      setIsCreateOpen(false);
    }
  };

  const deleteEval = async (eval_id: string) => {
    try {
      const res = await tenantFetch(`/api/config/evaluation/${eval_id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) throw new Error(t('evaluation.deleteError'));
      setDatasets((prev) => prev.filter((config) => config.id !== eval_id));
    } catch (err: any) {
      console.log('删除评估任务出错: ', err);
    } finally {
      setDeleteTarget(null);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <div className="flex items-center gap-2">
          <h1 className="text-base font-semibold">{t('evaluation.datasetAndEval')}</h1>
          <span className="text-xs text-muted-foreground hidden md:inline">
            · {t('evaluation.datasetDescription')}
          </span>
        </div>
        <div className="ml-auto">
          <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="w-4 h-4 mr-1" />
                {t('evaluation.createDataset')}
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <FileStack className="w-4 h-4 text-primary" />
                  {t('evaluation.newEvalTask')}
                </DialogTitle>
                <DialogDescription className="text-xs">
                  {t('evaluation.evalTaskDescription')}
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-3 py-2">
                <div className="space-y-1.5">
                  <Label htmlFor="dataset_name" className="text-xs font-medium">
                    {t('evaluation.datasetName')} <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="dataset_name"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    className="h-8 text-xs"
                    placeholder={t('evaluation.datasetName')}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="dataset_desc" className="text-xs font-medium">
                    {t('evaluation.datasetDesc')}
                  </Label>
                  <Textarea
                    id="dataset_desc"
                    value={datasetDesc}
                    onChange={(e) => setDatasetDesc(e.target.value)}
                    className="text-xs resize-none"
                    rows={3}
                    placeholder={t('evaluation.datasetDesc')}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" size="sm" onClick={() => setIsCreateOpen(false)}>
                  {t('common.cancel')}
                </Button>
                <Button
                  onClick={createNewEvalDataset}
                  size="sm"
                  type="submit"
                  disabled={isCreateLoading || !datasetName.trim()}
                >
                  {isCreateLoading ? t('evaluation.submitting') : t('evaluation.submit')}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </HeaderPortal>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-5">
          {loading ? (
            <PageLoading />
          ) : datasets.length === 0 ? (
            <div className="empty-state mt-8">
              <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
                <Database className="w-7 h-7" />
              </div>
              <p className="text-base font-semibold mb-1">{t('evaluation.noDataset')}</p>
              <p className="text-sm text-muted-foreground text-center max-w-md mb-4">
                {t('evaluation.datasetDescription')}
              </p>
              <Button size="sm" onClick={() => setIsCreateOpen(true)}>
                <Plus className="w-4 h-4 mr-1" />
                {t('evaluation.createFirstDataset')}
              </Button>
            </div>
          ) : (
            <div className="rounded-lg border border-border overflow-hidden bg-background">
              <Table className="table-modern">
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[220px] pl-4 text-xs text-muted-foreground">
                      {t('evaluation.dataset')}
                    </TableHead>
                    <TableHead className="w-[80px] text-xs text-muted-foreground">
                      {t('evaluation.type')}
                    </TableHead>
                    <TableHead className="text-xs text-muted-foreground">
                      {t('evaluation.description')}
                    </TableHead>
                    <TableHead className="w-[80px] text-xs text-muted-foreground text-right">
                      {t('evaluation.sampleCount')}
                    </TableHead>
                    <TableHead className="w-[80px] text-xs text-muted-foreground text-right">
                      {t('evaluation.experimentCount')}
                    </TableHead>
                    <TableHead className="w-[160px] text-xs text-muted-foreground">
                      {t('evaluation.createTime')}
                    </TableHead>
                    <TableHead className="w-[60px] text-xs text-muted-foreground text-right pr-3">
                      {t('evaluation.actions')}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {datasets.map((dataset) => {
                    const isBuiltin = dataset.name === 'GAIA';
                    return (
                      <TableRow
                        key={dataset.id}
                        className="cursor-pointer group h-10"
                        onClick={(e) => {
                          const target = e.target as HTMLElement;
                          if (target.closest('[data-stop-click]')) return;
                          router.push(`/evaluation/${dataset.id}`);
                        }}
                      >
                        <TableCell className="pl-4">
                          <div className="flex items-center gap-2">
                            <div className="flex items-center justify-center w-7 h-7 rounded-md bg-primary/10 text-primary shrink-0 text-xs font-semibold">
                              {(dataset.name || 'D').charAt(0).toUpperCase()}
                            </div>
                            <span className="text-xs font-medium truncate group-hover:text-primary transition-colors">
                              {dataset.name}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={`h-5 px-1.5 text-[10px] ${
                              isBuiltin
                                ? 'bg-blue-500/10 text-blue-600 border-blue-500/30'
                                : 'bg-purple-500/10 text-purple-600 border-purple-500/30'
                            }`}
                          >
                            {isBuiltin ? t('evaluation.builtin') : t('evaluation.custom')}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground max-w-xs truncate">
                          {dataset.description || t('knowledgebase.noDescription')}
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant="secondary" className="h-5 px-1.5 text-[10px] font-mono">
                            <FileStack className="w-2.5 h-2.5 mr-1" />
                            {dataset.dataset_count}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant="secondary" className="h-5 px-1.5 text-[10px] font-mono">
                            <FlaskConical className="w-2.5 h-2.5 mr-1" />
                            {dataset.experiments_count}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formatBeijingTime(dataset.created_at)}
                          </div>
                        </TableCell>
                        <TableCell
                          className="text-right pr-3"
                          data-stop-click
                          onClick={(e) => e.stopPropagation()}
                        >
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
                                onSelect={() => router.push(`/evaluation/${dataset.id}`)}
                              >
                                <Pencil />
                                {t('common.view')}
                              </DropdownMenuItem>
                              {!isBuiltin && (
                                <>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem
                                    onSelect={(e) => {
                                      e.preventDefault();
                                      setTimeout(() => setDeleteTarget(dataset), 0);
                                    }}
                                    className="text-destructive focus:text-destructive"
                                  >
                                    <Trash2 />
                                    {t('common.delete')}
                                  </DropdownMenuItem>
                                </>
                              )}
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </div>
      </div>

      {!loading && datasets.length > 0 && (
        <div className="flex-none border-t border-border bg-background/60 backdrop-blur-sm py-2">
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      )}

      {/* Delete confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
        title={t('evaluation.deleteConfirmTitle')}
        description={t('evaluation.deleteConfirmMessage')}
        target={deleteTarget ? { label: 'Dataset', value: deleteTarget.name } : undefined}
        onConfirm={() => {
          if (deleteTarget) deleteEval(deleteTarget.id);
        }}
      />
    </div>
  );
};

export default EvaluationPage;
