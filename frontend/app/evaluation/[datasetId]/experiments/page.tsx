'use client';

import React from 'react';
import { useState, useEffect, use } from 'react';
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
  MoreHorizontal,
  Copy,
  Eye,
  BookOpen,
  BarChart2,
  Trash2,
  Loader2,
  Clock,
  CheckCircle2,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { formatBeijingTime } from '@/app/knowledgebases/utils/utils';
import { toast } from 'sonner';
import { StatusBadge } from '@/app/evaluation/components/status-badge';
import { useExperiments } from '@/app/evaluation/[datasetId]/experiments/useExperiments';

export default function EvalExperimentsDetailsPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { datasetId } = use(params);
  const router = useRouter();
  const { t } = useI18n();

  const [page, setPage] = useState(1);
  const pageSize = 10;

  const {
    experiments,
    totalPages,
    isLoading,
    deleteExperiment,
  } = useExperiments({ datasetId, page, pageSize });

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const formatScore = (score: number, status: string) => {
    if (status === 'running' || status === 'pending') {
      return <span className="text-muted-foreground text-xs">{t('evaluation.inProgress')}</span>;
    }
    if (status === 'failed') {
      return <span className="text-muted-foreground text-xs">- -</span>;
    }
    return score.toFixed(2);
  };

  const copyExperimentId = (id: string) => {
    navigator.clipboard.writeText(id);
    toast.success(t('evaluation.copySuccess'));
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-3">
        {isLoading ? (
          <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t('evaluation.loadingExperiments')}
          </div>
        ) : experiments.length === 0 ? (
          <div className="empty-state mt-8">
            <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-4">
              <BarChart2 className="w-7 h-7" />
            </div>
            <p className="text-base font-semibold mb-1">{t('evaluation.noExperiments')}</p>
            <p className="text-sm text-muted-foreground text-center max-w-md">
              {t('evaluation.newExperimentHint')
                .replace('"样本"', `"${t('evaluation.samples')}"`)
                .replace('"Samples"', `"${t('evaluation.samples')}"`)}
            </p>
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden bg-background">
            <Table className="table-modern">
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[160px] pl-4 text-xs text-muted-foreground">
                    {t('evaluation.experimentId')}
                  </TableHead>
                  <TableHead className="w-[160px] text-xs text-muted-foreground">
                    {t('evaluation.experimentName')}
                  </TableHead>
                  <TableHead className="w-[80px] text-xs text-muted-foreground text-right">
                    {t('evaluation.samplesCount')}
                  </TableHead>
                  <TableHead className="w-[100px] text-xs text-muted-foreground">
                    {t('evaluation.status')}
                  </TableHead>
                  <TableHead className="w-[100px] text-xs text-muted-foreground text-right">
                    {t('evaluation.averageScore')}
                  </TableHead>
                  <TableHead className="w-[150px] text-xs text-muted-foreground">
                    {t('evaluation.createdTime')}
                  </TableHead>
                  <TableHead className="w-[150px] text-xs text-muted-foreground">
                    {t('evaluation.completedTime')}
                  </TableHead>
                  <TableHead className="w-[60px] text-right pr-3 text-xs text-muted-foreground">
                    {t('evaluation.operations')}
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {experiments.map((item) => (
                  <TableRow key={item.id} className="group h-10 cursor-pointer" onClick={(e) => {
                    const target = e.target as HTMLElement;
                    if (target.closest('[data-stop-click]')) return;
                    router.push(`/evaluation/${datasetId}/experiments/${item.id}`);
                  }}>
                    <TableCell className="pl-4">
                      <div className="flex items-center gap-1" data-stop-click>
                        <button
                          type="button"
                          className="text-xs font-mono text-primary hover:underline truncate max-w-[110px]"
                          onClick={() =>
                            router.push(`/evaluation/${datasetId}/experiments/${item.id}`)
                          }
                        >
                          {item.id.substring(0, 8)}
                        </button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => copyExperimentId(item.id)}
                          title={t('evaluation.copyExperimentId')}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    </TableCell>

                    <TableCell>
                      <Badge
                        variant="outline"
                        className="font-mono text-[10px] h-5 px-1.5 bg-blue-500/10 text-blue-600 border-blue-500/30"
                      >
                        {item.name}
                      </Badge>
                    </TableCell>

                    <TableCell className="text-right">
                      <Badge variant="secondary" className="h-5 px-1.5 text-[10px] font-mono">
                        {item.samples_count}
                      </Badge>
                    </TableCell>

                    <TableCell>
                      <StatusBadge status={item.status} />
                    </TableCell>

                    <TableCell className="text-right">
                      <span
                        className={`text-xs font-semibold tabular-nums ${
                          item.status === 'success'
                            ? item.avg_score >= 0.8
                              ? 'text-green-600'
                              : item.avg_score >= 0.6
                                ? 'text-amber-600'
                                : 'text-rose-600'
                            : 'text-muted-foreground'
                        }`}
                      >
                        {formatScore(item.avg_score, item.status)}
                      </span>
                    </TableCell>

                    <TableCell className="text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatBeijingTime(item.created_at)}
                      </div>
                    </TableCell>

                    <TableCell className="text-xs text-muted-foreground">
                      {['success', 'failed'].includes(item.status) ? (
                        <div className="flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" />
                          {formatBeijingTime(item.updated_at)}
                        </div>
                      ) : (
                        '-'
                      )}
                    </TableCell>

                    <TableCell className="text-right pr-3" data-stop-click>
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
                            onClick={() =>
                              router.push(`/evaluation/${datasetId}/experiments/${item.id}`)
                            }
                          >
                            <Eye />
                            {t('evaluation.viewDetails')}
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            onClick={() => deleteExperiment(item.id)}
                            className="text-destructive focus:text-destructive"
                          >
                            <Trash2 />
                            {t('evaluation.delete')}
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

      {!isLoading && experiments.length > 0 && (
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
