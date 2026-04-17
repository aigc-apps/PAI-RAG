'use client';

import { useI18n } from '@/app/providers/i18n';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardFooter
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Button } from '@/components/ui/button';
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge"; 
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  ChevronDown,
  ChevronRight,
  Clock,
  Terminal,
  MessageSquare,
  CheckCircle,
  Bot,
  CircleXIcon,
  ChevronUp,
  Settings2,
  BarChart2,
  Calendar,
  PieChart as PieChartIcon,
  TrendingDown,
  TrendingUp,
  XCircle,
} from "lucide-react";
import { Fragment } from "react";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import { EvalConfig } from "@/app/evaluation/[datasetId]/types";
import { ExperimentItem } from '@/app/evaluation/[datasetId]/types';
import { StatusBadge } from '@/app/evaluation/components/status-badge';

import { formatBeijingTime, calculateTimeDifference } from '@/app/knowledgebases/utils/utils';
import { toast } from 'sonner';
import { Switch } from '@/components/ui/switch';
import { RunConfig } from '@/app/evaluation/[datasetId]/types';
import { EvaluatorConfig } from '@/app/evaluation/[datasetId]/types';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
} from "recharts";
import { useState, useEffect, use, useRef, useCallback, useMemo } from "react";
import {
  getScoreDistributionData,
  getStatusDistributionData,
  STATUS_COLORS,
  hasTimingData,
  getAverageTime,
  getMinTime,
  getMaxTime,
  getAverageScore,
  type StatusKey
} from '@/app/evaluation/components/chart-utils';
import { ExperimentSampleDetails } from '@/app/evaluation/[datasetId]/types';
import { SampleDetailDialog } from '@/app/evaluation/components/sample-detail-dialog';
import { SampleItem } from '@/app/evaluation/[datasetId]/types';
import { useTenantFetch } from "@/hooks/use-tenant-fetch";
import { HeaderPortal } from '@/components/header-portal';
import { PageLoading } from '@/components/ui/loading';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { MoreHorizontal, Eye, RefreshCw, Check, X } from 'lucide-react';

const STATUS_OPTIONS = [
  { value: "running", label: "statusRunning" },
  { value: "success", label: "statusSuccess" },
  { value: "failed", label: "statusFailed" },
  { value: "pending", label: "statusPending" },
] as const;

type StatusType = (typeof STATUS_OPTIONS)[number]["value"];

export default function ExperimentDetailPage({ params }: { params: Promise<{ datasetId: string, expId: string }> }) {
  const { t } = useI18n();

  const { datasetId, expId } = use(params);
  const router = useRouter();

  // ========================
  // State Management
  // ========================

  const [evalConfig, setEvalConfig] = useState<EvalConfig>();
  const [experiment, setExperiment] = useState<ExperimentItem>();
  const [runConfig, setRunConfig] = useState<RunConfig>();
  const [evaluatorConfig, setEvaluatorConfig] = useState<EvaluatorConfig>();

  // Data state
  const [expItems, setExpItems] = useState<ExperimentSampleDetails[]>([]);
  const [allExpItemsForStats, setAllExpItemsForStats] = useState<ExperimentSampleDetails[]>([]); // For statistics
  const [expandedRows, setExpandedRows] = useState<string[]>([]);
  const [isDetailExpanded, setIsDetailExpanded] = useState(false);

  // Filter and pagination
  const [statusFilter, setStatusFilter] = useState<StatusType | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);

  // View sample dialog
  const [isSampleDialogOpen, setIsSampleDialogOpen] = useState(false);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
  const { tenantFetch } = useTenantFetch();
  const handleViewSample = async (sample_id: string) => {
    try {
        const response = await tenantFetch(`/api/config/evaluation/${datasetId}/samples/${sample_id}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) throw new Error(t('evaluation.fetchSampleFailed'));

        const data = await response.json();

        setSelectedSample(data.data);
        setIsSampleDialogOpen(true);
    } catch (error) {
        toast.error(t('evaluation.fetchSampleFailed'));
    }
  };

  // Reset to first page when filter changes
  useEffect(() => {
    setPage(1);
  }, [statusFilter]);

  // ========================
  // Data Fetching and Polling
  // ========================

  // Fetch all experiment sample data (for statistics, no pagination, no status filter)
  const fetchAllExperimentSamplesForStats = useCallback(async () => {
    try {
      // Use max size to get all data for statistics
      const params = new URLSearchParams({
        page: '1',
        size: '1000', // Backend limit is 1000, need pagination if exceeded
      });

      const response = await tenantFetch(
        `/api/config/evaluation/${datasetId}/experiments/${expId}/samples?${params.toString()}`
      );
      
      if (!response.ok) throw new Error(t('evaluation.fetchExperimentSamplesFailed'));
      const data = await response.json();
      
      // If total exceeds 1000, need to paginate to get all data
      if (data.data.total > 1000) {
        const allItems: ExperimentSampleDetails[] = [];
        const totalPages = Math.ceil(data.data.total / 1000);
        
        // Get first page data
        allItems.push(...data.data.items);
        
        // Get remaining pages
        for (let p = 2; p <= totalPages; p++) {
          const pageParams = new URLSearchParams({
            page: p.toString(),
            size: '1000',
          });
          const pageResponse = await tenantFetch(
            `/api/config/evaluation/${datasetId}/experiments/${expId}/samples?${pageParams.toString()}`
          );
          if (pageResponse.ok) {
            const pageData = await pageResponse.json();
            allItems.push(...pageData.data.items);
          }
        }
        
        setAllExpItemsForStats(allItems);
      } else {
        setAllExpItemsForStats(data.data.items);
      }
    } catch (err: any) {
      console.error("fetchAllExperimentSamplesForStats error:", err);
      // Statistics failure doesn't affect main flow, only log error
    }
  }, [datasetId, expId, tenantFetch]);

  // Fetch experiment samples (with filter and pagination)
  const fetchExperimentSamples = useCallback(async () => {
    try {
      // Build query parameters
      const params = new URLSearchParams({
        page: page.toString(),
        size: pageSize.toString(),
      });
      if (statusFilter) {
        params.append('status', statusFilter);
      }

      const response = await tenantFetch(
        `/api/config/evaluation/${datasetId}/experiments/${expId}/samples?${params.toString()}`
      );
      
      if (!response.ok) throw new Error(t('evaluation.fetchExperimentSamplesFailed'));
      const data = await response.json();
      
      setExpItems(data.data.items);
      setTotalPages(data.data.pages);
      setTotalItems(data.data.total);
    } catch (err: any) {
      console.error("fetchExperimentSamples error:", err);
      toast.error(t('evaluation.loadExperimentSamplesFailed'));
    }
  }, [datasetId, expId, page, pageSize, statusFilter, tenantFetch]);

  // Fetch experiment metadata (config info, etc.)
  const fetchExperimentDetails = useCallback(async () => {
    try {
      const [evalRes, expDataRes] = await Promise.all([
        tenantFetch(`/api/config/evaluation/${datasetId}`),
        tenantFetch(`/api/config/evaluation/${datasetId}/experiments/${expId}`),
      ]);

      // Fetch evaluation config
      if (!evalRes.ok) throw new Error(t('evaluation.fetchEvalConfigFailed'));
      const eval_data = await evalRes.json();
      setEvalConfig(eval_data.data);

      // Fetch experiment info
      if (!expDataRes.ok) throw new Error(t('evaluation.fetchExperimentInfoFailed'));
      const exp_data = await expDataRes.json();
      setExperiment(exp_data.data);

      // Fetch run config
      if (exp_data.data?.run_config_id) {
        const runConfigRes = await tenantFetch(`/api/config/evaluation/${datasetId}/runconfigs/${exp_data.data.run_config_id}`);
        if (runConfigRes.ok) {
          const runConfigData = await runConfigRes.json();
          setRunConfig(runConfigData.data);
        }
      }

      if (exp_data.data?.evaluator_config_id) {
        const evaluatorConfigRes = await tenantFetch(`/api/config/evaluation/${datasetId}/evalconfigs/${exp_data.data.evaluator_config_id}`);
        if (evaluatorConfigRes.ok) {
          const evaluatorConfigData = await evaluatorConfigRes.json();
          setEvaluatorConfig(evaluatorConfigData.data);
        }
      }
    } catch (err: any) {
      console.error("fetchExperimentDetails error:", err);
      toast.error(err.message || t('evaluation.loadExperimentDetailsFailed'));
    }
  }, [datasetId, expId, tenantFetch]);

  // ========================
  // Re-evaluate single sample
  // ========================
  const updateEvaluation = async (id: string) => {
    try {
      const response = await tenantFetch(
        `/api/config/evaluation/${datasetId}/experiments/${expId}/samples`,
        {
          method: "PUT",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ id: id }),
        }
      );

      if (response.ok) {
        toast.success(t('evaluation.reEvaluateStarted'));
      } else {
        const data = await response.json();
        toast.error(t('evaluation.reEvaluateStartFailed') + ": " + data.message);
      }
    } catch (error: any) {
      console.error("Re-evaluate failed:", error);
      toast.error(t('evaluation.reEvaluateFailed') + ": " + error.message);
    }
  };

  // ========================
  // Polling optimization (using useEffect + clearTimeout)
  // ========================

  useEffect(() => {
    let pollTimeout: NodeJS.Timeout | null = null;

    const startPolling = () => {
      // Check if there are unfinished items (using full data)
      const hasUnfinished = allExpItemsForStats.some(
        item => item.status !== 'success' && item.status !== 'failed'
      );

      if (hasUnfinished && experiment?.status !== 'success' && experiment?.status !== 'failed') {
        console.log('🔄 Unfinished experiments exist, refreshing in 3 seconds...');
        pollTimeout = setTimeout(() => {
          fetchExperimentSamples();
          fetchExperimentDetails();
          fetchAllExperimentSamplesForStats(); // Also update stats
        }, 3000);
      } else {
        console.log('✅ All experiments completed, polling stopped.');
      }
    };

    // Start polling on mount or data update
    startPolling();

    // Cleanup: clear timer on unmount or dependency change
    return () => {
      if (pollTimeout) {
        clearTimeout(pollTimeout);
        console.log('🧹 Polling timer cleared');
      }
    };
  }, [allExpItemsForStats, experiment?.status, fetchExperimentSamples, fetchExperimentDetails, fetchAllExperimentSamplesForStats]);

  // Load metadata on first mount
  useEffect(() => {
    fetchExperimentDetails();
  }, [fetchExperimentDetails]);

  // Load sample data (reload when page/filter changes)
  useEffect(() => {
    fetchExperimentSamples();
  }, [fetchExperimentSamples]);

  // Load all sample data for statistics (only when experiment ID changes)
  useEffect(() => {
    fetchAllExperimentSamplesForStats();
  }, [fetchAllExperimentSamplesForStats]);

  // ========================
  // Interaction Functions
  // ========================

  const toggleRow = (id: string) => {
    setExpandedRows(prev => prev.includes(id) ? [] : [id]);
  };

  const isRowExpanded = (id: string) => expandedRows.includes(id);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  // ========================
  // UI Rendering
  // ========================

  if (!experiment) {
    return <PageLoading className="h-full" label={t('evaluation.loadingExperiment')} />;
  }

  const statusLabel = (value: string | null) => {
    if (!value) return null;
    const opt = STATUS_OPTIONS.find(o => o.value === value);
    return opt ? t(`evaluation.${opt.label}`) : value;
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button variant="link" className="px-0 h-auto" onClick={() => router.push('/evaluation')}>
                  {t('evaluation.title')}
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button variant="link" className="px-0 h-auto" onClick={() => router.push(`/evaluation/${datasetId}`)}>
                  {evalConfig?.name}
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage className="font-semibold">{experiment.name}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <Badge variant="secondary" className="text-[10px] font-mono bg-muted text-muted-foreground">
          ID: {experiment.id}
        </Badge>
        <StatusBadge status={experiment.status} />
      </HeaderPortal>

      <SampleDetailDialog
        open={isSampleDialogOpen}
        onOpenChange={setIsSampleDialogOpen}
        sample={selectedSample}
        mode="view"
      />

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {/* Summary stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <div className="rounded-lg border border-border bg-card px-3 py-2">
            <div className="text-[11px] text-muted-foreground">{t('evaluation.samplesCount')}</div>
            <div className="text-lg font-semibold tabular-nums">{experiment.samples_count}</div>
          </div>
          <div className="rounded-lg border border-border bg-card px-3 py-2">
            <div className="text-[11px] text-muted-foreground">{t('evaluation.averageScore')}</div>
            <div className="text-lg font-semibold tabular-nums text-primary">{getAverageScore(allExpItemsForStats)}</div>
          </div>
          {hasTimingData(allExpItemsForStats) && (
            <>
              <div className="rounded-lg border border-border bg-card px-3 py-2">
                <div className="text-[11px] text-muted-foreground flex items-center gap-1">
                  <TrendingDown className="h-3 w-3" />
                  {t('evaluation.shortest')}
                </div>
                <div className="text-lg font-semibold tabular-nums text-green-600">
                  {getMinTime(allExpItemsForStats)}<span className="text-xs text-muted-foreground ml-0.5">s</span>
                </div>
              </div>
              <div className="rounded-lg border border-border bg-card px-3 py-2">
                <div className="text-[11px] text-muted-foreground flex items-center gap-1">
                  <TrendingUp className="h-3 w-3" />
                  {t('evaluation.longest')}
                </div>
                <div className="text-lg font-semibold tabular-nums text-rose-600">
                  {getMaxTime(allExpItemsForStats)}<span className="text-xs text-muted-foreground ml-0.5">s</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <div className="flex flex-col rounded-lg border border-border bg-card p-3 h-[200px]">
            <h3 className="text-xs font-semibold mb-2 flex items-center gap-1.5 text-muted-foreground">
              <BarChart2 className="h-3.5 w-3.5" /> {t('evaluation.scoreDistribution')}
            </h3>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={getScoreDistributionData(allExpItemsForStats)} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="range" tick={{ fontSize: 11 }} height={22} />
                  <YAxis tick={{ fontSize: 11 }} width={28} />
                  <Tooltip contentStyle={{ fontSize: '11px', padding: '4px 8px' }} />
                  <Bar dataKey="count" fill="oklch(0.588 0.200 264)" name={t('evaluation.samplesCount')} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="flex flex-col rounded-lg border border-border bg-card p-3 h-[200px]">
            <h3 className="text-xs font-semibold mb-2 flex items-center gap-1.5 text-muted-foreground">
              <PieChartIcon className="h-3.5 w-3.5" /> {t('evaluation.statusDistribution')}
            </h3>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={getStatusDistributionData(allExpItemsForStats)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name} ${value}`}
                    outerRadius={55}
                    paddingAngle={1}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {getStatusDistributionData(allExpItemsForStats).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={STATUS_COLORS[entry.name as StatusKey] || "#8884d8"} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ fontSize: '11px', padding: '4px 8px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} height={22} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Collapsible config details */}
        <div className="rounded-lg border border-border bg-card">
          <button
            type="button"
            onClick={() => setIsDetailExpanded(!isDetailExpanded)}
            className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-muted/40 transition-colors"
          >
            <span className="text-xs font-semibold flex items-center gap-1.5">
              <Settings2 className="h-3.5 w-3.5" />
              {isDetailExpanded ? t('evaluation.collapseDetails') : t('evaluation.expandDetails')}
            </span>
            {isDetailExpanded ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>
          {isDetailExpanded && (
            <div className="border-t border-border p-4 grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* App Settings */}
              <div>
                <h4 className="text-[11px] font-semibold mb-2 flex items-center gap-1.5 text-muted-foreground uppercase tracking-wider">
                  <Settings2 className="h-3 w-3" /> {t('evaluation.appSettings')}
                </h4>
                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">{t('evaluation.baseModel')}</span>
                    <span className="font-mono text-[11px] truncate max-w-[180px]">{runConfig?.model_id || "—"}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">{t('evaluation.webSearch')}</span>
                    {runConfig?.enable_search ? (
                      <Check className="text-green-600 h-3.5 w-3.5" />
                    ) : (
                      <X className="text-muted-foreground h-3.5 w-3.5" />
                    )}
                  </div>
                  <div className="flex items-start justify-between gap-3">
                    <span className="text-muted-foreground shrink-0">MCP</span>
                    <div className="flex flex-wrap gap-1 justify-end">
                      {Array.isArray(runConfig?.mcp_ids) && runConfig.mcp_ids.length > 0 ? (
                        runConfig.mcp_ids.map((mcp, idx) => (
                          <Badge key={idx} variant="secondary" className="h-5 px-1.5 text-[10px] font-normal max-w-[120px] truncate">
                            {mcp}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-muted-foreground text-[11px]">—</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-start justify-between gap-3">
                    <span className="text-muted-foreground shrink-0">{t('knowledgebase.title')}</span>
                    <div className="flex flex-wrap gap-1 justify-end">
                      {Array.isArray(runConfig?.kb_ids) && runConfig.kb_ids.length > 0 ? (
                        runConfig.kb_ids.map((kb, idx) => (
                          <Badge key={idx} variant="secondary" className="h-5 px-1.5 text-[10px] font-normal max-w-[120px] truncate">
                            {kb}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-muted-foreground text-[11px]">—</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">{t('apps.guardrail')}</span>
                    <div className="flex items-center gap-1">
                      {runConfig?.enable_input_guardrail ? (
                        <Badge className="h-4 px-1 text-[9px] bg-blue-500/10 text-blue-700 border-blue-500/30">IN</Badge>
                      ) : null}
                      {runConfig?.enable_output_guardrail ? (
                        <Badge className="h-4 px-1 text-[9px] bg-blue-500/10 text-blue-700 border-blue-500/30">OUT</Badge>
                      ) : null}
                      {!runConfig?.enable_input_guardrail && !runConfig?.enable_output_guardrail && (
                        <span className="text-muted-foreground text-[11px]">—</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Evaluator Settings */}
              <div>
                <h4 className="text-[11px] font-semibold mb-2 flex items-center gap-1.5 text-muted-foreground uppercase tracking-wider">
                  <BarChart2 className="h-3 w-3" /> {t('evaluation.evaluatorSettings')}
                </h4>
                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">{t('evaluation.evaluatorType')}</span>
                    <Badge
                      variant="outline"
                      className={`h-5 px-1.5 text-[10px] ${
                        evaluatorConfig?.type === 'ExactMatch'
                          ? 'bg-blue-500/10 text-blue-700 border-blue-500/30'
                          : 'bg-purple-500/10 text-purple-700 border-purple-500/30'
                      }`}
                    >
                      {evaluatorConfig?.type === 'ExactMatch' ? t('evaluation.exactMatch') : t('evaluation.llmJudge')}
                    </Badge>
                  </div>
                  {evaluatorConfig?.type === 'ExactMatch' && (
                    <>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-muted-foreground">{t('evaluation.caseSensitive')}</span>
                        {evaluatorConfig.case_sensitive ? (
                          <Check className="text-green-600 h-3.5 w-3.5" />
                        ) : (
                          <X className="text-muted-foreground h-3.5 w-3.5" />
                        )}
                      </div>
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-muted-foreground">{t('evaluation.ignorePunctuation')}</span>
                        {evaluatorConfig.ignore_punctuation ? (
                          <Check className="text-green-600 h-3.5 w-3.5" />
                        ) : (
                          <X className="text-muted-foreground h-3.5 w-3.5" />
                        )}
                      </div>
                    </>
                  )}
                  {evaluatorConfig?.type === 'LLMJudge' && (
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-muted-foreground">{t('evaluation.model')}</span>
                      <span className="font-mono text-[11px] truncate max-w-[180px]">{evaluatorConfig.model_id || t('evaluation.notSpecified')}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Time */}
              <div>
                <h4 className="text-[11px] font-semibold mb-2 flex items-center gap-1.5 text-muted-foreground uppercase tracking-wider">
                  <Calendar className="h-3 w-3" /> {t('evaluation.createdTime')}
                </h4>
                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">{t('evaluation.createdTime')}</span>
                    <span className="font-mono text-[11px]">{formatBeijingTime(experiment.created_at)}</span>
                  </div>
                  {['success', 'failed'].includes(experiment.status) && (
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-muted-foreground">{t('evaluation.completedTime')}</span>
                      <span className="font-mono text-[11px]">{formatBeijingTime(experiment.updated_at)}</span>
                    </div>
                  )}
                  {hasTimingData(allExpItemsForStats) && (
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-muted-foreground">{t('evaluation.average')}</span>
                      <span className="font-mono text-[11px] text-blue-600">
                        {getAverageTime(allExpItemsForStats)}s
                      </span>
                    </div>
                  )}
                  {experiment.description && (
                    <div className="pt-2 border-t border-border mt-2">
                      <p className="text-[11px] text-muted-foreground line-clamp-2">
                        {experiment.description}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 执行详情表格 */}
        <div className="rounded-lg border border-border bg-background overflow-hidden">
          <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-border">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold">{t('evaluation.executionDetails') || '执行详情'}</h2>
              <Badge variant="secondary" className="h-5 px-1.5 text-[10px] font-mono">{totalItems}</Badge>
            </div>
            <div className="flex items-center gap-2">
              <Select
                value={statusFilter || ''}
                onValueChange={(value) => {
                  setStatusFilter(value === '' ? null : (value as StatusType));
                  setPage(1);
                }}
              >
                <SelectTrigger className="h-7 text-xs w-[120px]">
                  <SelectValue placeholder={t('evaluation.filterByStatus') || '状态筛选'} />
                </SelectTrigger>
                <SelectContent>
                  {STATUS_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="text-xs">
                      {t(`evaluation.${option.label}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {statusFilter && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs text-muted-foreground hover:text-destructive"
                  onClick={() => {
                    setStatusFilter(null);
                    setPage(1);
                  }}
                >
                  <XCircle className="h-3 w-3 mr-1" />
                  {t('common.reset')}
                </Button>
              )}
            </div>
          </div>
          <Table className="table-modern">
            <TableHeader>
              <TableRow className="h-8">
                <TableHead className="w-[40px] pl-3 py-1"></TableHead>
                <TableHead className="w-[120px] text-xs text-muted-foreground px-2 py-1">{t('evaluation.sampleId')}</TableHead>
                <TableHead className="text-xs text-muted-foreground px-2 py-1">{t('evaluation.question')}</TableHead>
                <TableHead className="w-[100px] text-xs text-muted-foreground px-2 py-1">{t('evaluation.status')}</TableHead>
                <TableHead className="w-[140px] text-xs text-muted-foreground px-2 py-1">{t('evaluation.averageScore') || '得分'}</TableHead>
                <TableHead className="w-[80px] text-xs text-muted-foreground px-2 py-1 text-right">{t('evaluation.duration') || '耗时'}</TableHead>
                <TableHead className="w-[60px] text-xs text-muted-foreground px-2 py-1 text-right pr-3">{t('evaluation.operations')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {expItems.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="h-20 text-center text-xs text-muted-foreground">
                    {t('common.noData')}
                  </TableCell>
                </TableRow>
              ) : (
                expItems.map((sample) => {
                  const expanded = isRowExpanded(sample.id);
                  return (
                    <Fragment key={sample.id}>
                      <TableRow
                        className={`group h-9 cursor-pointer ${expanded ? 'bg-muted/30' : ''}`}
                        onClick={(e) => {
                          const target = e.target as HTMLElement;
                          if (target.closest('[data-stop-click]')) return;
                          toggleRow(sample.id);
                        }}
                      >
                        <TableCell className="pl-3 px-2 py-0.5">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleRow(sample.id);
                            }}
                            aria-expanded={expanded}
                          >
                            {expanded ? (
                              <ChevronDown className="h-3.5 w-3.5" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5" />
                            )}
                          </Button>
                        </TableCell>
                        <TableCell
                          className="px-2 py-0.5"
                          data-stop-click
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            className="text-xs font-mono text-primary hover:underline"
                            onClick={() => handleViewSample(sample.sample_id)}
                          >
                            {sample.sample_id.substring(0, 8)}
                          </button>
                        </TableCell>
                        <TableCell
                          className="text-xs px-2 py-0.5"
                          style={{
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            maxWidth: 0,
                          }}
                        >
                          {sample.input}
                        </TableCell>
                        <TableCell className="px-2 py-0.5">
                          <StatusBadge status={sample.status} />
                        </TableCell>
                        <TableCell className="px-2 py-0.5">
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  sample.status === 'success'
                                    ? 'bg-green-500'
                                    : sample.status === 'failed'
                                      ? 'bg-rose-500'
                                      : 'bg-blue-500'
                                }`}
                                style={{ width: `${Math.min(100, sample.score * 100)}%` }}
                              />
                            </div>
                            <span className="text-xs font-mono tabular-nums w-8 text-right">{sample.score}</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground px-2 py-0.5 text-right tabular-nums">
                          {sample.started_at && sample.updated_at
                            ? calculateTimeDifference(sample.started_at, sample.updated_at)
                            : '-'}
                        </TableCell>
                        <TableCell
                          className="text-right pr-3 px-2 py-0.5"
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
                              <DropdownMenuItem onSelect={() => handleViewSample(sample.sample_id)}>
                                <Eye />
                                {t('evaluation.viewDetails')}
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem onSelect={() => updateEvaluation(sample.id)}>
                                <RefreshCw />
                                {t('evaluation.reevaluate') || '重新评估'}
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                        {expanded && (
                          <TableRow className="bg-muted/10 hover:bg-muted/10">
                            <TableCell colSpan={7} className="p-0 border-0">
                              <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
                                {/* 左：问答对 */}
                                <div className="space-y-3">
                                  {sample.trace_id && (
                                    <div className="text-[11px] font-mono text-muted-foreground">
                                      TraceId: {sample.trace_id}
                                    </div>
                                  )}
                                  <div>
                                    <div className="flex items-center gap-1.5 mb-1">
                                      <MessageSquare className="h-3 w-3 text-blue-600" />
                                      <span className="text-[11px] font-semibold text-blue-700 uppercase tracking-wider">问题</span>
                                    </div>
                                    <div className="text-xs bg-muted/40 p-2 rounded border border-border whitespace-pre-wrap leading-relaxed max-h-[120px] overflow-y-auto">
                                      {sample.input}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="flex items-center gap-1.5 mb-1">
                                      <CheckCircle className="h-3 w-3 text-green-600" />
                                      <span className="text-[11px] font-semibold text-green-700 uppercase tracking-wider">参考答案</span>
                                    </div>
                                    <div className="text-xs bg-muted/40 p-2 rounded border border-border whitespace-pre-wrap leading-relaxed max-h-[160px] overflow-y-auto">
                                      {sample.expected_output}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="flex items-center gap-1.5 mb-1">
                                      <Bot className="h-3 w-3 text-purple-600" />
                                      <span className="text-[11px] font-semibold text-purple-700 uppercase tracking-wider">模型响应</span>
                                    </div>
                                    <div className="text-xs bg-muted/40 p-2 rounded border border-border whitespace-pre-wrap leading-relaxed max-h-[240px] overflow-y-auto">
                                      {sample.actual_output}
                                    </div>
                                  </div>
                                  {sample.reason && (
                                    <div>
                                      <div className="flex items-center gap-1.5 mb-1">
                                        <MessageSquare className="h-3 w-3 text-amber-600" />
                                        <span className="text-[11px] font-semibold text-amber-700 uppercase tracking-wider">得分原因</span>
                                      </div>
                                      <div className="text-xs bg-amber-500/5 p-2 rounded border border-amber-500/20 whitespace-pre-wrap leading-relaxed">
                                        {sample.reason}
                                      </div>
                                    </div>
                                  )}
                                </div>

                                {/* 右：时间线 + 日志 */}
                                <div className="space-y-3">
                                  <div>
                                    <div className="flex items-center gap-1.5 mb-1.5">
                                      <Clock className="h-3 w-3 text-muted-foreground" />
                                      <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">执行时间线</span>
                                    </div>
                                    <div className="text-xs space-y-1.5 pl-3 border-l-2 border-border ml-1">
                                      <div className="flex items-baseline gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground shrink-0 relative -left-[17px] -mr-2" />
                                        <span className="text-muted-foreground w-10">创建</span>
                                        <span className="font-mono text-[11px]">{formatBeijingTime(sample.created_at) || 'N/A'}</span>
                                      </div>
                                      <div className="flex items-baseline gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground shrink-0 relative -left-[17px] -mr-2" />
                                        <span className="text-muted-foreground w-10">开始</span>
                                        <span className="font-mono text-[11px]">
                                          {sample.started_at ? formatBeijingTime(sample.started_at) : 'N/A'}
                                        </span>
                                      </div>
                                      <div className="flex items-baseline gap-2">
                                        <div
                                          className={`w-1.5 h-1.5 rounded-full shrink-0 relative -left-[17px] -mr-2 ${
                                            ['success', 'failed'].includes(sample.status) ? 'bg-primary' : 'bg-blue-500 animate-pulse'
                                          }`}
                                        />
                                        <span className="text-muted-foreground w-10">完成</span>
                                        <span className="font-mono text-[11px]">
                                          {['success', 'failed'].includes(sample.status)
                                            ? formatBeijingTime(sample.updated_at)
                                            : '进行中...'}
                                        </span>
                                      </div>
                                    </div>
                                  </div>

                                  <div>
                                    <div className="flex items-center gap-1.5 mb-1.5">
                                      <Terminal className="h-3 w-3 text-muted-foreground" />
                                      <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">执行日志</span>
                                    </div>
                                    <div className="text-[11px] bg-slate-50 dark:bg-slate-900/40 p-2 rounded border border-border max-h-[320px] overflow-y-auto font-mono">
                                      {sample.execution_metadata && sample.execution_metadata.length > 0 ? (
                                        <div className="space-y-2">
                                          {sample.execution_metadata.map((item, index) => {
                                            try {
                                              const args = JSON.parse(item.function.arguments);
                                              return (
                                                <div key={index} className="border-l-2 border-blue-500 pl-2 py-1">
                                                  <div className="flex items-center gap-1 mb-1">
                                                    <Badge className="bg-blue-500/10 text-blue-700 border-blue-500/30 h-4 px-1 text-[9px] font-mono">
                                                      #{index}
                                                    </Badge>
                                                    <span className="text-[11px] font-semibold text-blue-700">
                                                      {item.function.name}
                                                    </span>
                                                  </div>
                                                  <div className="space-y-0.5 pl-1">
                                                    <div className="text-[10px] text-muted-foreground">参数:</div>
                                                    <div className="bg-background rounded px-1.5 py-0.5 border border-border">
                                                      {Object.entries(args).map(([key, value]) => (
                                                        <div key={key} className="flex text-[10px]">
                                                          <span className="text-blue-600 shrink-0">{key}:</span>
                                                          <span className="ml-1 truncate">{String(value)}</span>
                                                        </div>
                                                      ))}
                                                    </div>
                                                    <div className="text-[10px] text-muted-foreground pt-1">结果:</div>
                                                    <div className="bg-background rounded px-1.5 py-0.5 border border-border max-h-[80px] overflow-y-auto whitespace-pre-wrap text-[10px]">
                                                      {item.observation?.result}
                                                    </div>
                                                  </div>
                                                </div>
                                              );
                                            } catch {
                                              return (
                                                <div key={index} className="border-l-2 border-rose-500 pl-2 py-1 text-rose-600 text-[10px]">
                                                  <div className="font-medium">解析错误</div>
                                                  <div>无法解析: {item.id}</div>
                                                </div>
                                              );
                                            }
                                          })}
                                        </div>
                                      ) : (
                                        <div className="text-muted-foreground text-center py-4">
                                          暂无日志信息
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        {expItems.length > 0 && (
          <div className="flex justify-center py-2">
            <PaginationComponent
              currentPage={page}
              totalPages={totalPages}
              onPageChange={handlePageChange}
            />
          </div>
        )}
      </div>
    </div>
  );
}