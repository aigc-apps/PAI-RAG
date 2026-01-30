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
  Loader2
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
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="p-6 justify-center">
            <div className="flex items-center space-x-2">
              <Loader2 className="h-8 w-8 animate-spin" />
              <h2 className="text-xl font-medium">{t('evaluation.loadingExperiment')}</h2>
            </div>
            <p className="text-gray-500 mt-2">{t('evaluation.loadingExperimentDesc')}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen px-6 py-4 space-y-6">
      {/* Breadcrumb */}
      <div className="flex-none mb-2 flex items-center gap-2">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button variant="link" className="px-0" onClick={() => router.push('/evaluation')}>
                  {t('evaluation.title')}
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button variant="link" className="px-0" onClick={() => router.push(`/evaluation/${datasetId}`)}>
                  {evalConfig?.name}
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>{t('evaluation.experimentColon')} {experiment.name}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>

      <SampleDetailDialog
          open={isSampleDialogOpen}
          onOpenChange={setIsSampleDialogOpen}
          sample={selectedSample}
          mode="view"
      />

      <div className="flex-1 overflow-y-auto px-2">
        {/* Experiment Overview Card */}
        <Card className="mb-4">
          <CardHeader>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
              <CardTitle className="text-xl">{t('evaluation.experimentColon')} {experiment.name}</CardTitle>
            </div>
          </CardHeader>

          {/* Charts Area */}
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 pb-2">
              {/* Score Distribution */}
              <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                  <BarChart2 className="h-3.5 w-3.5" /> {t('evaluation.scoreDistribution')}
                </h3>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getScoreDistributionData(allExpItemsForStats)} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f1f1" />
                      <XAxis dataKey="range" tick={{ fontSize: 14 }} height={25} />
                      <YAxis tick={{ fontSize: 14 }} width={40} />
                      <Tooltip contentStyle={{ fontSize: '14px', padding: '4px 8px' }} />
                      <Legend wrapperStyle={{ fontSize: '14px' }} height={25} />
                      <Bar dataKey="count" fill="#8884d8" name={t('evaluation.samplesCount')} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Status Distribution */}
              <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
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
                        outerRadius={60}
                        paddingAngle={1}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {getStatusDistributionData(allExpItemsForStats).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={STATUS_COLORS[entry.name as StatusKey] || "#8884d8"} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ fontSize: '14px', padding: '4px 8px' }} />
                      <Legend wrapperStyle={{ fontSize: '14px', paddingTop: '5px' }} height={25} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Execution Time */}
              {hasTimingData(allExpItemsForStats) && (
                <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                  <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                    <Clock className="h-3.5 w-3.5" /> {t('evaluation.executionTime')}
                  </h3>
                  <div className="flex-1 flex flex-col justify-center min-h-0">
                    <div className="grid grid-cols-1 gap-2 flex-1">
                      <div className="flex group bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/10 p-2 rounded border border-blue-200 dark:border-blue-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-blue-100 dark:bg-blue-900/40 rounded">
                            <BarChart2 className="h-3 w-3 text-blue-600 dark:text-blue-400" />
                          </div>
                          <span className="text-xs font-medium text-blue-700 dark:text-blue-300">{t('evaluation.average')}</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-medium text-blue-600 dark:text-blue-400 tabular-nums">
                            {getAverageTime(allExpItemsForStats)}
                          </span>
                          <span className="text-xs text-blue-500 dark:text-blue-500">s</span>
                        </div>
                      </div>
                      <div className="flex group bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/10 p-2 rounded border border-green-200 dark:border-green-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-green-100 dark:bg-green-900/40 rounded">
                            <TrendingDown className="h-3 w-3 text-green-600 dark:text-green-400" />
                          </div>
                          <span className="text-xs font-medium text-green-700 dark:text-green-300">{t('evaluation.shortest')}</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-medium text-green-600 dark:text-green-400 tabular-nums">
                            {getMinTime(allExpItemsForStats)}
                          </span>
                          <span className="text-xs text-green-500 dark:text-green-500">s</span>
                        </div>
                      </div>
                      <div className="flex group bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-900/10 p-2 rounded border border-red-200 dark:border-red-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-red-100 dark:bg-red-900/40 rounded">
                            <TrendingUp className="h-3 w-3 text-red-600 dark:text-red-400" />
                          </div>
                          <span className="text-xs font-medium text-red-700 dark:text-red-300">{t('evaluation.longest')}</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-medium text-red-600 dark:text-red-400 tabular-nums">
                            {getMaxTime(allExpItemsForStats)}
                          </span>
                          <span className="text-xs text-red-500 dark:text-red-500">s</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Basic Info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-2">
              <div className="flex flex-wrap items-center gap-6 text-sm md:col-span-2">
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">ID:</span>
                  <span className="font-mono text-xs">{experiment.id}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">{t('evaluation.status')}:</span>
                  <StatusBadge status={experiment.status} />
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">{t('evaluation.samplesCount')}:</span>
                  <span className="font-semibold">{experiment.samples_count}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">{t('evaluation.averageScore')}:</span>
                  <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200 text-xs py-0.5 px-2">
                    {getAverageScore(allExpItemsForStats)}
                  </Badge>
                </div>
              </div>
              <div className="md:col-span-1 flex justify-end">
                <Button
                  variant="ghost"
                  onClick={() => setIsDetailExpanded(!isDetailExpanded)}
                  className="flex items-center gap-2 text-blue-500"
                >
                  {isDetailExpanded ? (
                    <>
                      <ChevronUp className="h-4 w-4" /> {t('evaluation.collapseDetails')}
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-4 w-4" /> {t('evaluation.expandDetails')}
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>

          {/* Detailed Config (Collapsible) */}
          {isDetailExpanded && (
            <CardContent className="border-t pt-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 space-x-16">
                {/* App Settings */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-1.5 text-sm">
                    <Settings2 className="h-4 w-4" /> {t('evaluation.appSettings')}
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">{t('evaluation.baseModel')}</span>
                      <span className="font-medium">{runConfig?.model_id || "—"}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">联网搜索</span>
                      {runConfig?.enable_search ? (
                        <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                      ) : (
                        <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Agentic</span>
                      {runConfig?.enable_agent ? (
                        <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                      ) : (
                        <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500 text-xs">MCP</span>
                      <div className="mt-1">
                        {Array.isArray(runConfig?.mcp_ids) && runConfig?.mcp_ids.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {runConfig.mcp_ids.map((mcp, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs py-0.5 px-1.5">
                                {mcp}
                              </Badge>
                            ))}
                          </div>
                        ) : (
                          <span className="text-muted-foreground text-xs">未配置</span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500 text-xs">知识库</span>
                      <div className="mt-1">
                        {Array.isArray(runConfig?.kb_ids) && runConfig?.kb_ids.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {runConfig.kb_ids.map((kb, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs py-0.5 px-1.5">
                                {kb}
                              </Badge>
                            ))}
                          </div>
                        ) : (
                          <span className="text-muted-foreground text-xs">未配置</span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500 text-xs">安全护栏</span>
                      <div className="mt-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs">输入/输出</span>
                          <div className="flex items-center gap-2">
                            <Switch checked={runConfig?.enable_input_guardrail || false} disabled className="h-4 w-8" />
                            <Switch checked={runConfig?.enable_output_guardrail || false} disabled className="h-4 w-8" />
                          </div>
                        </div>
                        {runConfig?.guardrail_hint && (
                          <div className="text-xs bg-muted p-1.5 rounded mt-1 truncate" title={runConfig.guardrail_hint}>
                            {runConfig.guardrail_hint}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* 评估设置 + 时间 */}
                <div className="flex flex-col gap-4">
                  <div>
                    <h3 className="font-semibold mb-3 flex items-center gap-1.5 text-sm">
                      <BarChart2 className="h-4 w-4" /> 评估器设置
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">类型</span>
                        <Badge variant="outline" className="text-xs py-0.5 px-2">
                          {evaluatorConfig?.type === "ExactMatch" ? "精确匹配" : "LLM 评判"}
                        </Badge>
                      </div>
                      {evaluatorConfig?.type === "ExactMatch" && (
                        <>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-500">大小写</span>
                            {evaluatorConfig?.case_sensitive ? (
                              <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                            ) : (
                              <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                            )}
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-500">标点</span>
                            {evaluatorConfig?.ignore_punctuation ? (
                              <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                            ) : (
                              <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                            )}
                          </div>
                        </>
                      )}
                      {evaluatorConfig?.type === "LLMJudge" && (
                        <div className="flex items-center justify-between">
                          <span className="text-gray-500">模型</span>
                          <span className="font-medium text-xs">{evaluatorConfig?.model_id || "未指定"}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-3 flex items-center gap-1.5 text-sm">
                      <Calendar className="h-4 w-4" /> 时间信息
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">创建</span>
                        <span className="text-xs">{formatBeijingTime(experiment.created_at)}</span>
                      </div>
                      {['success', 'failed'].includes(experiment.status) && (
                        <div className="flex items-center justify-between">
                          <span className="text-gray-500">完成</span>
                          <span className="text-xs">{formatBeijingTime(experiment.updated_at)}</span>
                        </div>
                      )}
                      {experiment.description && (
                        <div className="flex items-center justify-between">
                          <span className="text-gray-500 text-xs">描述</span>
                          <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                            {experiment.description}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          )}
        </Card>

        {/* 执行详情表格 */}
        <Card>
          <CardHeader>
            <CardTitle>执行详情 ({totalItems})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[5%]"></TableHead>
                    <TableHead className="w-[15%]">样本ID</TableHead>
                    <TableHead className="w-[40%]">问题</TableHead>
                    <TableHead className="w-[20%]">
                      <div className="flex items-center gap-1">
                        <span>状态</span>
                        <div className="relative">
                          <Select
                            value={statusFilter || ""}
                            onValueChange={(value) => {
                              setStatusFilter(value === "" ? null : (value as StatusType));
                              setPage(1);
                            }}
                          >
                            <SelectTrigger className="h-6 text-xs w-[90px]">
                              <SelectValue placeholder="筛选" />
                            </SelectTrigger>
                            <SelectContent>
                              {STATUS_OPTIONS.map((option) => (
                                <SelectItem key={option.value} value={option.value}>
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          {statusFilter && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-4 w-4 absolute -right-6 top-1/2 -translate-y-1/2 hover:bg-transparent"
                              onClick={(e) => {
                                e.stopPropagation();
                                setStatusFilter(null);
                                setPage(1);
                              }}
                              aria-label="清除筛选"
                            >
                              <XCircle className="h-3 w-3 text-red-500 hover:text-red-700" />
                            </Button>
                          )}
                        </div>
                      </div>
                    </TableHead>
                    <TableHead className="w-[10%]">得分</TableHead>
                    <TableHead>耗时</TableHead>
                    <TableHead className="w-[10%]">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {expItems.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} className="h-24 text-center">
                        暂无数据
                      </TableCell>
                    </TableRow>
                  ) : (
                    expItems.map((sample) => (
                      <Fragment key={sample.id}>
                        <TableRow className={isRowExpanded(sample.id) ? "bg-muted/50" : ""}>
                          <TableCell>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => toggleRow(sample.id)}
                              aria-expanded={isRowExpanded(sample.id)}
                              aria-label={isRowExpanded(sample.id) ? "收起详情" : "展开详情"}
                            >
                              {isRowExpanded(sample.id) ? (
                                <ChevronDown className="h-4 w-4" />
                              ) : (
                                <ChevronRight className="h-4 w-4" />
                              )}
                            </Button>
                          </TableCell>
                          <TableCell className="font-medium">
                            <Button
                              variant="link"
                              onClick={() => handleViewSample(sample.sample_id)}
                            >
                               {sample.sample_id.substring(0,10)}...
                            </Button>
                            </TableCell>
                          <TableCell className="whitespace-normal break-words min-w-[250px] max-w-[400px] py-2">
                            {sample.input.substring(0, 200)}...
                          </TableCell>
                          <TableCell>
                            <StatusBadge status={sample.status} />
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center">
                              <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                <div
                                  className={`h-2 rounded-full ${sample.status === 'success' ? 'bg-green-500' : (sample.status === 'failed') ? 'bg-red-500' : 'bg-blue-500'}`}
                                  style={{ width: `${Math.min(100, sample.score * 100)}%` }}
                                ></div>
                              </div>
                              <span>{sample.score}</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            {sample.started_at && sample.updated_at
                              ? calculateTimeDifference(sample.started_at, sample.updated_at)
                              : "-"}
                          </TableCell>
                          <TableCell>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => toggleRow(sample.id)}
                            >
                              {isRowExpanded(sample.id) ? "收起" : "详情"}
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => updateEvaluation(sample.id)}
                            >
                              重新评估
                            </Button>
                          </TableCell>
                        </TableRow>
                        {expandedRows.includes(sample.id) && (
                          <TableRow className="bg-muted/20">
                            <TableCell colSpan={8} className="p-0 border-0 w-full">
                              <div className="p-4 bg-background border rounded-lg">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                                  {/* 问答对评估 */}
                                  <div className="space-y-4">
                                    <h4 className="font-medium flex items-center gap-2 text-lg">
                                      <MessageSquare className="h-5 w-5 text-primary" /> 问答对评估
                                    </h4>
                                    <div className="space-y-4">
                                    {sample.trace_id && (
                                        <div className="space-y-2">
                                          <div className="flex items-center gap-2">
                                          <span className="font-medium text-gray-800 dark:text-gray-300">TraceId:</span>
                                            <span className="text-muted-foreground text-sm">{sample.trace_id}</span>
                                          </div>
                                        </div>
                                      )}
                                      <div className="space-y-2">
                                        <div className="flex items-center gap-2">
                                          <div className="bg-blue-100 dark:bg-blue-900/30 px-3 py-1 rounded-lg text-blue-800 dark:text-blue-300 font-medium flex items-center">
                                            <MessageSquare className="h-4 w-4 mr-2" /> 问题
                                          </div>
                                          <span className="text-muted-foreground text-sm">用户输入的问题</span>
                                        </div>
                                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border min-h-[80px]">
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.input}</p>
                                        </div>
                                      </div>
                                      <div className="space-y-2">
                                        <div className="flex items-center gap-2">
                                          <div className="bg-green-100 dark:bg-green-900/30 px-3 py-1 rounded-lg text-green-800 dark:text-green-300 font-medium flex items-center">
                                            <CheckCircle className="h-4 w-4 mr-2" /> 参考答案
                                          </div>
                                          <span className="text-muted-foreground text-sm">预期的标准答案</span>
                                        </div>
                                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border min-h-[120px] overflow-y-auto">
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.expected_output}</p>
                                        </div>
                                      </div>
                                      <div className="space-y-2">
                                        <div className="flex items-center gap-2">
                                          <div className="bg-purple-100 dark:bg-purple-900/30 px-3 py-1 rounded-lg text-purple-800 dark:text-purple-300 font-medium flex items-center">
                                            <Bot className="h-4 w-4 mr-2" /> 模型响应
                                          </div>
                                          <span className="text-muted-foreground text-sm">LLM生成的响应</span>
                                        </div>
                                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border h-[400px] overflow-y-auto ">
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.actual_output}</p>
                                        </div>
                                      </div>
                                    </div>
                                  </div>

                                  {/* 执行信息 */}
                                  <div className="space-y-6">
                                    <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <MessageSquare className="h-5 w-5 text-primary" />
                                        <h4 className="font-medium text-lg">得分原因</h4>
                                      </div>
                                      <div className="p-4 bg-muted rounded-lg">
                                        <p className="whitespace-pre-wrap leading-relaxed">{sample.reason}</p>
                                      </div>
                                    </div>
                                    <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <Clock className="h-5 w-5 text-muted-foreground" />
                                        <h4 className="font-medium text-lg">执行时间线</h4>
                                      </div>
                                      <div className="text-sm space-y-3 pl-6 border-l-2 border-border">
                                        <div className="flex items-start">
                                          <div className="w-3 h-3 rounded-full bg-primary mt-1 mr-3"></div>
                                          <div>
                                            <span className="text-muted-foreground font-medium">创建: </span>
                                            <span className="ml-2">{formatBeijingTime(sample.created_at) || "N/A"}</span>
                                          </div>
                                        </div>
                                        <div className="flex items-start">
                                          <div className="w-3 h-3 rounded-full bg-primary mt-1 mr-3"></div>
                                          <div>
                                            <span className="text-muted-foreground font-medium">开始: </span>
                                            <span className="ml-2">{sample.started_at ? formatBeijingTime(sample.started_at) : "N/A"} </span>
                                          </div>
                                        </div>
                                        <div className="flex items-start">
                                          {['success', 'failed'].includes(sample.status) ? (
                                            <div className="w-3 h-3 rounded-full bg-primary mt-1 mr-3"></div>
                                          ) : (
                                            <div className="w-3 h-3 rounded-full bg-success mt-1 mr-3"></div>
                                          )}
                                          <div>
                                            <span className="text-muted-foreground font-medium">完成: </span>
                                            <span className="ml-2">{['success', 'failed'].includes(sample.status) ? formatBeijingTime(sample.updated_at) : "进行中..."}</span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                    <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <Terminal className="h-5 w-5 text-muted-foreground" />
                                        <h4 className="font-medium text-lg">执行日志</h4>
                                      </div>
                                      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border h-[500px] overflow-y-auto font-mono text-sm">
                                        {sample.execution_metadata && sample.execution_metadata.length > 0 ? (
                                          <div className="space-y-4">
                                            {sample.execution_metadata.map((item, index) => {
                                              try {
                                                const args = JSON.parse(item.function.arguments);
                                                return (
                                                  <div key={index} className="border-l-2 border-blue-500 pl-3 py-1">
                                                    <div className="flex items-start gap-2 mb-2">
                                                      <Badge className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2 py-0.5 rounded text-xs font-medium">
                                                        {index}
                                                      </Badge>
                                                      <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2 py-0.5 rounded text-xs font-medium">
                                                        {item.function.name}
                                                      </span>
                                                    </div>
                                                    <div className="ml-2 mb-2">
                                                      <span className="text-xs font-medium text-gray-600 dark:text-gray-400">参数:</span>
                                                      <div className="ml-2 mt-1 text-xs bg-white dark:bg-gray-700 rounded p-1 border border-border">
                                                        {Object.entries(args).map(([key, value]) => (
                                                          <div key={key} className="flex">
                                                            <span className="text-blue-600 dark:text-blue-400">{key}:</span>
                                                            <span className="ml-1 truncate max-w-[200px]">{String(value)}</span>
                                                          </div>
                                                        ))}
                                                      </div>
                                                    </div>
                                                    <div className="ml-2 mb-2">
                                                      <span className="text-xs font-medium text-gray-600 dark:text-gray-400">观察结果:</span>
                                                      <div className="ml-2 mt-1 text-xs bg-white dark:bg-gray-700 rounded p-1 border border-border whitespace-pre-wrap h-[100px] overflow-y-auto">
                                                        <p>{item.observation?.result}</p>
                                                      </div>
                                                    </div>
                                                  </div>
                                                );
                                              } catch (e) {
                                                console.error("解析执行元数据错误:", e);
                                                return (
                                                  <div key={index} className="border-l-2 border-red-500 pl-3 py-1 text-red-600 dark:text-red-400 text-xs">
                                                    <div className="font-medium">解析错误</div>
                                                    <div>无法解析: {item.id}</div>
                                                  </div>
                                                );
                                              }
                                            })}
                                          </div>
                                        ) : (
                                          <div className="text-gray-500 dark:text-gray-400 flex items-center justify-center h-full">
                                            暂无日志信息
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
          <CardFooter className="flex justify-center">
            <div className="py-6">
              <PaginationComponent
                currentPage={page}
                totalPages={totalPages}
                onPageChange={handlePageChange}
              />
            </div>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}