'use client';

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
  { value: "running", label: "运行中" },
  { value: "success", label: "成功" },
  { value: "failed", label: "失败" },
  { value: "pending", label: "等待中" },
] as const;

type StatusType = (typeof STATUS_OPTIONS)[number]["value"];

export default function ExperimentDetailPage({ params }: { params: Promise<{ datasetId: string, expId: string }> }) {
  const { datasetId, expId } = use(params);
  const router = useRouter();

  // ========================
  // 状态管理
  // ========================

  const [evalConfig, setEvalConfig] = useState<EvalConfig>();
  const [experiment, setExperiment] = useState<ExperimentItem>();
  const [runConfig, setRunConfig] = useState<RunConfig>();
  const [evaluatorConfig, setEvaluatorConfig] = useState<EvaluatorConfig>();

  // 数据状态
  const [expItems, setExpItems] = useState<ExperimentSampleDetails[]>([]);
  const [expandedRows, setExpandedRows] = useState<string[]>([]);
  const [isDetailExpanded, setIsDetailExpanded] = useState(false);

  // 筛选与分页
  const [statusFilter, setStatusFilter] = useState<StatusType | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);

  // 查看样本对话框
  const [isSampleDialogOpen, setIsSampleDialogOpen] = useState(false);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
  const { tenantFetch } = useTenantFetch();
  const handleViewSample = async (sample_id: string) => {
    try {
        const response = await tenantFetch(`/api/config/evaluation/${datasetId}/samples/${sample_id}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) throw new Error("获取数据失败");

        const data = await response.json();

        setSelectedSample(data.data);
        setIsSampleDialogOpen(true);
    } catch (error) {
        toast.error('获取数据失败');
    }
  };

  // 当筛选条件变化时，重置到第一页
  useEffect(() => {
    setPage(1);
  }, [statusFilter]);

  // ========================
  // 数据获取与轮询
  // ========================

  // 获取实验样本数据（支持筛选和分页）
  const fetchExperimentSamples = useCallback(async () => {
    try {
      // 构建查询参数
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
      
      if (!response.ok) throw new Error('获取评估实验样本失败');
      const data = await response.json();
      
      setExpItems(data.data.items);
      setTotalPages(data.data.pages);
      setTotalItems(data.data.total);
    } catch (err: any) {
      console.error("fetchExperimentSamples 错误:", err);
      toast.error("加载实验样本失败");
    }
  }, [datasetId, expId, page, pageSize, statusFilter, tenantFetch]);

  // 获取实验元数据（配置信息等）
  const fetchExperimentDetails = useCallback(async () => {
    try {
      const [evalRes, expDataRes] = await Promise.all([
        tenantFetch(`/api/config/evaluation/${datasetId}`),
        tenantFetch(`/api/config/evaluation/${datasetId}/experiments/${expId}`),
      ]);

      // 获取评估配置
      if (!evalRes.ok) throw new Error('获取评估配置失败');
      const eval_data = await evalRes.json();
      setEvalConfig(eval_data.data);

      // 获取实验信息
      if (!expDataRes.ok) throw new Error('获取实验信息失败');
      const exp_data = await expDataRes.json();
      setExperiment(exp_data.data);

      // 获取运行配置
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
      console.error("fetchExperimentDetails 错误:", err);
      toast.error(err.message || "加载实验详情失败");
    }
  }, [datasetId, expId, tenantFetch]);

  // ========================
  // 重新评估单条样本
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
        toast.success("重新评估已启动！");
      } else {
        const data = await response.json();
        toast.error("重新评估启动失败: " + data.message);
      }
    } catch (error: any) {
      console.error("Re-evaluate failed:", error);
      toast.error("重新评估失败: " + error.message);
    }
  };

  // ========================
  // 轮询优化（使用 useEffect + clearTimeout）
  // ========================

  useEffect(() => {
    let pollTimeout: NodeJS.Timeout | null = null;

    const startPolling = () => {
      // 检查是否有未完成项
      const hasUnfinished = expItems.some(
        item => item.status !== 'success' && item.status !== 'failed'
      );

      if (hasUnfinished && experiment?.status !== 'success' && experiment?.status !== 'failed') {
        console.log('🔄 存在未完成实验，3秒后重新拉取...');
        pollTimeout = setTimeout(() => {
          fetchExperimentSamples();
          fetchExperimentDetails();
        }, 3000);
      } else {
        console.log('✅ 所有实验已完成，停止轮询。');
      }
    };

    // 组件挂载或数据更新时启动轮询
    startPolling();

    // 清理函数：组件卸载或依赖变化时清除定时器
    return () => {
      if (pollTimeout) {
        clearTimeout(pollTimeout);
        console.log('🧹 轮询定时器已清理');
      }
    };
  }, [expItems, experiment?.status, fetchExperimentSamples, fetchExperimentDetails]);

  // 首次加载元数据
  useEffect(() => {
    fetchExperimentDetails();
  }, [fetchExperimentDetails]);

  // 加载样本数据（page/filter 变化时重新加载）
  useEffect(() => {
    fetchExperimentSamples();
  }, [fetchExperimentSamples]);

  // ========================
  // 交互函数
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
  // UI渲染
  // ========================

  if (!experiment) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="p-6 justify-center">
            <div className="flex items-center space-x-2">
              <Loader2 className="h-8 w-8 animate-spin" />
              <h2 className="text-xl font-medium">Loading Experiment</h2>
            </div>
            <p className="text-gray-500 mt-2">Please wait while we load the experiment data.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen px-6 py-4 space-y-6">
      {/* 面包屑 */}
      <div className="flex-none mb-2 flex items-center gap-2">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button variant="link" className="px-0" onClick={() => router.push('/evaluation')}>
                  评估
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
              <BreadcrumbPage>实验：{experiment.name}</BreadcrumbPage>
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
        {/* 实验概览卡片 */}
        <Card className="mb-4">
          <CardHeader>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
              <CardTitle className="text-xl">实验: {experiment.name}</CardTitle>
            </div>
          </CardHeader>

          {/* 图表区域 */}
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 pb-2">
              {/* 得分分布 */}
              <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                  <BarChart2 className="h-3.5 w-3.5" /> 得分分布
                </h3>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getScoreDistributionData(expItems)} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f1f1" />
                      <XAxis dataKey="range" tick={{ fontSize: 14 }} height={25} />
                      <YAxis tick={{ fontSize: 14 }} width={40} />
                      <Tooltip contentStyle={{ fontSize: '14px', padding: '4px 8px' }} />
                      <Legend wrapperStyle={{ fontSize: '14px' }} height={25} />
                      <Bar dataKey="count" fill="#8884d8" name="样本数" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 状态分布 */}
              <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                  <PieChartIcon className="h-3.5 w-3.5" /> 状态分布
                </h3>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={getStatusDistributionData(expItems)}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, value }) => `${name} ${value}`}
                        outerRadius={60}
                        paddingAngle={1}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {getStatusDistributionData(expItems).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={STATUS_COLORS[entry.name as StatusKey] || "#8884d8"} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ fontSize: '14px', padding: '4px 8px' }} />
                      <Legend wrapperStyle={{ fontSize: '14px', paddingTop: '5px' }} height={25} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 执行耗时 */}
              {hasTimingData(expItems) && (
                <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                  <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                    <Clock className="h-3.5 w-3.5" /> 执行耗时
                  </h3>
                  <div className="flex-1 flex flex-col justify-center min-h-0">
                    <div className="grid grid-cols-1 gap-2 flex-1">
                      <div className="flex group bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/10 p-2 rounded border border-blue-200 dark:border-blue-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-blue-100 dark:bg-blue-900/40 rounded">
                            <BarChart2 className="h-3 w-3 text-blue-600 dark:text-blue-400" />
                          </div>
                          <span className="text-xs font-medium text-blue-700 dark:text-blue-300">平均</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-medium text-blue-600 dark:text-blue-400 tabular-nums">
                            {getAverageTime(expItems)}
                          </span>
                          <span className="text-xs text-blue-500 dark:text-blue-500">s</span>
                        </div>
                      </div>
                      <div className="flex group bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/10 p-2 rounded border border-green-200 dark:border-green-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-green-100 dark:bg-green-900/40 rounded">
                            <TrendingDown className="h-3 w-3 text-green-600 dark:text-green-400" />
                          </div>
                          <span className="text-xs font-medium text-green-700 dark:text-green-300">最短</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-medium text-green-600 dark:text-green-400 tabular-nums">
                            {getMinTime(expItems)}
                          </span>
                          <span className="text-xs text-green-500 dark:text-green-500">s</span>
                        </div>
                      </div>
                      <div className="flex group bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-900/10 p-2 rounded border border-red-200 dark:border-red-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-red-100 dark:bg-red-900/40 rounded">
                            <TrendingUp className="h-3 w-3 text-red-600 dark:text-red-400" />
                          </div>
                          <span className="text-xs font-medium text-red-700 dark:text-red-300">最长</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-medium text-red-600 dark:text-red-400 tabular-nums">
                            {getMaxTime(expItems)}
                          </span>
                          <span className="text-xs text-red-500 dark:text-red-500">s</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* 基础信息 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-2">
              <div className="flex flex-wrap items-center gap-6 text-sm md:col-span-2">
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">ID:</span>
                  <span className="font-mono text-xs">{experiment.id}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">状态:</span>
                  <StatusBadge status={experiment.status} />
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">样本数:</span>
                  <span className="font-semibold">{experiment.samples_count}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">平均得分:</span>
                  <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200 text-xs py-0.5 px-2">
                    {getAverageScore(expItems)}
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
                      <ChevronUp className="h-4 w-4" /> 收起详情
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-4 w-4" /> 展开详情
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>

          {/* 详细配置（可折叠） */}
          {isDetailExpanded && (
            <CardContent className="border-t pt-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 space-x-16">
                {/* 应用设置 */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-1.5 text-sm">
                    <Settings2 className="h-4 w-4" /> 应用设置
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">基模型</span>
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