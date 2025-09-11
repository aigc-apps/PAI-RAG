'use client';

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useState, useEffect, use, useRef, useCallback } from "react";
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
  TrendingUp
} from "lucide-react";
import { Fragment } from "react";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import { EvalConfig } from '@/app/evaluation/[evalId]/page';
import { ExperimentItem, getStatusBadge } from "@/app/evaluation/[evalId]/experiments/page";
import { formatBeijingTime, calculateTimeDifference } from '@/app/knowledgebases/utils/utils';
import { toast } from 'sonner';
import { Switch } from '@/components/ui/switch';
import { EvalRunConfig } from '@/app/evaluation/[evalId]/settings/page';
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

type ExperimentDetailsItem = {
  id: string
  input: string
  expected_output: string
  actual_output: string
  status: string
  score: number
  reason: string
  dataset_metadata?: {
    Steps?: string
    Tools?: string
  }
  execution_metadata?: {
    id: string
    index: number
    function: {
      name: string
      arguments: string
    }
    type: string
    observation: string | null
  }[]
  created_at: string
  started_at: string | null
  updated_at: string
}


export default function ExperimentDetailPage({ params }: { params: Promise<{ evalId: string, expId: string }> }) {
  const { evalId, expId } = use(params);
  const router = useRouter();
  const [evalConfig, setEvalConfig] = useState<EvalConfig>();
  const [experiment, setExperiment] = useState<ExperimentItem>();
  const [runDetailItems, setRunDetailItems] = useState<ExperimentDetailsItem[]>([]);
  const [allExpItems, setAllItems] = useState<ExperimentDetailsItem[]>([]);
  const [expandedRows, setExpandedRows] = useState<string[]>([]);
  const [evalRunConfig, setEvalRunConfig] = useState<EvalRunConfig>();

  const [page, setPage] = useState(1);
  const pageRef = useRef(page);
  const [totalPages, setTotalPages] = useState(1);
  const pageSize = 10;
  let isRefreshing = false;
  const [isDetailExpanded, setIsDetailExpanded] = useState(false);


  const fetchExperimentDetails = useCallback(async () => {
    if (isRefreshing) {
      console.log("list already refreshing.")
      return;
    }
    console.log("Refreshing...");

    try {
      isRefreshing = true;
      const [evalRes, expDataRes, detailsRes] = await Promise.all([
        fetch(`/api/config/evaluation/${evalId}`),
        fetch(`/api/config/evaluation/${evalId}/experiments/${expId}`),
        fetch(`/api/config/evaluation/${evalId}/experiments/${expId}/details?page=${pageRef.current}&size=${pageSize}`),
      ]);

      if (!evalRes.ok) throw new Error('获取实验失败');
      const eval_data = await evalRes.json();
      const evalData = eval_data.data;
      console.log('evalData:', evalData);
      setEvalConfig(evalData);

      if (!expDataRes.ok) throw new Error('获取实验失败');

      const exp_data = await expDataRes.json();
      const expData = exp_data.data;
      console.log('expData:', expData);
      setExperiment(expData);

      try {
        const [evalRes] = await Promise.all([
          fetch(`/api/config/evaluation/${evalId}/configs/${expData?.run_config_id}`),
        ]);
        if (!evalRes.ok) throw new Error('获取run_config失败');
        const eval_data = await evalRes.json();
        const evalData = eval_data.data;
        console.log('setEvalRunConfig:', evalData);
        setEvalRunConfig(evalData);
      } catch (err: any) {
        toast.error(err.message);
      }

      const exp_json_data = await detailsRes.json();
      const data = exp_json_data.data.items;
      console.log('detailsRes:', exp_json_data);
      setRunDetailItems(data || []);
      setTotalPages(exp_json_data.data.pages);

      const tmpAllItems = [];
      for (let curPage = 1; curPage <= exp_json_data.data.pages; curPage++) {
          console.log("start loading all items for page ", curPage)
          const tmpPageSize = 1000;
          const response = await fetch(`/api/config/evaluation/${evalId}/experiments/${expId}/details?page=${curPage}&size=${tmpPageSize}`);
          const data = await response.json();
          tmpAllItems.push(...data.data.items);
      }
      setAllItems(tmpAllItems);
      console.log("finish loading all items", tmpAllItems.length)


      const kb_files = data as ExperimentDetailsItem[];
      const files_unfinished = kb_files.some(
        (file) => file.status !== 'success' && file.status !== 'failed',
      );

      if (files_unfinished) {
        console.log('存在未完成的实验，继续检查状态。');
        setTimeout(() => {
          isRefreshing = false;
          fetchExperimentDetails(); // 依赖 ref 获取最新 page
        }, 3000);
      } else {
        console.log('实验已完成。');
      }
      isRefreshing = false;
    } catch (err: any) {
      isRefreshing = false;
      toast.error(err.message);
    }
  }, [expId]);

  useEffect(() => {
    pageRef.current = page;
  }, [page]);

  useEffect(() => {
    fetchExperimentDetails();
  }, [fetchExperimentDetails, page]);


  // 切换行的展开状态
  const toggleRow = (id: string) => {
    if (expandedRows.includes(id)) {
      setExpandedRows([]);
    } else {
      setExpandedRows([id]);
    }
  }

  // 检查行是否展开
  const isRowExpanded = (id: string) => expandedRows.includes(id);

  const getScoreDistributionData = () => {
    const ranges = [
      { range: "0-0.2", min: 0, max: 0.2, count: 0 },
      { range: "0.2-0.4", min: 0.2, max: 0.4, count: 0 },
      { range: "0.4-0.6", min: 0.4, max: 0.6, count: 0 },
      { range: "0.6-0.8", min: 0.6, max: 0.8, count: 0 },
      { range: "0.8-1.0", min: 0.8, max: 1.0, count: 0 },
    ];

    allExpItems.forEach((item) => {
      if (item.score !== undefined) {
        const range = ranges.find(r => item.score >= r.min && item.score < r.max);
        if (range) {
          range.count++;
        } else if (item.score === 1.0) {
          ranges[ranges.length - 1].count++;
        }
      }
    });

    return ranges.map(r => ({ range: r.range, count: r.count }));
  };

  type StatusKey = 'success' | 'failed' | 'running' | 'pending';

  const getStatusDistributionData = () => {
    const statusCount: Record<StatusKey, number> = {
      success: 0,
      failed: 0,
      running: 0,
      pending: 0,
    };

    allExpItems.forEach((item) => {
      if ((item.status as StatusKey) in statusCount) {
        statusCount[item.status as StatusKey]++;
      }
    });

    return Object.entries(statusCount)
      .filter(([_, count]) => count > 0)
      .map(([name, value]) => ({ name, value }));
  };

  const STATUS_COLORS: Record<StatusKey, string> = {
    success: "#10b981", // green-500
    failed: "#ef4444",  // red-500
    running: "#3b82f6", // blue-500
    pending: "#f59e0b", // amber-500
  };

  const hasTimingData = () => {
    return allExpItems.some(item => item.started_at && item.updated_at);
  };

  const getAverageTime = () => {
    const formatUTC = (str: string | null) =>
      str ? str.replace(' ', 'T').replace(/\.\d+$/, '') + 'Z' : '';
    const times = allExpItems
      .filter(item => item.started_at && item.updated_at)
      .map(item => {
        const start = new Date(formatUTC(item.started_at)).getTime();
        const end = new Date(formatUTC(item.updated_at)).getTime();
        return (end - start) / 1000;
      });

    if (times.length === 0) return "N/A";
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    return avg.toFixed(2);
  };

  const getMinTime = () => {
    const formatUTC = (str: string | null) =>
      str ? str.replace(' ', 'T').replace(/\.\d+$/, '') + 'Z' : '';
    const times = allExpItems
      .filter(item => item.started_at && item.updated_at)
      .map(item => {
        const start = new Date(formatUTC(item.started_at)).getTime();
        const end = new Date(formatUTC(item.updated_at)).getTime();
        return (end - start) / 1000;
      });

    if (times.length === 0) return "N/A";
    return Math.min(...times).toFixed(2);
  };

  const getMaxTime = () => {
    const formatUTC = (str: string | null) =>
      str ? str.replace(' ', 'T').replace(/\.\d+$/, '') + 'Z' : '';
    const times = allExpItems
      .filter(item => item.started_at && item.updated_at)
      .map(item => {
        const start = new Date(formatUTC(item.started_at)).getTime();
        const end = new Date(formatUTC(item.updated_at)).getTime();
        return (end - start) / 1000;
      });

    if (times.length === 0) return "N/A";
    return Math.max(...times).toFixed(2);
  };

  if (!experiment) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="p-6 text-center">
            <h2 className="text-2xl font-bold">Loading Experiment</h2>
            <p className="text-gray-500 mt-2">Please wait while we load the experiment data.</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  return (
    <div className="flex flex-col h-screen px-6 py-4 space-y-6">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="mb-2 flex items-center gap-2">
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() => router.push('/evaluation')}
                    >
                      评估
                    </Button>
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem>
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() => router.push(`/evaluation/${evalId}`)}
                    >
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
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-2">
        <Card className="mb-4">
          <CardHeader>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
              <CardTitle className="text-2xl">实验: {experiment.name}</CardTitle>
            </div>
          </CardHeader>

          {/* 详情摘要*/}
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 pb-2">
              {/* 得分分布柱状图 */}
              <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                  <BarChart2 className="h-3.5 w-3.5" /> 得分分布
                </h3>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getScoreDistributionData()} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f1f1" />
                      <XAxis
                        dataKey="range"
                        tick={{ fontSize: 14 }}
                        height={25}
                      />
                      <YAxis
                        tick={{ fontSize: 14 }}
                        width={40}
                      />
                      <Tooltip
                        contentStyle={{
                          fontSize: '14px',
                          padding: '4px 8px'
                        }}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: '14px' }}
                        height={25}
                      />
                      <Bar dataKey="count" fill="#8884d8" name="样本" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 状态分布饼图 */}
              <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                  <PieChartIcon className="h-3.5 w-3.5" /> 状态分布
                </h3>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                      <Pie
                        data={getStatusDistributionData()}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={60}
                        fill="#8884d8"
                        dataKey="value"
                        paddingAngle={1}
                      >
                        {getStatusDistributionData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={STATUS_COLORS[entry.name as StatusKey] || "#8884d8"} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          fontSize: '14px',
                          padding: '4px 8px'
                        }}
                      />
                      <Legend
                        wrapperStyle={{
                          fontSize: '14px',
                          paddingTop: '5px'
                        }}
                        height={25}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* 执行耗时分析 */}
              {hasTimingData() && (
                <div className="flex flex-col bg-card rounded-lg border p-3 hover:shadow-sm transition-shadow h-[220px]">
                  <h3 className="font-semibold mb-2 flex items-center gap-1.5 text-sm">
                    <Clock className="h-3.5 w-3.5" /> 执行耗时
                  </h3>
                  <div className="flex-1 flex flex-col justify-center min-h-0">
                    <div className="grid grid-cols-1 gap-2 flex-1">
                      {/* 平均耗时 */}
                      <div className="flex group bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/10 p-2 rounded border border-blue-200 dark:border-blue-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-blue-100 dark:bg-blue-900/40 rounded">
                            <BarChart2 className="h-3 w-3 text-blue-600 dark:text-blue-400" />
                          </div>
                          <span className="text-xs font-medium text-blue-700 dark:text-blue-300">平均</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-bold text-blue-600 dark:text-blue-400 tabular-nums">
                            {getAverageTime()}
                          </span>
                          <span className="text-xs text-blue-500 dark:text-blue-500">s</span>
                        </div>
                      </div>

                      {/* 最短耗时 */}
                      <div className="flex group bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/10 p-2 rounded border border-green-200 dark:border-green-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-green-100 dark:bg-green-900/40 rounded">
                            <TrendingDown className="h-3 w-3 text-green-600 dark:text-green-400" />
                          </div>
                          <span className="text-xs font-medium text-green-700 dark:text-green-300">最短</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-bold text-green-600 dark:text-green-400 tabular-nums">
                            {getMinTime()}
                          </span>
                          <span className="text-xs text-green-500 dark:text-green-500">s</span>
                        </div>
                      </div>

                      {/* 最长耗时 */}
                      <div className="flex group bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-900/10 p-2 rounded border border-red-200 dark:border-red-800 hover:shadow transition-all duration-200 gap-6">
                        <div className="flex items-center gap-1.5 mb-1">
                          <div className="p-0.5 bg-red-100 dark:bg-red-900/40 rounded">
                            <TrendingUp className="h-3 w-3 text-red-600 dark:text-red-400" />
                          </div>
                          <span className="text-xs font-medium text-red-700 dark:text-red-300">最长</span>
                        </div>
                        <div className="flex items-baseline gap-1">
                          <span className="text-xl font-bold text-red-600 dark:text-red-400 tabular-nums">
                            {getMaxTime()}
                          </span>
                          <span className="text-xs text-red-500 dark:text-red-500">s</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-2">
              <div className="flex flex-wrap items-center gap-6 text-sm md:col-span-2">
                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">ID:</span>
                  <span className="font-mono text-xs">{experiment.id}</span>
                </div>

                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">状态:</span>
                  {getStatusBadge(experiment.status)}
                </div>

                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">样本数:</span>
                  <span className="font-semibold">{experiment.samples_count}</span>
                </div>

                <div className="flex items-center gap-1">
                  <span className="text-gray-500 font-medium">平均得分:</span>
                  <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200 text-xs py-0.5 px-2">
                    {experiment.avg_score ? experiment.avg_score.toFixed(2) : "0.0"}
                  </Badge>
                </div>
              </div>

              {/* 展开/收起按钮 */}
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

          {/* 详细信息（可折叠） */}
          {isDetailExpanded && (
            <CardContent className="border-t pt-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 space-x-16">
                {/* 左侧：应用设置 */}
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-1.5 text-sm">
                    <Settings2 className="h-4 w-4" /> 应用设置
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">基模型</span>
                      <span className="font-medium">{evalRunConfig?.model_id || "—"}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">联网搜索</span>
                      {evalRunConfig?.enable_search ? (
                        <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                      ) : (
                        <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500">Agentic</span>
                      {evalRunConfig?.enable_agent ? (
                        <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                      ) : (
                        <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500 text-xs">MCP</span>
                      <div className="mt-1">
                        {Array.isArray(evalRunConfig?.mcp_ids) && evalRunConfig?.mcp_ids.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {evalRunConfig.mcp_ids.map((mcp, idx) => (
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
                        {Array.isArray(evalRunConfig?.kb_ids) && evalRunConfig?.kb_ids.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {evalRunConfig.kb_ids.map((kb, idx) => (
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
                            <Switch
                              checked={evalRunConfig?.enable_input_guardrail || false}
                              disabled
                              className="h-4 w-8"
                            />
                            <Switch
                              checked={evalRunConfig?.enable_output_guardrail || false}
                              disabled
                              className="h-4 w-8"
                            />
                          </div>
                        </div>
                        {evalRunConfig?.guardrail_hint && (
                          <div className="text-xs bg-muted p-1.5 rounded mt-1 truncate" title={evalRunConfig.guardrail_hint}>
                            {evalRunConfig.guardrail_hint}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* 右侧：评估设置 + 时间信息（上下排列） */}
                <div className="flex flex-col gap-4">
                  {/* 评估设置 */}
                  <div>
                    <h3 className="font-semibold mb-3 flex items-center gap-1.5 text-sm">
                      <BarChart2 className="h-4 w-4" /> 评估设置
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500">类型</span>
                        <Badge variant="outline" className="text-xs py-0.5 px-2">
                          {evalRunConfig?.evaluator_config?.name === "ExactMatch" ? "精确匹配" : "LLM 评判"}
                        </Badge>
                      </div>
                      {evalRunConfig?.evaluator_config?.name === "ExactMatch" && (
                        <>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-500">大小写</span>
                            {evalRunConfig.evaluator_config.case_sensitive ? (
                              <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                            ) : (
                              <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                            )}
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-gray-500">标点</span>
                            {evalRunConfig.evaluator_config.ignore_punctuation ? (
                              <CheckCircle className="text-green-500 h-3.5 w-3.5" />
                            ) : (
                              <CircleXIcon className="text-red-500 h-3.5 w-3.5" />
                            )}
                          </div>
                        </>
                      )}
                      {evalRunConfig?.evaluator_config?.name === "LLMJudge" && (
                        <div className="flex items-center justify-between">
                          <span className="text-gray-500">模型</span>
                          <span className="font-medium text-xs">{evalRunConfig.evaluator_config.model_id || "未指定"}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 时间信息 */}
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

        <Card>
          <CardHeader>
            <CardTitle>执行详情 ({allExpItems.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[50px]"></TableHead>
                    <TableHead className="w-[15%]">样本ID</TableHead>
                    <TableHead className="w-[40%]">问题</TableHead>
                    <TableHead className="w-[10%]">状态</TableHead>
                    <TableHead className="w-[10%]">得分</TableHead>
                    {/* <TableHead>原因</TableHead> */}
                    <TableHead>耗时</TableHead>
                    <TableHead className="w-[100px]">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {runDetailItems.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} className="h-24 text-center">
                        暂无数据
                      </TableCell>
                    </TableRow>
                  ) : (
                    runDetailItems.map((sample) => (
                      <Fragment key={sample.id}>
                        <TableRow
                          className={isRowExpanded(sample.id) ? "bg-muted/50" : ""}
                        >
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
                          <TableCell className="font-medium">{sample.id}</TableCell>
                          <TableCell
                            className="whitespace-normal break-words min-w-[250px] max-w-[400px] py-2"
                          >
                            {sample.input.substring(0, 200)}...
                          </TableCell>
                          <TableCell>
                            {getStatusBadge(sample.status)}
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
                          </TableCell>
                        </TableRow>

                        {expandedRows.includes(sample.id) && (
                          <TableRow className="bg-muted/20">
                            <TableCell colSpan={8} className="p-0 border-0 w-full">
                              <div className="p-4 bg-background border rounded-lg">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                                  {/* 左侧：问答对评估 */}
                                  <div className="space-y-4">
                                    <h4 className="font-medium flex items-center gap-2 text-lg">
                                      <MessageSquare className="h-5 w-5 text-primary" />
                                      <span>问答对评估</span>
                                    </h4>

                                    <div className="space-y-4">
                                      {/* 问题 */}
                                      <div className="space-y-2">
                                        <div className="flex items-center gap-2">
                                          <div className="bg-blue-100 dark:bg-blue-900/30 px-3 py-1 rounded-lg text-blue-800 dark:text-blue-300 font-medium flex items-center">
                                            <MessageSquare className="h-4 w-4 mr-2" />
                                            问题
                                          </div>
                                          <span className="text-muted-foreground text-sm">用户输入的问题</span>
                                        </div>
                                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border min-h-[80px]">
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.input}</p>
                                        </div>
                                      </div>

                                      {/* 参考答案 */}
                                      <div className="space-y-2">
                                        <div className="flex items-center gap-2">
                                          <div className="bg-green-100 dark:bg-green-900/30 px-3 py-1 rounded-lg text-green-800 dark:text-green-300 font-medium flex items-center">
                                            <CheckCircle className="h-4 w-4 mr-2" />
                                            参考答案
                                          </div>
                                          <span className="text-muted-foreground text-sm">预期的标准答案</span>
                                        </div>
                                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border min-h-[120px] overflow-y-auto">
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.expected_output}</p>
                                        </div>
                                      </div>

                                      {/* 模型响应 */}
                                      <div className="space-y-2">
                                        <div className="flex items-center gap-2">
                                          <div className="bg-purple-100 dark:bg-purple-900/30 px-3 py-1 rounded-lg text-purple-800 dark:text-purple-300 font-medium flex items-center">
                                            <Bot className="h-4 w-4 mr-2" />
                                            模型响应
                                          </div>
                                          <span className="text-muted-foreground text-sm">LLM生成的响应</span>
                                        </div>
                                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border h-[400px] overflow-y-auto ">
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.actual_output}</p>
                                        </div>
                                      </div>
                                    </div>
                                  </div>

                                  {/* 右侧：执行信息 */}
                                  <div className="space-y-6">
                                    {/* 得分原因 */}
                                    <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <MessageSquare className="h-5 w-5 text-primary" />
                                        <h4 className="font-medium text-lg">得分原因</h4>
                                      </div>
                                      <div className="p-4 bg-muted rounded-lg">
                                        <p className="text-muted-foreground leading-relaxed">{sample.reason}</p>
                                      </div>
                                    </div>

                                    {/* 执行时间线 */}
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
                                    {/* 执行日志 */}
                                    <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <Terminal className="h-5 w-5 text-muted-foreground" />
                                        <h4 className="font-medium text-lg">执行日志</h4>
                                      </div>
                                      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border h-[500px] overflow-y-auto font-mono text-sm">
                                        {sample.execution_metadata && sample.execution_metadata?.length > 0 ? (
                                          <div className="space-y-4">
                                            {sample.execution_metadata.map((item, index) => {
                                              try {
                                                // 解析函数参数
                                                const args = JSON.parse(item.function.arguments);
                                                console.log("Parsed args:", args);
                                                return (
                                                  <div key={index} className="border-l-2 border-blue-500 pl-3 py-1">
                                                    <div className="flex items-start gap-2 mb-2">
                                                      <Badge className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2 py-0.5 rounded text-xs font-medium">{index}</Badge>
                                                      <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2 py-0.5 rounded text-xs font-medium">
                                                        {item.function.name}
                                                      </span>
                                                    </div>

                                                    {/* 参数展示 */}
                                                    <div className="ml-2 mb-2">
                                                      <span className="text-xs font-medium text-gray-600 dark:text-gray-400">参数:</span>
                                                      {/* <p>{args}</p> */}
                                                      <div className="ml-2 mt-1 text-xs bg-white dark:bg-gray-700 rounded p-1 border border-border">
                                                        {Object.entries(args).map(([key, value]) => (
                                                          <div key={key} className="flex">
                                                            <span className="text-blue-600 dark:text-blue-400">{key}:</span>
                                                            <span className="ml-1 truncate max-w-[200px]">{String(value)}</span>
                                                          </div>
                                                        ))}
                                                      </div>
                                                    </div>

                                                    {/* 结果展示 */}
                                                    <div className="ml-2 mb-2">
                                                      <span className="text-xs font-medium text-gray-600 dark:text-gray-400">观察结果:</span>
                                                      <div className="ml-2 mt-1 text-xs bg-white dark:bg-gray-700 rounded p-1 border border-border whitespace-pre-wrap h-[100px] overflow-y-auto">
                                                        <p>{item.observation}</p>
                                                      </div>
                                                    </div>
                                                  </div>
                                                );
                                              } catch (e) {
                                                console.error("Error parsing execution metadata:", e);
                                                return (
                                                  <div key={index} className="border-l-2 border-red-500 pl-3 py-1 text-red-600 dark:text-red-400 text-xs">
                                                    <div className="font-medium">解析错误</div>
                                                    <div>无法解析执行元数据: {item.id}</div>
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

  )
}