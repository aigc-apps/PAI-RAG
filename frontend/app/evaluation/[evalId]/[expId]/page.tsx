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
import { ChevronDown, ChevronRight, Clock, Terminal, MessageSquare, CheckCircle, Bot, CircleXIcon } from "lucide-react";
import { Fragment } from "react";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import { EvalConfig } from '@/app/evaluation/[evalId]/page';
import { ExperimentItem, getStatusBadge } from "@/app/evaluation/[evalId]/experiments/page";
import { formatBeijingTime, calculateTimeDifference } from '@/app/knowledgebases/utils/utils';
import { toast } from 'sonner';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { EvalRunConfig } from '@/app/evaluation/[evalId]/settings/page';

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
      arguments: string // JSON stringified
    }
    type: string
    observation: string | null // JSON stringified
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
  const [expandedRows, setExpandedRows] = useState<string[]>([]);
  const [evalRunConfig, setEvalRunConfig] = useState<EvalRunConfig>();
  
  const [page, setPage] = useState(1);
  const pageRef = useRef(page);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const pageSize = 6;
  let isRefreshing = false;

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
          isRefreshing=false;
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
        <Card className="mb-6">
          <CardHeader>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
              <CardTitle className="text-2xl">实验: {experiment.name}</CardTitle>
              <div className="flex items-center gap-2">
                {getStatusBadge(experiment.status)}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid  grid-cols-3 flex gap-6">
                <div className="col-span-1">
                  <h3 className="font-semibold mb-2">详情</h3>
                  <div className="space-y-2">
                    <p><span className="text-gray-500">ID:</span> {experiment.id}</p>
                    <p><span className="text-gray-500">描述:</span> {experiment.description}</p>
                    <p><span className="text-gray-500">状态:</span> {getStatusBadge(experiment.status)} </p>
                    <p><span className="text-gray-500">所有样本数:</span> {experiment.samples_count}</p>
                    <p><span className="text-gray-500">平均得分:</span>
                      <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200">
                        {experiment.avg_score?(experiment.avg_score.toFixed(2)):("0.0")}
                      </Badge>
                    </p>
                    <p><span className="text-gray-500">创建时间:</span> {formatBeijingTime(experiment.created_at)}</p>
                    {['success', 'failed'].includes(experiment.status) && (
                      <p><span className="text-gray-500">完成时间:</span> {formatBeijingTime(experiment.updated_at)}</p>
                    )}
                  </div>
                </div>
                <div className="col-span-1">
                  <h3 className="font-semibold mb-2">应用设置</h3>
                  <div className="space-y-2">
                    <div className="flex">
                      <span className="text-gray-500">基模型:</span> {evalRunConfig?.model_id}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">联网搜索:</span> 
                      {evalRunConfig?.enable_search ? (
                          <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                      ) : (
                          <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">Agentic模式:</span> 
                       {evalRunConfig?.enable_agent ? (
                          <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                      ) : (
                          <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">MCP Server:</span> 
                      {Array.isArray(evalRunConfig?.mcp_ids) && evalRunConfig?.mcp_ids.length === 0 ? (
                          <p className="text-muted-foreground pl-2">尚未配置MCP</p>
                      ) : (
                          Array.isArray(evalRunConfig?.mcp_ids) && evalRunConfig?.mcp_ids.map((mcp, idx) => (
                              <Badge key={mcp || idx}>{mcp}</Badge>
                          ))
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">知识库:</span>
                      {Array.isArray(evalRunConfig?.kb_ids) && evalRunConfig?.kb_ids.length === 0 ? (
                          <p className="text-muted-foreground pl-2">尚未配置知识库</p>
                      ) : (
                          Array.isArray(evalRunConfig?.kb_ids) && evalRunConfig?.kb_ids.map((kb, idx) => (
                              <Badge key={kb || idx}>{kb}</Badge>
                          ))
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">安全护栏:</span>
                      <div className="gap-4 pl-6 text-sm items-center">
                        <div className="flex space-y-2">
                            <Label htmlFor="input_guardrail" className="w-[120px]">
                                输入护栏
                            </Label>

                            <Switch
                                id="enable_input_check"
                                checked={evalRunConfig?.enable_input_guardrail || false}
                            />
                            
                        </div>
                        <div className="flex space-y-2">
                            <Label htmlFor="output_guardrail" className="w-[120px]">
                                输出护栏
                            </Label>

                            <Switch
                                id="enable_output_check"
                                checked={evalRunConfig?.enable_output_guardrail || false}
                            />
                            
                        </div>

                        <div className="flex space-y-1">
                            <Label htmlFor="guardrail_hint" className="w-[120px]">
                                默认护栏提示
                            </Label>
                            <span>{evalRunConfig?.guardrail_hint ?? ''}</span>
                        </div>
                    </div>
                    </div>
                  </div>
                </div>
                <div className="col-span-1">
                  <h3 className="font-semibold mb-2">评估设置</h3>
                  <div className="space-y-2">
                    <div className="flex space-y-2">
                      <span className="text-gray-500 pr-6">评估器类型:</span>
                      <Badge variant="outline">
                          {evalRunConfig?.evaluator_config?.name === "ExactMatch" ? "精确匹配" : "LLM 评判"}
                      </Badge>
                    </div>
                    {evalRunConfig?.evaluator_config?.name === "ExactMatch" && (
                      <div>
                        <div className="flex space-y-2">
                          <span className="text-gray-500">区分大小写:</span> 
                          {evalRunConfig?.evaluator_config.case_sensitive ? (
                              <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                          ) : (
                              <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                          )}
                        </div>
                        <div className="flex space-y-2">
                          <span className="text-gray-500">忽略标点:</span> 
                          {evalRunConfig?.evaluator_config.ignore_punctuation ? (
                              <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                          ) : (
                              <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                          )}
                        </div>
                      </div>
                    )}
                    {evalRunConfig?.evaluator_config?.name === "LLMJudge" && (
                      <div className="flex">
                        <span className="text-gray-500">评估模型:</span> 
                        {evalRunConfig?.evaluator_config.model_id || "未指定"}
                      </div>
                    )}
                  </div>
                </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>执行详情 ({runDetailItems.length})</CardTitle>
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
                              {sample.input.substring(0,200)}...
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
                          {/* <TableCell className="max-w-xs truncate" title={sample.reason}>
                            {sample.reason}
                          </TableCell> */}
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
                                          ):(
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
                                    {/* <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <Terminal className="h-5 w-5 text-muted-foreground" />
                                        <h4 className="font-medium text-lg">执行日志</h4>
                                      </div>
                                      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border h-[200px] overflow-y-auto font-mono text-sm">
                                        暂无日志信息
                                      </div>
                                    </div> */}
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
                                                // 解析观察结果
                                                const observation = item.observation ? JSON.parse(item.observation) : null;
                                                console.log("Parsed observation:", observation);
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