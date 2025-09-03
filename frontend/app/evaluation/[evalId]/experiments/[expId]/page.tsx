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

type ExperimentDetailsItem = {
  id: string
  input: string
  expected_output: string
  actual_output: string
  status: string
  score: number
  dataset_metadata?: {
    Steps?: string
    Tools?: string
  }
  execution_metadata?: {}
  created_at: string
  updated_at: string
}


export default function ExperimentDetailPage({ params }: { params: Promise<{ evalId: string, expId: string }> }) {
  const { evalId, expId } = use(params);
  const router = useRouter();
  const [evalConfig, setEvalConfig] = useState<EvalConfig>();
  const [experiment, setExperiment] = useState<ExperimentItem>();
  const [runDetailItems, setRunDetailItems] = useState<ExperimentDetailsItem[]>([]);
  const [expandedRows, setExpandedRows] = useState<string[]>([]);
  
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
        const url = `/api/config/evaluation/${evalId}/experiments/${expId}/details?page=${pageRef.current}&size=${pageSize}`

        try {
          isRefreshing = true;
          const files_res = await fetch(url);
          if (!files_res.ok) throw new Error('获取实验列表失败');
    
          const exp_json_data = await files_res.json();
          console.log('获取实验reponse:', exp_json_data);
          const data = exp_json_data.data.items;
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

    useEffect(() => {
          const fetchConfigs = async () => {
              setIsLoading(true);
              try {
                  const [evalRes, expDataRes] = await Promise.all([
                    fetch(`/api/config/evaluation/${evalId}`),
                    fetch(`/api/config/evaluation/${evalId}/experiments/${expId}`),
                  ]);
                  
                  const eval_data = await evalRes.json();
                  const evalData = eval_data.data;
                  console.log('evalData:', evalData);
                  setEvalConfig(evalData);

                  const exp_data = await expDataRes.json();
                  const expData = exp_data.data;
                  console.log('expData:', expData);
                  setExperiment(expData);
              } catch (err: any) {
                  toast.error(err.message);
              } finally {
                  setIsLoading(false);
              }
          };
          fetchConfigs();
      }, []);

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
            <h2 className="text-2xl font-bold">Experiment Not Found</h2>
            <p className="text-gray-500 mt-2">The experiment with the specified ID does not exist.</p>
          </CardContent>
        </Card>
      </div>
    )
  }


  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  // 获取状态徽章的变体
  const getStatusVariant = (status: string) => {
    if (status === "success") return "secondary"
    if (status === "failed") return "destructive"
    return "default"
  }

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
                  <BreadcrumbLink asChild>
                    <Button
                      variant="link"
                      className="px-0"
                      onClick={() => router.push(`/evaluation/${evalId}/experiments`)}
                    >
                      experiments
                    </Button>
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem>
                  <BreadcrumbPage>{experiment.name}</BreadcrumbPage>
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
            <div className="grid  grid-cols-2 flex gap-6">
                <div className="col-span-1">
                  <h3 className="font-semibold mb-2">详情</h3>
                  <div className="space-y-2">
                    <p><span className="text-gray-500">ID:</span> {experiment.id}</p>
                    <p><span className="text-gray-500">描述:</span> {experiment.description}</p>
                    <p><span className="text-gray-500">状态:</span> {getStatusBadge(experiment.status)} </p>
                    <p><span className="text-gray-500">所有样本数:</span> {experiment.samples_count}</p>
                    <p><span className="text-gray-500">平均得分:</span>
                      <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200">
                        {experiment.avg_score.toFixed(2)}
                      </Badge>
                    </p>
                    <p><span className="text-gray-500">创建时间:</span> {formatBeijingTime(experiment.created_at)}</p>
                    {['success', 'failed'].includes(experiment.status) && (
                      <p><span className="text-gray-500">完成时间:</span> {formatBeijingTime(experiment.updated_at)}</p>
                    )}
                  </div>
                </div>
                <div className="col-span-1">
                  <h3 className="font-semibold mb-2">实验设置</h3>
                  <div className="space-y-2">
                    <div className="flex">
                      <span className="text-gray-500">基模型:</span> {experiment.run_config.model_id}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">联网搜索:</span> 
                      {experiment.run_config.enable_search ? (
                          <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                      ) : (
                          <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">Agentic模式:</span> 
                       {experiment.run_config.enable_agent ? (
                          <CheckCircle className="text-green-500 h-4 w-4 ml-2" />
                      ) : (
                          <CircleXIcon className="text-red-500 h-4 w-4 ml-2" />
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">MCP Server:</span> 
                      {Array.isArray(experiment.run_config.mcp_ids) && experiment.run_config.mcp_ids.length === 0 ? (
                          <p className="text-muted-foreground pl-2">尚未配置MCP</p>
                      ) : (
                          Array.isArray(experiment.run_config.mcp_ids) && experiment.run_config.mcp_ids.map((mcp, idx) => (
                              <Badge key={mcp || idx}>{mcp}</Badge>
                          ))
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">知识库:</span>
                      {Array.isArray(experiment.run_config.kb_ids) && experiment.run_config.kb_ids.length === 0 ? (
                          <p className="text-muted-foreground pl-2">尚未配置知识库</p>
                      ) : (
                          Array.isArray(experiment.run_config.kb_ids) && experiment.run_config.kb_ids.map((kb, idx) => (
                              <Badge key={kb || idx}>{kb}</Badge>
                          ))
                      )}
                    </div>
                    <div className="flex">
                      <span className="text-gray-500">安全护栏:</span>
                      <div className="flex gap-4 pl-6 text-sm items-center">
                        <div className="space-y-2">
                            <Switch
                                id="enable_input_check"
                                checked={experiment.run_config.enable_input_guardrail || false}
                            />
                            <Label htmlFor="input_guardrail" className="w-[120px]">
                                输入护栏
                            </Label>
                        </div>
                        <div className="space-y-2">
                            <Switch
                                id="enable_output_check"
                                checked={experiment.run_config.enable_output_guardrail || false}
                            />
                            <Label htmlFor="output_guardrail" className="w-[120px]">
                                输出护栏
                            </Label>
                        </div>

                        <div className="space-y-1">
                            <Input
                                className="w-120"
                                value={experiment.run_config.guardrail_hint}
                                disabled
                            />
                            <Label htmlFor="guardrail_hint" className="w-[120px]">
                                默认护栏提示
                            </Label>
                        </div>
                    </div>
                    </div>
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
                                  className={`h-2 rounded-full ${sample.status === 'success' ? 'bg-green-500' : sample.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'}`}
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
                            {sample.created_at && sample.updated_at
                              ? calculateTimeDifference(sample.created_at, sample.updated_at)
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
                                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border min-h-[120px] overflow-y-auto">
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.actual_output}</p>
                                        </div>
                                      </div>
                                    </div>
                                  </div>

                                  {/* 右侧：执行信息 */}
                                  <div className="space-y-6">
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
                                            <span className="text-muted-foreground font-medium">开始: </span>
                                            <span className="ml-2">{formatBeijingTime(sample.created_at) || "N/A"}</span>
                                          </div>
                                        </div>
                                        <div className="flex items-start">
                                          <div className="w-3 h-3 rounded-full bg-success mt-1 mr-3"></div>
                                          <div>
                                            <span className="text-muted-foreground font-medium">完成: </span>
                                            <span className="ml-2">{formatBeijingTime(sample.updated_at) || "进行中..."}</span>
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
                                      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-border h-[200px] overflow-y-auto font-mono text-sm">
                                        {/* {sample.logs || "暂无日志信息"} */}
                                        暂无日志信息
                                      </div>
                                    </div>

                                    {/* 得分原因 */}
                                    {/* <div className="space-y-3">
                                      <div className="flex items-center gap-2">
                                        <MessageSquare className="h-5 w-5 text-primary" />
                                        <h4 className="font-medium text-lg">得分原因</h4>
                                      </div>
                                      <div className="p-4 bg-muted rounded-lg">
                                        <p className="text-muted-foreground leading-relaxed">{sample.reason}</p>
                                      </div>
                                    </div> */}
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