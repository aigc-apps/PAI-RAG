'use client';

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useState, useEffect, use, useRef } from "react";
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
import { ChevronDown, ChevronRight, Clock, Terminal, MessageSquare, CheckCircle, Bot } from "lucide-react";
import { Fragment } from "react";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";

// 定义样本类型
interface Sample {
  id: string
  question: string
  answer: string
  response: string
  status: "success" | "failed" | "running"
  score: number
  reason: string
  start_time: string
  end_time: string
  logs?: string
}

// 实验数据类型（与您提供的mockData一致）
type ExperimentItem = {
  id: string
  count: number
  settings: {
    llm: string
    mcp: string[]
    search: boolean
  }
  status: "running" | "success" | "failed"
  avg_score: number
  create_time: string
  finished_time: string
}

// 模拟实验数据（您可以从您的数据源导入）
const mockData: ExperimentItem[] = [
  {
    id: "exp-f4c3-4cad-be07",
    count: 2,
    settings: {
      llm: "qwen-max",
      mcp: ["amap", "browser-use"],
      search: false,
    },
    status: "running",
    avg_score: 0.0,
    create_time: "2025-08-28 17:09",
    finished_time: ""
  },
  {
    id: "exp-3d80-4913-a07d",
    count: 1,
    settings: {
      llm: "qwen-max",
      mcp: ["browser-use"],
      search: true,
    },
    status: "success",
    avg_score: 0.7,
    create_time: "2025-08-28 16:21",
    finished_time: "2025-08-28 16:40"
  }
]

export default function ExperimentDetailPage({ params }: { params: Promise<{ evalId: string, expId: string }> }) {
  const { evalId, expId } = use(params);
  const experiment = mockData.find(exp => exp.id === expId)
  const router = useRouter();
  // 修改状态管理，支持多行同时展开
  const [expandedRows, setExpandedRows] = useState<string[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const pageSize = 3;

  // 切换行的展开状态
  const toggleRow = (id: string) => {
    setExpandedRows(prev =>
      prev.includes(id)
        ? prev.filter(rowId => rowId !== id)
        : [...prev, id]
    )
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

  useEffect(() => {
    const fetchConfigs = async () => {
      setIsLoading(true);

      // 模拟API调用获取评估数据
      const mockData: Sample[] = [
        {
          id: "sample-9863-ijhg-01",
          question: "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.",
          answer: "17",
          response: "17",
          status: "success",
          score: 1.0,
          reason: "Generated correct response within time limit",
          start_time: "2025-08-29 08:50",
          end_time: "2025-08-29 08:59",
          logs: "Successfully processed request\nUsed browser-use MCP\nReturned valid response"
        },
        {
          id: "sample-9863-ijhg-02",
          question: "Of the authors (First M. Last) that worked on the paper \"Pie Menus or Linear Menus, Which Is Better?\" in 2015, what was the title of the first paper authored by the one that had authored prior papers?",
          answer: "Mapping Human Oriented Information to Software Agents for Online Systems Usage",
          response: "Online Systems Usage",
          status: "failed",
          score: 0.1,
          reason: "Wrong",
          start_time: "2025-08-29 08:53",
          end_time: "2025-08-29 08:56",
          logs: "Error: Timeout after 30 seconds\nAPI call failed\nRetrying..."
        },
        {
          id: "sample-9863-ijhg-03",
          question: "In Emily Midkiff's June 2014 article in a journal named for the one of Hreidmar's sons that guarded his house, what word was quoted from two different authors in distaste for the nature of dragon depictions?",
          answer: "fluffy",
          response: "",
          status: "running",
          score: 0.0,
          reason: "Execution still in progress",
          start_time: "2025-08-29 09:10",
          end_time: "2025-08-29 09:29",
          logs: "Processing request...\nWaiting for browser-use MCP response"
        },
        {
          id: "sample-9863-ijhg-04",
          question: "Under DDC 633 on Bielefeld University Library's BASE, as of 2020, from what country was the unknown language article with a flag unique from the others?",
          answer: "Guatemala",
          response: "",
          status: "running",
          score: 0.0,
          reason: "Execution still in progress",
          start_time: "2025-08-29 09:50",
          end_time: "2025-08-29 09:55",
          logs: "Processing request...\nWaiting for browser-use MCP response"
        },
        {
          id: "sample-9863-ijhg-05",
          question: "What is the capital of France and what is the square root of 144? Please provide both answers separated by a comma.",
          answer: "Paris, 12",
          response: "Paris 12",
          status: "success",
          score: 0.9,
          reason: "Generated correct response within time limit",
          start_time: "2025-08-29 08:50",
          end_time: "2025-08-29 08:59",
          logs: "Successfully processed request\nUsed browser-use MCP\nReturned valid response"
        }
      ];

      // 计算分页 [[7]]
      const startIndex = (page - 1) * pageSize;
      const paginatedData = mockData.slice(startIndex, startIndex + pageSize);

      setSamples(paginatedData);
      setTotalPages(Math.ceil(mockData.length / pageSize));
      setIsLoading(false);
    };

    fetchConfigs();
  }, [page]);

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
                      {evalId}
                    </Button>
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem>
                  <BreadcrumbPage>{expId}</BreadcrumbPage>
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
              <CardTitle className="text-2xl">实验: {experiment.id}</CardTitle>
              <Badge variant={getStatusVariant(experiment.status)} className="w-fit">
                {experiment.status.charAt(0).toUpperCase() + experiment.status.slice(1)}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">详情</h3>
                  <div className="space-y-2">
                    <p><span className="text-gray-500">状态:</span> {experiment.status}</p>
                    <p><span className="text-gray-500">所有样本数:</span> {experiment.count}</p>
                    <p><span className="text-gray-500">平均得分:</span>
                      <Badge variant="secondary" className="ml-2">
                        {experiment.avg_score.toFixed(2)}
                      </Badge>
                    </p>
                    <p><span className="text-gray-500">创建时间:</span> {experiment.create_time}</p>
                    {experiment.finished_time && (
                      <p><span className="text-gray-500">完成时间:</span> {experiment.finished_time}</p>
                    )}
                  </div>
                </div>

                
              </div>
              <div className="space-y-4">
                  <h3 className="font-semibold mb-2">设置</h3>
                  <div className="space-y-2">
                    <p><span className="text-gray-500">LLM:</span> {experiment.settings.llm}</p>
                    <p><span className="text-gray-500">MCP:</span>
                      {experiment.settings.mcp.map((mcp, index) => (
                        <Badge key={index} variant="outline" className="ml-1">
                          {mcp}
                        </Badge>
                      ))}
                    </p>
                    <p><span className="text-gray-500">搜索:</span>
                      <Badge variant={experiment.settings.search ? "secondary" : "destructive"} className="ml-2">
                        {experiment.settings.search ? "Yes" : "No"}
                      </Badge>
                    </p>
                  </div>
                </div>

              {/* <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">性能指标</h3>
                  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-gray-500">所有样本数:</span>
                      <span className="font-medium">{experiment.count}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                      <div
                        className="bg-blue-600 h-2.5 rounded-full"
                        style={{ width: `${Math.min(100, experiment.avg_score * 100)}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between mt-1 text-xs text-gray-500">
                      <span>0.0</span>
                      <span>1.0</span>
                    </div>
                  </div>
                </div>
              </div> */}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>样本执行详情 ({samples.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[50px]"></TableHead>
                    <TableHead>样本ID</TableHead>
                    <TableHead>问题</TableHead>
                    <TableHead>状态</TableHead>
                    <TableHead>得分</TableHead>
                    <TableHead>原因</TableHead>
                    <TableHead>耗时</TableHead>
                    <TableHead className="w-[100px]">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {samples.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} className="h-24 text-center">
                        暂无数据
                      </TableCell>
                    </TableRow>
                  ) : (
                    samples.map((sample) => (
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
                          <TableCell className="font-medium">{sample.question.substring(0, 30)}...</TableCell>
                          <TableCell>
                            <Badge variant={getStatusVariant(sample.status)}>
                              {sample.status.charAt(0).toUpperCase() + sample.status.slice(1)}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center">
                              <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                <div
                                  className={`h-2 rounded-full ${sample.status === 'success' ? 'bg-green-500' : sample.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'}`}
                                  style={{ width: `${Math.min(100, sample.score * 100)}%` }}
                                ></div>
                              </div>
                              <span>{sample.score.toFixed(2)}</span>
                            </div>
                          </TableCell>
                          <TableCell className="max-w-xs truncate" title={sample.reason}>
                            {sample.reason}
                          </TableCell>
                          <TableCell>
                            {sample.start_time && sample.end_time
                              ? `${Math.floor((new Date(sample.end_time).getTime() - new Date(sample.start_time).getTime()) / 1000)}s`
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
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.question}</p>
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
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.answer}</p>
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
                                          <p className="whitespace-pre-wrap leading-relaxed">{sample.response}</p>
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
                                            <span className="ml-2">{sample.start_time || "N/A"}</span>
                                          </div>
                                        </div>
                                        <div className="flex items-start">
                                          <div className="w-3 h-3 rounded-full bg-success mt-1 mr-3"></div>
                                          <div>
                                            <span className="text-muted-foreground font-medium">完成: </span>
                                            <span className="ml-2">{sample.end_time || "进行中..."}</span>
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
                                        {sample.logs || "暂无日志信息"}
                                      </div>
                                    </div>

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