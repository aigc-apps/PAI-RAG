"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Loader2, CheckCircle } from "lucide-react";
import { PreviewButton } from "@/components/ui/preview-button";

interface KnowledgeBaseFile {
  id: string;
  name: string;
  type: string;
  size: string;
  status: string;
  content: string;
  uploadedAt: string;
}

interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  files?: KnowledgeBaseFile[]; // 新增文件列表字段
}
export default function KnowledgeBaseDetailPage({
  knowledgebase_id,
  setActiveTab,
}: {
  knowledgebase_id: string;
  setActiveTab: (tab: string) => void;
}) {
  const [knowledgebases, setKnowledgeBases] = useState(Array<KnowledgeBase>); // 知识库列表
  const [knowledgebasesloading, setKnowledgeBasesLoading] = useState(true); // 加载状态
  const [knowledgebasesrror, setKnowledgeBasesError] = useState(""); // 错误信息
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        // const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        // // 模拟 API 请求：/v1/knowledgebases
        // const res = await fetch(`http://localhost:${port}/v1/knowledgebases`);
        // if (!res.ok) throw new Error("获取知识库列表失败");
        // const data = await res.json();

        // // 并行获取每个知识库的文件列表
        // const knowledgeBasesWithFiles = await Promise.all(
        //   data.map(async (kb) => {
        //     try {
        //       // 模拟 API 请求：/v1/knowledgebases/${kb.id}/files
        //       const files = await fetch(`http://localhost:${port}/v1/knowledgebases/${kb.id}/files`);
        //       return { ...kb, files }; // 合并文件列表
        //     } catch (err) {
        //       console.error(`获取知识库 ${kb.id} 文件失败`, err);
        //       return { ...kb, files: [] }; // 失败时返回空数组
        //     }
        //   })
        // );

        // const data = [
        //     { id: '1', name: '产品文档库', description: '包含所有产品技术规格与使用指南' },
        //     { id: '2', name: '技术白皮书', description: '深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计' },
        //     { id: '3', name: '用户指南', description: '从入门到精通的全流程操作手册' },
        //     { id: '4', name: 'API 文档', description: 'RESTful 接口规范与示例' }
        // ]

        const knowledgeBasesWithFiles = [
          {
            id: "1",
            name: "产品文档库",
            description: "包含所有产品技术规格与使用指南",
            files: [
              {
                id: "f1",
                name: "产品规格书.pdf",
                type: "PDF",
                size: "2.1MB",
                status: "done",
                content:
                  '# 系统文档指南\n\n## 简介\n\n这是使用现代样式渲染的 Markdown 文档示例。以下展示了各种格式的渲染效果：\n\n### 标题层级\n\n#### 三级标题下的四级标题\n\n- 支持无序列表\n\n- 支持有序列表\n\n1. 嵌套有序列表\n\n2. 第二项\n\n**强调文本** 和 `行内代码` 示例\n\n```python\n\n# 代码块示例\n\ndef hello():\n\nprint("现代 Markdown 样式")',
                uploadedAt: "2025-03-15",
              },
              {
                id: "f2",
                name: "安装指南.pdf",
                type: "PDF",
                size: "1.8MB",
                status: "pending",
                content: "# 安装指南",
                uploadedAt: "2025-03-10",
              },
              {
                id: "f3",
                name: "API文档.pdf",
                type: "PDF",
                size: "3.2MB",
                status: "pending",
                content: "# API文档",
                uploadedAt: "2025-03-05",
              },
            ],
          },
          {
            id: "2",
            name: "技术白皮书",
            description: "深度解析核心算法与架构设计",
            files: [
              {
                id: "f4",
                name: "分布式架构设计.pdf",
                type: "PDF",
                size: "4.5MB",
                status: "pending",
                content: "# 分布式架构设计",
                uploadedAt: "2025-03-18",
              },
              {
                id: "f5",
                name: "机器学习白皮书.pdf",
                type: "PDF",
                size: "6.2MB",
                status: "done",
                content: "# 机器学习白皮书",
                uploadedAt: "2025-03-12",
              },
            ],
          },
          {
            id: "3",
            name: "用户指南",
            description: "从入门到精通的全流程操作手册",
            files: [],
          },
          {
            id: "4",
            name: "API 文档",
            description: "RESTful 接口规范与示例",
            files: [],
          },
        ];

        setKnowledgeBases(knowledgeBasesWithFiles || []); // 更新状态
      } catch (err: any) {
        setKnowledgeBasesError(err || "加载失败");
      } finally {
        setKnowledgeBasesLoading(false);
      }
    };
    fetchConfigs();
  }, []);
  const knowledgebase = knowledgebases.find((kb) => kb.id === knowledgebase_id);
  if (!knowledgebase) {
    return <div className="p-6">加载中...</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6 flex items-center gap-2">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button
                  variant="link"
                  className="px-0"
                  onClick={() => setActiveTab(`/knowledgebase`)}
                >
                  知识库
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>{knowledgebase.name}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>知识库详情</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4">名称：{knowledgebase.name}</p>
          <p className="text-muted-foreground mb-4">
            描述：{knowledgebase.description}
          </p>

          {knowledgebase.files && knowledgebase.files.length > 0 ? (
            <>
              <h3 className="text-lg font-semibold mt-6 mb-3">文件列表</h3>
              <ScrollArea className="h-[400px] rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>文件名</TableHead>
                      <TableHead>文件格式</TableHead>
                      <TableHead>文件大小</TableHead>
                      <TableHead>状态</TableHead>
                      <TableHead>上传时间</TableHead>
                      <TableHead>操作</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {knowledgebase.files.map((file) => (
                      <TableRow key={file.id}>
                        <TableCell>
                          <Button
                            variant="link"
                            className="font-medium text-blue-600"
                          >
                            {file.name}
                          </Button>
                        </TableCell>
                        <TableCell>{file.type}</TableCell>
                        <TableCell>{file.size}</TableCell>
                        <TableCell>
                          {file.status === "pending" ? (
                            <div className="flex items-center text-yellow-500">
                              <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                              解析中
                            </div>
                          ) : file.status === "done" ? (
                            <div className="flex items-center text-green-500">
                              <CheckCircle className="mr-1 h-4 w-4" />
                              解析完成
                            </div>
                          ) : (
                            <span>{file.status}</span> // 兜底显示原始状态
                          )}
                        </TableCell>
                        <TableCell>{file.uploadedAt}</TableCell>
                        <TableCell>
                          {/* <Button variant="link" className="text-sm text-blue-600">预览</Button> */}
                          <PreviewButton markdownContent={file.content} />
                          <Button
                            variant="link"
                            className="text-sm text-blue-600"
                          >
                            查看切片
                          </Button>
                          <Button
                            variant="link"
                            className="text-sm text-blue-600"
                          >
                            删除
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            </>
          ) : (
            <p className="text-muted-foreground">暂无文件</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
