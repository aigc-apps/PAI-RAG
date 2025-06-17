"use client";

import { Button } from "@/components/ui/button";
import React, { useState, useEffect } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { ChevronRight, Plus } from "lucide-react";

interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
}

export default function KnowledgeBase({
  setActiveTab,
}: {
  setActiveTab: (tab: string) => void;
}) {
  const [knowledgebases, setKnowledgeBases] = useState(Array<KnowledgeBase>); // 知识库列表
  const [knowledgebasesloading, setKnowledgeBasesLoading] = useState(true); // 加载状态
  const [knowledgebasesrror, setKnowledgeBasesError] = useState(""); // 错误信息

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        // const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        // const res = await fetch(`http://localhost:${port}/v1/knowledgebases`);
        // if (!res.ok) throw new Error("获取知识库列表失败");
        // const data = await res.json();
        const data = [
          {
            id: "1",
            name: "产品文档库",
            description: "包含所有产品技术规格与使用指南",
          },
          {
            id: "2",
            name: "技术白皮书",
            description:
              "深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计,深度解析核心算法与架构设计",
          },
          {
            id: "3",
            name: "用户指南",
            description: "从入门到精通的全流程操作手册",
          },
          { id: "4", name: "API 文档", description: "RESTful 接口规范与示例" },
        ];
        setKnowledgeBases(data || []); // 更新状态
      } catch (err: any) {
        setKnowledgeBasesError(err || "加载失败");
      } finally {
        setKnowledgeBasesLoading(false);
      }
    };

    fetchConfigs();
  }, []);

  return (
    <div className="p-6 space-y-6">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">知识库</h1>
        <Button
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 w-40"
          onClick={() => setActiveTab("/knowledgebase/create")}
        >
          <Plus className="w-6 h-6" />
          新建知识库
        </Button>
      </div>

      {/* 卡片容器 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
        {knowledgebases.map((base) => (
          <Card
            key={base.id}
            className="flex flex-col border rounded-lg shadow-sm h-full"
          >
            <CardHeader>
              <CardTitle className="text-sm font-medium">{base.name}</CardTitle>
            </CardHeader>
            {base.description && (
              <CardContent className="pt-0">
                <p className="text-xs text-muted-foreground">
                  {base.description
                    ? base.description.slice(0, 50) +
                      (base.description.length > 50 ? "..." : "")
                    : ""}
                </p>
              </CardContent>
            )}
            <CardFooter className="mt-auto pt-0 flex justify-end">
              <Button
                variant="link"
                className="text-sm text-primary text-blue-600 hover:text-primary/80 underline-offset-4 hover:underline"
                onClick={() =>
                  setActiveTab(`/knowledgebase/details/${base.id}`)
                }
              >
                查看详情 <ChevronRight className="ml-1" size={16} />
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
