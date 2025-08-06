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
import { ChevronRight, Plus, BookTextIcon } from "lucide-react";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";

export interface KnowledgeBase {
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
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const kbSizePerPage = 6;

  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const API_BASE =
          process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
        const res = await fetch(
          `${API_BASE}/v1/config/knowledgebases?page=${page}&size=${kbSizePerPage}`,
        );
        if (!res.ok) throw new Error("获取知识库列表失败");
        const json_data = await res.json();
        const data = json_data.data.items;
        setKnowledgeBases(data || []); // 更新状态
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setKnowledgeBasesError(err || "加载失败");
      } finally {
        setKnowledgeBasesLoading(false);
      }
    };

    fetchConfigs();
  }, [page]);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };
  const deleteKnowledgebase = async (kb_id: string) => {
    try {
      const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
      const res = await fetch(`${API_BASE}/v1/config/knowledgebases/${kb_id}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!res.ok) {
        throw new Error("删除失败，请检查网络或配置");
      }

      // 显示成功提示（可选）

      // 删除成功后更新本地状态
      setKnowledgeBases((prev) => prev.filter((config) => config.id !== kb_id));
    } catch (err: any) {}
    // 显示错误提示
  };

  return (
    <div className="flex flex-col h-screen p-6 space-y-6">
      {/* 顶部标题栏 */}
      <div className="flex justify-between items-center h-1/10">
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
      <div className="h-4/5">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
          {knowledgebases.map((base) => (
            <Card
              key={base.id}
              className="flex flex-col border rounded-lg shadow-sm h-full gap-4 py-4"
            >
              <CardHeader>
                <CardTitle className="text-lg flex gap-2">
                  <BookTextIcon className="h-7" /> {base.name}
                </CardTitle>
              </CardHeader>

              <CardContent className="pt-0">
                <p className="text-md text-muted-foreground line-clamp-3">
                  {base.description
                    ? base.description
                    : "暂时还没有描述，可以去设置页面添加哦。"}
                </p>
              </CardContent>
              <CardFooter className="mt-auto pt-0 flex justify-end pt-3 px-2">
                <Button
                  variant="link"
                  onClick={() => deleteKnowledgebase(base.id)}
                  className="text-sm text-primary text-red-600 hover:text-primary/80 underline-offset-4 hover:underline"
                >
                  删除
                </Button>

                <Button
                  variant="link"
                  className="text-md text-primary text-blue-600 hover:text-primary/80 underline-offset-4 hover:underline"
                  onClick={() =>
                    setActiveTab(`/knowledgebase/details/${base.id}`)
                  }
                >
                  查看详情 <ChevronRight className="ml-1" size={20} />
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
      {/* 分页组件 */}
      <div className="flex justify-center items-center h-1/10">
        <PaginationComponent
          currentPage={page}
          totalPages={totalPages}
          onPageChange={handlePageChange}
        />
      </div>
    </div>
  );
}
