"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Save, AlertCircleIcon } from "lucide-react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

import { KbConfig, MetadataConfig, KbConfigCard } from "../kbconfig";

export default function KnowledgeBaseCreatePage({
  setActiveTab,
}: {
  setActiveTab: (tab: string) => void;
}) {
  const kbConfig: KbConfig = {
    id: "",
    name: "",
    description: "",
    chunk_config: {
      parser_type: "Sentence",
      separator: "\n\n",
      chunk_size: "1000",
      chunk_overlap: "50",
    },
    embedding_model: "BAAI/bge-m3",
    retrieval_config: {
      retrieval_mode: "vector",
      top_k: 5,
      similarity_threshold: 0.4,
      rerank_model: "",
      vector_weight: 0.7,
    },
    metadata_configs: [] as MetadataConfig[],
  };

  const [createErrorMsg, setCreateErrorMsg] = useState("");

  const handleCreateSuccess = (kbConfig: KbConfig) => {
    console.log("创建知识库成功", kbConfig);
    setCreateErrorMsg("");
    setActiveTab(`/knowledgebase/details/${kbConfig.id}`);
  };

  const handleCancel = () => {
    console.log("取消创建知识库。");
    setCreateErrorMsg("");
    setActiveTab(`/knowledgebase`);
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-none">
        <div className="p-2 space-y-2">
          <div className="mb-2 flex items-center gap-2">
            {/* 面包屑导航 */}
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
                  <BreadcrumbPage>新建知识库</BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
          <div className="mb-2 flex items-center gap-2">
            <Button
              variant="outline"
              className="h-8 w-8"
              onClick={() => setActiveTab("/knowledgebase")}
            >
              <ArrowLeft />
            </Button>
            <h1 className="text-2xl font-bold">新建知识库</h1>
          </div>
        </div>
      </div>
      {/* 可滚动内容区域 */}
      <div className="flex-1 overflow-y-auto">
        <KbConfigCard
          kbConfig={kbConfig}
          isCreate={true}
          onSaveSuccess={handleCreateSuccess}
          onCancel={handleCancel}
        />
      </div>
    </div>
  );
}
