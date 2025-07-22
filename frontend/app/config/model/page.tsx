"use client";

import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import LlmConfigPage from "@/app/config/model/llm/page";
import EmbConfigPage from "@/app/config/model/embedding/page";

export default function ModelConfigPage() {
  return (
    <div id="model">
      <div className="flex flex-col h-screen p-6 space-y-6">
        {/* 顶部标题栏 */}
        <div className="flex justify-between items-center h-1/10">
          <h1 className="text-2xl font-bold">模型</h1>
        </div>

        {/* 卡片容器 */}
        <div className="h-4/5">
          <div className="">
            <Tabs defaultValue="llms">
              <TabsList className="py-4 bg-muted rounded-lg flex-none">
                <TabsTrigger value="llms" className="p-4">
                  LLM
                </TabsTrigger>
                <TabsTrigger value="embeddings" className="p-4">
                  Embedding
                </TabsTrigger>
              </TabsList>
              <TabsContent value="llms" className="py-4">
                <LlmConfigPage />
              </TabsContent>
              <TabsContent value="embeddings" className="py-4">
                <EmbConfigPage />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
