'use client';

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import LlmConfigPage from '@/app/config/model/llm/page';
import EmbConfigPage from '@/app/config/model/embedding/page';
import RerankerConfigPage from '@/app/config/model/reranker/page';

export default function ModelConfigPage() {
  return (
    <div id="model">
      <div className="flex flex-col h-screen p-2">
        {/* 顶部标题栏 */}
        <div className="flex items-center h-1/10">
          <h1 className="text-xl font-medium">模型</h1>
        </div>

        {/* 卡片容器 */}
        <div className="flex-1 overflow-y-auto">
          <div className="">
            <Tabs defaultValue="llms">
              <TabsList className="py-4 bg-muted rounded-lg flex-none">
                <TabsTrigger value="llms" className="p-4">
                  LLM
                </TabsTrigger>
                <TabsTrigger value="embeddings" className="p-4">
                  Embedding
                </TabsTrigger>
                <TabsTrigger value="rerankers" className="p-4">
                  Reranker
                </TabsTrigger>
              </TabsList>
              <TabsContent value="llms" className="py-4">
                <LlmConfigPage />
              </TabsContent>
              <TabsContent value="embeddings" className="py-4">
                <EmbConfigPage />
              </TabsContent>
              <TabsContent value="rerankers" className="py-4">
                <RerankerConfigPage />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
