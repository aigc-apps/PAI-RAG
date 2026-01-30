'use client';

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import LlmConfigPage from '@/app/config/model/llm/page';
import EmbConfigPage from '@/app/config/model/embedding/page';
import RerankerConfigPage from '@/app/config/model/reranker/page';
import { useI18n } from '@/app/providers/i18n';

export default function ModelConfigPage() {
  const { t } = useI18n();

  return (
    <div id="model">
      <div className="flex flex-col h-screen p-2">
        <div className="flex items-center h-1/10">
          <h1 className="text-xl font-medium">{t('config.model.title')}</h1>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="">
            <Tabs defaultValue="llms">
              <TabsList className="py-4 bg-muted rounded-lg flex-none">
                <TabsTrigger value="llms" className="p-4">
                  {t('config.model.tabLlm')}
                </TabsTrigger>
                <TabsTrigger value="embeddings" className="p-4">
                  {t('config.model.tabEmbedding')}
                </TabsTrigger>
                <TabsTrigger value="rerankers" className="p-4">
                  {t('config.model.tabReranker')}
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
