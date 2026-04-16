'use client';

import React from 'react';
import LlmConfigPage from '@/app/config/model/llm/page';
import EmbConfigPage from '@/app/config/model/embedding/page';
import RerankerConfigPage from '@/app/config/model/reranker/page';
import { useI18n } from '@/app/providers/i18n';

export default function ModelConfigPage() {
  const { t } = useI18n();

  return (
    <div id="model">
      <div className="flex flex-col h-screen px-6 py-4">
        <div className="flex items-center pb-4">
          <h1 className="page-title">{t('config.model.title')}</h1>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="space-y-4 pb-6">
            <LlmConfigPage />
            <EmbConfigPage />
            <RerankerConfigPage />
          </div>
        </div>
      </div>
    </div>
  );
}
