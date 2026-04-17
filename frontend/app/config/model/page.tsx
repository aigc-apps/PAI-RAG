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
      <div className="space-y-4 pb-6">
        <LlmConfigPage />
        <EmbConfigPage />
        <RerankerConfigPage />
      </div>
    </div>
  );
}
