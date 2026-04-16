'use client';
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { KbConfig, KbConfigCard } from '../kbconfig';
import { useRouter } from 'next/navigation';
import { useI18n } from '@/app/providers/i18n';
import { HeaderPortal } from '@/components/header-portal';

export default function KnowledgeBaseCreatePage() {
  const { t } = useI18n();

  const kbConfig: KbConfig = {
    id: '',
    name: '',
    description: '',
    chunk_config: {
      parser_type: 'structure',
      separator: '\n\n',
      chunk_size: '1000',
      chunk_overlap: '50',
      image_caption_model: undefined,
    },
    embedding_model: 'BAAI/bge-m3',
    retrieval_config: {
      retrieval_mode: 'vector',
      top_k: 5,
      similarity_threshold: 0.2,
      enable_rerank: false,
      rerank_model: '',
      vector_weight: 0.7,
    },
  };

  const router = useRouter();

  const handleCreateSuccess = (created: KbConfig) => {
    router.push(`/knowledgebases/${created.id}`);
  };

  const handleCancel = () => {
    router.push('/knowledgebases');
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <HeaderPortal>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Button
                  variant="link"
                  className="px-0 h-auto"
                  onClick={() => router.push('/knowledgebases')}
                >
                  {t('knowledgebase.title')}
                </Button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage className="font-semibold">
                {t('knowledgebase.createPageTitle')}
              </BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </HeaderPortal>
      <div className="flex-1 min-h-0">
        <div className="max-w-5xl mx-auto h-full flex flex-col">
          <KbConfigCard
            kbConfig={kbConfig}
            isCreate={true}
            onSaveSuccess={handleCreateSuccess}
            onCancel={handleCancel}
          />
        </div>
      </div>
    </div>
  );
}
