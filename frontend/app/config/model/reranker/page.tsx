'use client';

import React, { useState, useEffect } from 'react';
import { TrashIcon, Edit, AlertCircleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { RerankerModelDialog } from '@/app/config/model/reranker/modelDialog';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

interface RerankerConfig {
  id: string;
  model_id: string;
  model_name: string;
  api_key: string;
  base_url: string;
  type?: string;
}

const newrerankerconfig: RerankerConfig = {
  id: '',
  model_id: '',
  model_name: '',
  api_key: '',
  base_url: '',
  type: 'OpenAICompatible',
};
export default function RerankerConfigPage() {
  const { t } = useI18n();
  const [editRerankerConfig, setEditRerankerConfig] =
    useState<RerankerConfig>(newrerankerconfig);
  const [rerankerconfigs, setRerankerConfigs] = useState<RerankerConfig[]>([]);
  const [modelloading, setModelLoading] = useState(true);
  const [modelerror, setModelError] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const modelSizePerPage = 8;

  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);

  const { tenantFetch } = useTenantFetch();
  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/rerankers?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error(t('config.model.fetchRerankerListFailed'));
        const json_data = await res.json();
        // Convert backend type values to frontend format
        const reverseTypeMapping: Record<string, string> = {
          'openai_like': 'OpenAICompatible',
          'dashscope': 'DashScope',
        };
        const data = json_data.data.items.map((item: RerankerConfig) => ({
          ...item,
          type: item.type ? (reverseTypeMapping[item.type] || item.type) : 'OpenAICompatible',
        }));
        setRerankerConfigs(data);
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || t('config.model.loadFailed'));
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, [page, rerankerconfigs.length]);

  const handleCreateSuccess = (llmConfig: RerankerConfig) => {
    setRerankerConfigs((prev) => [...prev, llmConfig]);
    console.log(t('config.model.createRerankerSuccess'), llmConfig);
    setEditRerankerConfig(newrerankerconfig);
  };

  const handleSaveSuccess = (llmConfig: RerankerConfig) => {
    setRerankerConfigs((prev) =>
      prev.map((config) => (config.id === llmConfig.id ? llmConfig : config)),
    );
    console.log(t('config.model.editRerankerSuccess'), llmConfig);
    setEditRerankerConfig(newrerankerconfig);
  };

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const removeModel = async (id: string, model_type: string) => {
    setErrorMsg('');
    try {
      console.log('removeModel: id: ', id, 'model_type: ', model_type);
      const res = await tenantFetch(`/api/config/${model_type}/${id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        setErrorMsg(t('config.model.deleteModelFailed', { modelType: model_type }));
        return;
      }

      // Delete success, update local state
      if (model_type === 'rerankers') {
        setRerankerConfigs((prev) => prev.filter((config) => config.id !== id));
      }
    } catch (err: any) {
      setErrorMsg(t('config.model.deleteFailed'));
    }
  };

  return (
    <div id="llm">
      <div className="grid grid-cols-1 pb-8">
        <Button
          onClick={() => {
            setIsCreateOpen(true);
            setEditRerankerConfig(newrerankerconfig);
          }}
        >
          {t('config.model.addRerankerModel')}
        </Button>
        <RerankerModelDialog
          isAdd={isCreateOpen ? true : false}
          isOpen={isEditOpen || isCreateOpen}
          setIsOpen={(open: boolean) => {
            if (!open) {
              setEditRerankerConfig(newrerankerconfig);
            }
            setIsEditOpen(open);
            setIsCreateOpen(open);
          }}
          rerankerConfig={editRerankerConfig || newrerankerconfig}
          onSaveSuccess={(reranker: RerankerConfig) => {
            if (isCreateOpen) {
              handleCreateSuccess(reranker);
            } else {
              handleSaveSuccess(reranker);
            }
          }}
        />
      </div>
      {rerankerconfigs.length > 0 ? (
        <div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
            {rerankerconfigs.map((reranker) => (
              <Card
                key={reranker.id}
                className="flex flex-col border rounded-lg shadow-sm h-full pt-2 pb-2"
              >
                <CardHeader>
                  <CardTitle className="text-sm font-medium">
                    <div className="flex items-center gap-3 flex-wrap">
                      <Badge className="bg-red-100 text-red-800">
                        {reranker.model_name}
                      </Badge>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="truncate">{reranker.model_id}</p>
                  <p className="truncate text-muted-foreground pt-2 text-xs">
                    {reranker.base_url}
                  </p>
                </CardContent>
                <CardFooter className="mt-auto pt-0 flex justify-end pb-0">
                  <Button
                    variant="link"
                    onClick={() => removeModel(reranker.id, 'rerankers')}
                    className="text-sm text-primary text-red-600 hover:text-primary/80 underline-offset-4 hover:underline"
                  >
                    <TrashIcon className="ml-1" size={16} />
                  </Button>

                  <Button
                    variant="link"
                    className="text-sm text-primary text-blue-600 hover:text-primary/80 underline-offset-4 hover:underline"
                    onClick={() => {
                      setEditRerankerConfig(reranker);
                      setIsEditOpen(true);
                    }}
                  >
                    <Edit className="ml-1" size={16} />
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
          <div className="flex justify-center items-center h-1/10 py-6">
            <PaginationComponent
              currentPage={page}
              totalPages={totalPages}
              onPageChange={handlePageChange}
            />
          </div>
        </div>
      ) : (
        <div className="flex justify-center items-center h-1/10 py-6">
          <h3 className="text-lg font-medium text-gray-700 py-6">
            {t('config.model.noModelsYet')}
          </h3>
        </div>
      )}
      <div className="block w-full">
        {errorMsg !== '' && (
          <Alert variant="destructive">
            <AlertCircleIcon />
            <AlertDescription>
              <p>{errorMsg}</p>
            </AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
}
