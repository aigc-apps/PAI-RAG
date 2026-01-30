'use client';

import React, { useState, useEffect } from 'react';
import {
  TrashIcon,
  Edit,
  AlertCircleIcon,
  Loader2,
  CheckCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { EmbeddingModelDialog, EmbConfig } from '@/app/config/model/embedding/modelDialog';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

const newembconfig: EmbConfig = {
  id: '',
  model_id: '',
  model_name: '',
  type: '',
  api_key: '',
  endpoint: '',
  dimension: 0,
  embed_batch_size: 0,
  is_ready: false,
  is_default: false,
};
export default function EmbConfigPage() {
  const { t } = useI18n();

  const [editEmbConfig, setEditEmbConfig] = useState<EmbConfig>(newembconfig);
  const [embconfigs, setEmbConfigs] = useState<EmbConfig[]>([]);
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
          `/api/config/embeddings?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error(t('config.model.fetchModelListFailed'));
        const json_data = await res.json();
        const data = json_data.data.items;
        setEmbConfigs(data);
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || t('config.model.loadFailed'));
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, [page, embconfigs.length, isEditOpen]);

  const handleCreateSuccess = (llmConfig: EmbConfig) => {
    setEmbConfigs((prev) => [...prev, llmConfig]);
    console.log(t('config.model.createModelSuccess'), llmConfig);
    setEditEmbConfig(newembconfig);
  };

  const handleSaveSuccess = (llmConfig: EmbConfig) => {
    setEmbConfigs((prev) =>
      prev.map((config) => (config.id === llmConfig.id ? llmConfig : config)),
    );
    console.log(t('config.model.editModelSuccess'), llmConfig);
    setEditEmbConfig(newembconfig);
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
      if (model_type === 'embeddings') {
        setEmbConfigs((prev) => prev.filter((config) => config.id !== id));
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
            setEditEmbConfig(newembconfig);
          }}
        >
          {t('config.model.addEmbeddingModel')}
        </Button>
        <EmbeddingModelDialog
          isAdd={isCreateOpen ? true : false}
          isOpen={isEditOpen || isCreateOpen}
          setIsOpen={(open: boolean) => {
            if (!open) {
              setEditEmbConfig(newembconfig);
            }
            setIsEditOpen(open);
            setIsCreateOpen(open);
          }}
          embConfig={editEmbConfig || newembconfig}
          onSaveSuccess={(emb: EmbConfig) => {
            if (isCreateOpen) {
              handleCreateSuccess(emb);
            } else {
              handleSaveSuccess(emb);
            }
          }}
        />
      </div>
      {embconfigs.length > 0 ? (
        <div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
            {embconfigs.map((emb) => (
              <Card
                key={emb.id}
                className="flex flex-col border rounded-lg shadow-sm h-full pt-4 pb-2"
              >
                <CardHeader>
                  <CardTitle className="text-sm font-medium">
                    <div className="flex items-center gap-3 flex-wrap">
                      {emb.is_default && (
                        <Badge className="bg-red-100 text-red-800">{t('config.model.default')}</Badge>
                      )}
                      <Badge className="bg-yellow-100 text-yellow-800">
                        {emb.model_name}
                      </Badge>
                      <Badge className="bg-blue-100 text-blue-800">
                        {emb.type}
                      </Badge>
                      {emb.type === 'local' ? (
                        <Badge
                          className={
                            emb.is_ready
                              ? 'bg-green-100 text-green-800'
                              : 'bg-gray-100 text-gray-800'
                          }
                        >
                          {emb.is_ready ? (
                            <span className="inline-flex items-center">
                              {' '}
                              {t('config.model.available')}{' '}
                              <CheckCircle className="h-3 w-3 text-green-500" />{' '}
                            </span>
                          ) : (
                            <span className="inline-flex items-center">
                              {' '}
                              {t('config.model.downloading')}{' '}
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            </span>
                          )}
                        </Badge>
                      ) : (
                        <Badge className="bg-green-100 text-green-800">
                          <span className="inline-flex items-center">
                            {' '}
                            {t('config.model.available')}{' '}
                            <CheckCircle className="h-3 w-3 text-green-500" />{' '}
                          </span>
                        </Badge>
                      )}
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0 pb-0">
                  <p className="truncate">{emb.model_id}</p>
                  <p className="truncate text-xs text-muted-foreground pt-2">
                    {emb.endpoint}
                  </p>
                </CardContent>
                <CardFooter className="mt-auto pt-0 flex justify-end pb-0">
                  <Button
                    variant="link"
                    onClick={() => removeModel(emb.id, 'embeddings')}
                    className="text-sm text-primary text-red-600 hover:text-primary/80 underline-offset-4 hover:underline"
                  >
                    <TrashIcon className="ml-1" size={16} />
                  </Button>

                  <Button
                    variant="link"
                    className="text-sm text-primary text-blue-600 hover:text-primary/80 underline-offset-4 hover:underline"
                    onClick={() => {
                      setEditEmbConfig(emb);
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
