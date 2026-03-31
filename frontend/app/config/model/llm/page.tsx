'use client';

import React, { useState, useEffect } from 'react';
import { TrashIcon, Edit, AlertCircleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LLMModelDialog } from '@/app/config/model/llm/modelDialog';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export interface LlmConfig {
  id: string;
  model_id: string;
  source: string;
  model: string;
  api_key: string;
  base_url: string;
  max_context: number;
  enabled: boolean;
  vision_support: boolean;
  enable_thinking: boolean; // 是否支持思考模式
  temperature: number; // 温度参数
}

const newllmconfig: LlmConfig = {
  id: '',
  model_id: '',
  source: '',
  model: '',
  base_url: '',
  api_key: '',
  vision_support: false,
  max_context: 0,
  enabled: true,
  enable_thinking: false, // 默认支持思考模式
  temperature: 0.1, // 默认温度 0.1
};
export default function LlmConfigPage() {
  const { t } = useI18n();
  const [editLlmConfig, setEditLlmConfig] = useState<LlmConfig>(newllmconfig);
  const [llmconfigs, setLlmConfigs] = useState<LlmConfig[]>([]); // 存储 LLM 配置
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(''); // 错误信息
  const [errorMsg, setErrorMsg] = useState(''); // 删除或更新时的错误信息

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
          `/api/config/llms?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error(t('config.model.fetchLlmListFailed'));
        const json_data = await res.json();
        const data = json_data.data.items;
        setLlmConfigs(data); // 合并
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || t('config.model.loadFailed'));
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, [page, llmconfigs.length]);

  const handleCreateSuccess = (llmConfig: LlmConfig) => {
    setLlmConfigs((prev) => [...prev, llmConfig]); // 追加新 LLM 配置
    console.log('创建LLM成功', llmConfig);
    setEditLlmConfig(newllmconfig);
  };

  const handleSaveSuccess = (llmConfig: LlmConfig) => {
    setLlmConfigs((prev) =>
      prev.map((config) => (config.id === llmConfig.id ? llmConfig : config)),
    );
    console.log('编辑LLM成功', llmConfig);
    setEditLlmConfig(newllmconfig);
  };

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  const handleActivateToggle = async (llm: LlmConfig) => {
    setErrorMsg('');
    llm.enabled = !llm.enabled;
    const url = `/api/config/llms/${llm.id}`;

    const res = await tenantFetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(llm), // 包装为数组
    });

    if (!res.ok) {
      setErrorMsg(t('config.model.updateStatusFailed'));
      return;
    }
    setLlmConfigs((prev) =>
      prev.map((c) => (c.id === llm.id ? { ...c, enabled: llm.enabled } : c)),
    );
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
        setErrorMsg(t('config.model.deleteModelFailed', { modelType: 'LLM' }));
        return;
      }

      // 删除成功后更新本地状态
      if (model_type === 'llms') {
        setLlmConfigs((prev) => prev.filter((config) => config.id !== id));
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
            setEditLlmConfig(newllmconfig);
          }}
        >
          {t('config.model.addLlmModel')}
        </Button>
        <LLMModelDialog
          isAdd={isCreateOpen ? true : false}
          isOpen={isEditOpen || isCreateOpen}
          setIsOpen={(open) => {
            if (!open) {
              setEditLlmConfig(newllmconfig); // 关闭时清空编辑数据
            }
            setIsEditOpen(open);
            setIsCreateOpen(open);
          }}
          llmConfig={editLlmConfig || newllmconfig}
          onSaveSuccess={(llm) => {
            if (isCreateOpen) {
              handleCreateSuccess(llm);
            } else {
              handleSaveSuccess(llm);
            }
          }}
        />
      </div>
      {llmconfigs.length > 0 ? (
        <div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
            {llmconfigs.map((llm) => (
              <Card
                key={llm.id}
                className="flex flex-col border rounded-lg shadow-sm h-full pt-4 pb-0 gap-2"
              >
                <CardHeader>
                  <CardTitle className="text-sm font-medium">
                    <div className="flex items-center gap-3 flex-wrap">
                      <Badge className="bg-blue-100 text-blue-800">
                        {llm.source}
                      </Badge>
                      <Badge className="bg-red-100 text-red-800">
                        {llm.model}
                      </Badge>
                      <Badge
                        className='bg-yellow-100 text-yellow-800'
                      >
                        {t('config.model.languageModel')}
                      </Badge>
                      {
                        llm.vision_support && (
                          <Badge className="bg-yellow-100 text-yellow-800">
                            {t('config.model.visionModel')}
                          </Badge>
                        )
                      }
                      {
                        llm.enable_thinking && (
                          <Badge className="bg-yellow-100 text-yellow-800">
                            {t('config.model.thinkingModel')}
                          </Badge>
                        )
                      }
                      {
                        llm.enabled ? (
                          <Badge className="bg-green-100 text-green-800">
                            {t('config.model.activated')}
                          </Badge>
                        ) : (
                          <Badge className="bg-gray-100 text-gray-800">
                            {t('config.model.deactivated')}
                          </Badge>
                        )
                      }
                      <Switch
                        checked={llm.enabled}
                        className="ml-auto rounded-full transition-color"
                        onCheckedChange={() => handleActivateToggle(llm)}
                      />
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0 pb-0">
                  <p className="truncate">{llm.model_id}</p>
                  <p className="truncate text-muted-foreground py-2 text-xs line-clamp-1">
                    {llm.base_url}
                  </p>
                </CardContent>
                <CardFooter className="mt-auto pt-0 pb-2 gap-4 flex justify-end">
                  <Button
                    variant="link"
                    onClick={() => removeModel(llm.id, 'llms')}
                    className="text-sm text-primary text-red-600 hover:text-primary/80 underline-offset-4 hover:underline"
                  >
                    <TrashIcon className="ml-1" size={16} />
                  </Button>

                  <Button
                    variant="link"
                    className="text-sm text-primary text-blue-600 hover:text-primary/80 underline-offset-4 hover:underline"
                    onClick={() => {
                      setEditLlmConfig(llm);
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
