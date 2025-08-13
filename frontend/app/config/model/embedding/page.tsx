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
import { EmbeddingModelDialog } from '@/app/config/model/embedding/modelDialog';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface EmbConfig {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
  api_key: string;
  endpoint: string;
  dimension: number;
  embed_batch_size: number;
  is_ready: boolean;
  is_default: boolean;
}

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
  const [editEmbConfig, setEditEmbConfig] = useState<EmbConfig>(newembconfig); // 存储 Embedding 配置
  const [embconfigs, setEmbConfigs] = useState<EmbConfig[]>([]); // 存储 Embedding 配置
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(''); // 错误信息
  const [errorMsg, setErrorMsg] = useState(''); // 删除时的错误信息

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const modelSizePerPage = 8;

  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await fetch(
          `/v1/config/embeddings?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error('获取Embedding模型列表失败');
        const json_data = await res.json();
        const data = json_data.data.items;
        setEmbConfigs(data); // 合并
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || '加载失败');
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, [page, embconfigs.length, isEditOpen]);

  const handleCreateSuccess = (llmConfig: EmbConfig) => {
    setEmbConfigs((prev) => [...prev, llmConfig]); // 追加新 Embedding 配置
    console.log('创建Embedding成功', llmConfig);
    setEditEmbConfig(newembconfig);
  };

  const handleSaveSuccess = (llmConfig: EmbConfig) => {
    setEmbConfigs((prev) =>
      prev.map((config) => (config.id === llmConfig.id ? llmConfig : config)),
    );
    console.log('编辑Embedding成功', llmConfig);
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

      const res = await fetch(`/v1/config/${model_type}/${id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!res.ok) {
        setErrorMsg(`${model_type}删除失败，请检查网络或配置`);
        return;
      }

      // 删除成功后更新本地状态
      if (model_type === 'embeddings') {
        setEmbConfigs((prev) => prev.filter((config) => config.id !== id));
      }
    } catch (err: any) {
      setErrorMsg('删除失败，请检查网络或配置');
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
          添加Embedding模型
        </Button>
        <EmbeddingModelDialog
          isAdd={isCreateOpen ? true : false}
          isOpen={isEditOpen || isCreateOpen}
          setIsOpen={(open: boolean) => {
            if (!open) {
              setEditEmbConfig(newembconfig); // 关闭时清空编辑数据
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
                className="flex flex-col border rounded-lg shadow-sm h-full"
              >
                <CardHeader>
                  <CardTitle className="text-sm font-medium">
                    <div className="flex items-center gap-3 flex-wrap">
                      {emb.is_default && (
                        <Badge className="bg-red-100 text-red-800">默认</Badge>
                      )}
                      <Badge className="bg-yellow-100 text-yellow-800">
                        {emb.model_name}
                      </Badge>
                      {/* <Badge className="bg-yellow-100 text-yellow-800">
                        {String(emb.dimension)}
                      </Badge> */}
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
                              可用{' '}
                              <CheckCircle className="h-3 w-3 text-green-500" />{' '}
                            </span>
                          ) : (
                            <span className="inline-flex items-center">
                              {' '}
                              下载中{' '}
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            </span>
                          )}
                        </Badge>
                      ) : (
                        <Badge className="bg-green-100 text-green-800">
                          <span className="inline-flex items-center">
                            {' '}
                            可用{' '}
                            <CheckCircle className="h-3 w-3 text-green-500" />{' '}
                          </span>
                        </Badge>
                      )}
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="truncate">{emb.model_id}</p>
                  <p className="truncate text-muted-foreground py-4">
                    {emb.endpoint}
                  </p>
                </CardContent>
                <CardFooter className="mt-auto pt-0 flex justify-end">
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
            暂无模型，请点击上方按钮添加
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
