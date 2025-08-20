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

interface RerankerConfig {
  id: string;
  model_id: string;
  model_name: string;
  api_key: string;
  base_url: string;
}

const newrerankerconfig: RerankerConfig = {
  id: '',
  model_id: '',
  model_name: '',
  api_key: '',
  base_url: '',
};
export default function RerankerConfigPage() {
  const [editRerankerConfig, setEditRerankerConfig] =
    useState<RerankerConfig>(newrerankerconfig); // 存储 Reranker 配置
  const [rerankerconfigs, setRerankerConfigs] = useState<RerankerConfig[]>([]); // 存储 Reranker 配置
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
          `${process.env.NEXT_PUBLIC_BACKEND_URL ?? ''}/v1/config/rerankers?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error('获取Reranker模型列表失败');
        const json_data = await res.json();
        const data = json_data.data.items;
        setRerankerConfigs(data); // 合并
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || '加载失败');
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, [page, rerankerconfigs.length]);

  const handleCreateSuccess = (llmConfig: RerankerConfig) => {
    setRerankerConfigs((prev) => [...prev, llmConfig]); // 追加新 Reranker 配置
    console.log('创建Reranker成功', llmConfig);
    setEditRerankerConfig(newrerankerconfig);
  };

  const handleSaveSuccess = (llmConfig: RerankerConfig) => {
    setRerankerConfigs((prev) =>
      prev.map((config) => (config.id === llmConfig.id ? llmConfig : config)),
    );
    console.log('编辑Reranker成功', llmConfig);
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
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL ?? ''}/v1/config/${model_type}/${id}`, {
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
      if (model_type === 'rerankers') {
        setRerankerConfigs((prev) => prev.filter((config) => config.id !== id));
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
            setEditRerankerConfig(newrerankerconfig);
          }}
        >
          添加Reranker模型
        </Button>
        <RerankerModelDialog
          isAdd={isCreateOpen ? true : false}
          isOpen={isEditOpen || isCreateOpen}
          setIsOpen={(open: boolean) => {
            if (!open) {
              setEditRerankerConfig(newrerankerconfig); // 关闭时清空编辑数据
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
                className="flex flex-col border rounded-lg shadow-sm h-full"
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
                  <p className="truncate text-muted-foreground py-4">
                    {reranker.base_url}
                  </p>
                </CardContent>
                <CardFooter className="mt-auto pt-0 flex justify-end">
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
