'use client';

import { useRouter } from "next/navigation";
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

interface UseDatasetActionsProps {
  datasetId: string;
}

interface ExperimentData {
  name: string;
  description: string;
  sample_ids: string[];
  run_config_id: string;
  evaluator_config_id: string;
}

export function useDatasetActions({ datasetId }: UseDatasetActionsProps) {
  const router = useRouter();
  const { tenantFetch } = useTenantFetch();
  // 运行样本（单条或批量）
  const runSamples = async (data: ExperimentData) => {
    try {
      const res = await tenantFetch(`/api/config/evaluation/${datasetId}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!res.ok) throw new Error('实验创建失败');

      const result = await res.json();
      toast.success('实验创建成功');
      router.push(`/evaluation/${datasetId}/experiments/${result.data.id}`);
      return result.data.id;
    } catch (error) {
      console.error('实验创建失败:', error);
      toast.error('实验创建失败');
      throw error;
    }
  };

  // 删除样本
  const deleteSample = async (sampleId: string) => {
    try {
      const res = await tenantFetch(`/api/config/evaluation/${datasetId}/samples/${sampleId}`, {
        method: 'DELETE',
      });

      if (!res.ok) throw new Error('删除失败');
      toast.success('删除成功');
      return true;
    } catch (error) {
      console.error('删除失败:', error);
      toast.error('删除失败');
      return false;
    }
  };

  // 上传文件
  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await tenantFetch(`/api/config/evaluation/${datasetId}/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await res.json();
      if (result.code === 200) {
        toast.success('上传成功');
        return result.data;
      }
      else {
        throw new Error(result.message);
      }
    } catch (error: any) {
      console.error('上传失败:', error.message);
      toast.error(`上传失败: ${error.message}`);
      throw error;
    }
  };

  return {
    runSamples,
    deleteSample,
    uploadFile,
  };
}