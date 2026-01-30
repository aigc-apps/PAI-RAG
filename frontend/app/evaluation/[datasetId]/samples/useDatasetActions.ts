'use client';

import { useRouter } from "next/navigation";
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

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
  const { t } = useI18n();
  
  // Run samples (single or batch)
  const runSamples = async (data: ExperimentData) => {
    try {
      const res = await tenantFetch(`/api/config/evaluation/${datasetId}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!res.ok) throw new Error(t('evaluation.experimentCreateFailed'));

      const result = await res.json();
      toast.success(t('evaluation.experimentCreateSuccess'));
      router.push(`/evaluation/${datasetId}/experiments/${result.data.id}`);
      return result.data.id;
    } catch (error) {
      console.error('Failed to create experiment:', error);
      toast.error(t('evaluation.experimentCreateFailed'));
      throw error;
    }
  };

  // Delete sample
  const deleteSample = async (sampleId: string) => {
    try {
      const res = await tenantFetch(`/api/config/evaluation/${datasetId}/samples/${sampleId}`, {
        method: 'DELETE',
      });

      if (!res.ok) throw new Error(t('common.deleteFailed'));
      toast.success(t('evaluation.deleteSuccess'));
      return true;
    } catch (error) {
      console.error('Failed to delete:', error);
      toast.error(t('common.deleteFailed'));
      return false;
    }
  };

  // Upload file
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
        toast.success(t('evaluation.uploadSuccess'));
        return result.data;
      }
      else {
        throw new Error(result.message);
      }
    } catch (error: any) {
      console.error('Failed to upload:', error.message);
      toast.error(t('evaluation.uploadFailedWithMsg', { msg: error.message }));
      throw error;
    }
  };

  return {
    runSamples,
    deleteSample,
    uploadFile,
  };
}