// app/evaluation/[datasetId]/experiments/useExperiments.ts

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { toast } from 'sonner';
import { ExperimentItem } from '@/app/evaluation/[datasetId]/types';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';


interface UseExperimentsProps {
  datasetId: string;
  page: number;
  pageSize: number;
}

export function useExperiments({ datasetId, page, pageSize }: UseExperimentsProps) {
  const [experiments, setExperiments] = useState<ExperimentItem[]>([]);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const isRefreshing = useRef(false);
  const { tenantFetch } = useTenantFetch();
  const { t } = useI18n();
  
  const fetchExperiments = useCallback(async () => {
    if (isRefreshing.current) {
      console.log(t('evaluation.refreshingExperimentList'));
      return;
    }

    console.log(t('evaluation.fetchingExperimentList'));
    isRefreshing.current = true;

    try {
      const url = `/api/config/evaluation/${datasetId}/experiments?page=${page}&size=${pageSize}`;
      const response = await tenantFetch(url);

      if (!response.ok) throw new Error(t('evaluation.fetchExperimentListFailed'));

      const data = await response.json();
      setExperiments(data.data.items || []);
      setTotalPages(data.data.pages);

      // Check if there are any unfinished experiments
      const hasUnfinished = data.data.items.some(
        (item: ExperimentItem) => !['success', 'failed'].includes(item.status)
      );

      if (hasUnfinished) {
        console.log(t('evaluation.hasUnfinishedExperiments'));
        setTimeout(() => {
          isRefreshing.current = false;
          fetchExperiments();
        }, 3000);
      } else {
        console.log(t('evaluation.allExperimentsCompleted'));
        isRefreshing.current = false;
      }
    } catch (err: any) {
      console.error('Failed to fetch experiment list:', err);
      toast.error(err.message || t('evaluation.loadExperimentListFailed'));
      isRefreshing.current = false;
    } finally {
      setIsLoading(false);
    }
  }, [datasetId, page, pageSize, t, tenantFetch]);

  // First load + refresh when page changes
  useEffect(() => {
    setIsLoading(true);
    fetchExperiments();
  }, [fetchExperiments]);

  // Delete experiment
  const deleteExperiment = async (id: string) => {
    try {
      const response = await tenantFetch(`/api/config/evaluation/${datasetId}/experiments/${id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) throw new Error(t('evaluation.deleteExperimentFailed'));

      setExperiments(prev => prev.filter(exp => exp.id !== id));
      toast.success(t('evaluation.experimentDeletedSuccess'));
    } catch (err: any) {
      console.error('Failed to delete experiment:', err);
      toast.error(t('evaluation.deleteExperimentFailed'));
    }
  };

  return {
    experiments,
    totalPages,
    isLoading,
    fetchExperiments,
    deleteExperiment,
  };
}