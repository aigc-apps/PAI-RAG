// app/evaluation/[datasetId]/experiments/useExperiments.ts

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { toast } from 'sonner';
import { ExperimentItem } from '@/app/evaluation/[datasetId]/types';


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

  const fetchExperiments = useCallback(async () => {
    if (isRefreshing.current) {
      console.log("实验列表正在刷新中...");
      return;
    }

    console.log("正在刷新实验列表...");
    isRefreshing.current = true;

    try {
      const url = `/api/config/evaluation/${datasetId}/experiments?page=${page}&size=${pageSize}`;
      const response = await fetch(url);

      if (!response.ok) throw new Error('获取实验列表失败');

      const data = await response.json();
      setExperiments(data.data.items || []);
      setTotalPages(data.data.pages);

      // 检查是否有未完成的实验
      const hasUnfinished = data.data.items.some(
        (item: ExperimentItem) => !['success', 'failed'].includes(item.status)
      );

      if (hasUnfinished) {
        console.log('存在未完成的实验，3秒后自动刷新...');
        setTimeout(() => {
          isRefreshing.current = false;
          fetchExperiments();
        }, 3000);
      } else {
        console.log('所有实验已完成。');
        isRefreshing.current = false;
      }
    } catch (err: any) {
      console.error('获取实验列表失败:', err);
      toast.error(err.message || '加载实验列表失败');
      isRefreshing.current = false;
    } finally {
      setIsLoading(false);
    }
  }, [datasetId, page, pageSize]);

  // 首次加载 + 页码变化时刷新
  useEffect(() => {
    setIsLoading(true);
    fetchExperiments();
  }, [fetchExperiments]);

  // 删除实验
  const deleteExperiment = async (id: string) => {
    try {
      const response = await fetch(`/api/config/evaluation/${datasetId}/experiments/${id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) throw new Error('删除失败');

      setExperiments(prev => prev.filter(exp => exp.id !== id));
      toast.success('实验删除成功');
    } catch (err: any) {
      console.error('删除实验失败:', err);
      toast.error('删除实验失败');
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