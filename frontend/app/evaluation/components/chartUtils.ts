// chartUtils.ts
import { ExperimentDetailsItem } from '@/app/evaluation/[evalId]/types';

export type StatusKey = 'success' | 'failed' | 'running' | 'pending';

export const getScoreDistributionData = (allExpItems: ExperimentDetailsItem[]) => {
  const ranges = [
    { range: "0-0.2", min: 0, max: 0.2, count: 0 },
    { range: "0.2-0.4", min: 0.2, max: 0.4, count: 0 },
    { range: "0.4-0.6", min: 0.4, max: 0.6, count: 0 },
    { range: "0.6-0.8", min: 0.6, max: 0.8, count: 0 },
    { range: "0.8-1.0", min: 0.8, max: 1.0, count: 0 },
  ];
  for (const item of allExpItems) {
    if (item.status === 'pending' || item.status === 'running') continue;
    if (item.score == null || typeof item.score !== 'number') continue;
    const range = ranges.find(r => item.score >= r.min && item.score <= r.max);
    if (range) {
      range.count++;
    }
  }
  return ranges.map(r => ({ range: r.range, count: r.count }));
};

export const getStatusDistributionData = (allExpItems: ExperimentDetailsItem[]) => {
  const statusCount: Record<StatusKey, number> = {
    success: 0,
    failed: 0,
    running: 0,
    pending: 0,
  };
  allExpItems.forEach((item) => {
    if ((item.status as StatusKey) in statusCount) {
      statusCount[item.status as StatusKey]++;
    }
  });
  return Object.entries(statusCount)
    .filter(([_, count]) => count > 0)
    .map(([name, value]) => ({ name, value }));
};

export const STATUS_COLORS: Record<StatusKey, string> = {
  success: "#10b981", // green-500
  failed: "#ef4444",  // red-500
  running: "#3b82f6", // blue-500
  pending: "#f59e0b", // amber-500
};

export const hasTimingData = (allExpItems: ExperimentDetailsItem[]) => {
  return allExpItems.some(item => item.started_at && item.updated_at);
};

export const getAverageTime = (allExpItems: ExperimentDetailsItem[]) => {
  const formatUTC = (str: string | null) =>
    str ? str.replace(' ', 'T').replace(/\.\d+$/, '') + 'Z' : '';
  const validDurations = allExpItems
    .filter(item => item.status !== 'pending' && item.status !== 'running')
    .filter(item => item.started_at && item.updated_at)
    .map(item => {
      const start = new Date(formatUTC(item.started_at)).getTime();
      const end = new Date(formatUTC(item.updated_at)).getTime();
      const duration = (end - start) / 1000; // 转为秒
      return isFinite(duration) && duration > 0 ? duration : null;
    })
    .filter((duration): duration is number => duration !== null);
  if (validDurations.length === 0) return "N/A";
  const avg = validDurations.reduce((sum, time) => sum + time, 0) / validDurations.length;
  return avg.toFixed(2);
};

export const getMinTime = (allExpItems: ExperimentDetailsItem[]) => {
  const formatUTC = (str: string | null) =>
    str ? str.replace(' ', 'T').replace(/\.\d+$/, '') + 'Z' : '';
  const validDurations = allExpItems
    .filter(item => item.status !== 'pending' && item.status !== 'running')
    .filter(item => item.started_at && item.updated_at)
    .map(item => {
      const start = new Date(formatUTC(item.started_at)).getTime();
      const end = new Date(formatUTC(item.updated_at)).getTime();
      const duration = (end - start) / 1000; // 秒
      return isFinite(duration) && duration > 0 ? duration : null;
    })
    .filter((duration): duration is number => duration !== null);
  if (validDurations.length === 0) return "N/A";
  return Math.min(...validDurations).toFixed(2);
};

export const getMaxTime = (allExpItems: ExperimentDetailsItem[]) => {
  const formatUTC = (str: string | null) =>
    str ? str.replace(' ', 'T').replace(/\.\d+$/, '') + 'Z' : '';
  const times = allExpItems
    .filter(item => item.started_at && item.updated_at)
    .map(item => {
      const start = new Date(formatUTC(item.started_at)).getTime();
      const end = new Date(formatUTC(item.updated_at)).getTime();
      return (end - start) / 1000;
    });
  if (times.length === 0) return "N/A";
  return Math.max(...times).toFixed(2);
};

export const getAverageScore = (allExpItems: ExperimentDetailsItem[]) => {
  const validScores = allExpItems
    .filter(item => item.status !== 'pending' && item.status !== 'running')
    .map(item => item.score)
    .filter(score => typeof score === 'number');
  if (validScores.length === 0) return "N/A";
  const avg = validScores.reduce((sum, score) => sum + score, 0) / validScores.length;
  return avg.toFixed(2);
};