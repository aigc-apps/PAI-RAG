// hooks/useTracingConfig.ts
import { useState, useEffect } from 'react';

interface TracingConfig {
  endpoint: string;
  token: string;
  service_name: string;
  enabled: boolean;
  region?: string;
}

export const useTracingConfig = () => {
  const [config, setConfig] = useState<TracingConfig | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  // ref: https://help.aliyun.com/zh/arms/tracing-analysis/before-you-begin
  const ENDPOINT_TO_REGION_MAP: Record<string, string> = {
    'dc-hz': 'cn-hangzhou',           // 华东1（杭州）
    'dc-sh': 'cn-shanghai',           // 华东2（上海）
    'dc-qd': 'cn-qingdao',            // 华北1（青岛）
    'dc-bj': 'cn-beijing',            // 华北2（北京）
    'dc-zb': 'cn-zhangjiakou',        // 华北3（张家口）
    'cn-huhehaote': 'cn-huhehaote',   // 华北5（呼和浩特）——直接用 region ID
    'cn-wulanchabu': 'cn-wulanchabu', // 华北6（乌兰察布）
    'dc-sz': 'cn-shenzhen',           // 华南1（深圳）
    'cn-heyuan': 'cn-heyuan',         // 华南2（河源）
    'cn-guangzhou': 'cn-guangzhou',   // 华南3（广州）
    'cn-chengdu': 'cn-chengdu',       // 西南1（成都）
    'dc-hk': 'cn-hongkong',           // 中国（香港）
    'dc-jp': 'ap-northeast-1',        // 日本（东京）
    'dc-sg': 'ap-southeast-1',        // 新加坡
    'ap-southeast-3': 'ap-southeast-3', // 马来西亚（吉隆坡）
    'dc-indonesia': 'ap-southeast-5', // 印尼（雅加达）
    'dc-frankfurt': 'eu-central-1',   // 德国（法兰克福）
    'dc-lundun': 'eu-west-1',         // 英国（伦敦）
    'dc-usw': 'us-west-1',            // 美国（硅谷）
    'us-east-1': 'us-east-1',         // 美国（弗吉尼亚）
  };

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        const res = await fetch(`/api/config/trace`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error('获取链路追踪配置失败');

        const data = await res.json();
        setConfig(data);
        let region = 'cn-hangzhou';
        for (const [key, mappedRegion] of Object.entries(ENDPOINT_TO_REGION_MAP)) {
          if (data.endpoint?.includes(key)) {
            region = mappedRegion;
            break;
          }
        }
        data.region = region;
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  return { config, loading, error };
};