'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { toast } from 'sonner';
import { Slider } from '@/components/ui/slider';
import { ExternalLink } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

const ENDPOINT_LIST = [
  'iqs.cn-zhangjiakou.aliyuncs.com',
  'iqs-vpc.cn-beijing.aliyuncs.com',
  'iqs-vpc.cn-zhangjiakou.aliyuncs.com',
  'iqs-vpc.cn-shanghai.aliyuncs.com',
  'iqs-vpc.cn-wulanchabu.aliyuncs.com',
  'iqs-vpc.cn-chengdu.aliyuncs.com',
  'iqs-vpc.cn-guangzhou.aliyuncs.com',
  'iqs-vpc.cn-shenzhen.aliyuncs.com',
  'iqs-vpc.cn-hangzhou.aliyuncs.com',
];

const MASK_API_KEY = '******';

export default function SearchConfig() {
  const { t } = useI18n();
  const [aliyunHasKey, setAliyunHasKey] = useState(false);
  const [tavilyHasKey, setTavilyHasKey] = useState(false);

  const [aliyunAK, setAliyunAK] = useState('');
  const [aliyunSK, setAliyunSK] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [searchCount, setSearchCount] = useState(10);
  const [tavilyApiKey, setTavilyApiKey] = useState('');
  const [searchEngineType, setSearchEngineType] = useState('aliyun');
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await tenantFetch(`/api/config/websearch`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        if (!res.ok) throw new Error(t('config.loadError'));
        const data = (await res.json()).data[0];
        setAliyunHasKey(!data.is_aliyun_empty);
        setTavilyHasKey(!data.is_tavily_empty);
        setAliyunAK(MASK_API_KEY);
        setAliyunSK(MASK_API_KEY);
        setEndpoint(data?.endpoint || '');
        setSearchCount(data?.search_count || 10);
        setSearchEngineType(data?.type || 'aliyun');
        setTavilyApiKey(MASK_API_KEY);
      } catch (err: any) {
        toast.error(err.message);
      }
    };
    fetchConfig();
  }, []);

  const handleSave = async () => {
    try {
      setIsSaving(true);
      const update_ak = aliyunAK === MASK_API_KEY ? '' : aliyunAK;
      const update_sk = aliyunSK === MASK_API_KEY ? '' : aliyunSK;
      const update_tavily_api_key = tavilyApiKey === MASK_API_KEY ? '' : tavilyApiKey;

      const res = await tenantFetch(`/api/config/websearch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          access_key_id: update_ak,
          access_key_secret: update_sk,
          type: searchEngineType,
          endpoint: endpoint,
          tavily_api_key: update_tavily_api_key,
          search_count: searchCount,
        }),
      });

      if (!res.ok) throw new Error(t('config.search.saveFailed'));
      toast.success(t('config.search.saveSuccess'));
    } catch (err: any) {
      toast.warning(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="search" className="settings-page">
      <div className="settings-page-header">
        <h1 className="page-title">{t('config.search.title')}</h1>
        <Button size="sm" onClick={handleSave} disabled={isSaving}>
          {isSaving ? t('config.search.saving') : t('config.search.saveSearchConfig')}
        </Button>
      </div>

      <div className="settings-page-content">
        <div className="space-y-6">
          {/* Engine selector */}
          <div>
            <div className="dialog-section-title mb-2">搜索引擎</div>
            <div className="flex items-center gap-3">
              <Select value={searchEngineType} onValueChange={setSearchEngineType}>
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder={t('config.search.selectSearchEnginePlaceholder')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="aliyun">{t('config.search.aliyunUniversalSearch')}</SelectItem>
                  <SelectItem value="tavily">{t('config.search.tavilySearch')}</SelectItem>
                </SelectContent>
              </Select>
              <a
                href={
                  searchEngineType === 'aliyun'
                    ? 'https://help.aliyun.com/document_detail/2870227.html'
                    : 'https://www.tavily.com/'
                }
                target="_blank"
                rel="noreferrer"
                className="text-xs text-primary hover:underline inline-flex items-center gap-1 whitespace-nowrap"
              >
                {t('config.search.activationGuide')}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {searchEngineType === 'aliyun' && (
            <div>
              <div className="dialog-section-title mb-2">Aliyun IQS 凭证</div>
              <div className="space-y-4">
                <div>
                  <label className="form-label">
                    {t('config.search.universalSearchEndpoint')}
                  </label>
                  <Select value={endpoint} onValueChange={setEndpoint}>
                    <SelectTrigger>
                      <SelectValue placeholder={t('config.search.selectRegionPlaceholder')} />
                    </SelectTrigger>
                    <SelectContent>
                      {ENDPOINT_LIST.map((ep) => (
                        <SelectItem key={ep} value={ep}>
                          {ep}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label htmlFor="aliyun_ak" className="form-label">
                      {t('config.search.accessKeyId')}
                    </label>
                    <Input
                      id="aliyun_ak"
                      defaultValue={aliyunHasKey ? '******' : ''}
                      type="password"
                      onChange={(e) => setAliyunAK(e.target.value)}
                      placeholder={t('config.search.accessKeyIdPlaceholder')}
                    />
                  </div>
                  <div>
                    <label htmlFor="aliyun_sk" className="form-label">
                      {t('config.search.accessKeySecret')}
                    </label>
                    <Input
                      id="aliyun_sk"
                      defaultValue={aliyunHasKey ? '******' : ''}
                      type="password"
                      onChange={(e) => setAliyunSK(e.target.value)}
                      placeholder={t('config.search.accessKeySecretPlaceholder')}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {searchEngineType === 'tavily' && (
            <div>
              <div className="dialog-section-title mb-2">Tavily 凭证</div>
              <div>
                <label htmlFor="tavily_key" className="form-label">
                  {t('config.search.tavilyApiKey')}
                </label>
                <Input
                  id="tavily_key"
                  defaultValue={tavilyHasKey ? '******' : ''}
                  type="password"
                  onChange={(e) => setTavilyApiKey(e.target.value)}
                  placeholder={t('config.search.tavilyApiKeyPlaceholder')}
                />
              </div>
            </div>
          )}

          <div>
            <div className="dialog-section-title mb-2">搜索参数</div>
            <div className="form-row-inline">
              <div className="form-row-inline-label">
                <span className="title">
                  {t('config.search.searchResultCount', { count: searchCount })}
                </span>
                <span className="hint">每次搜索返回的最多结果数 (1–20)</span>
              </div>
              <div className="w-48">
                <Slider
                  value={[searchCount]}
                  max={20}
                  min={1}
                  step={1}
                  onValueChange={(v: number[]) => setSearchCount(v[0])}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
