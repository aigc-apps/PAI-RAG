'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { Slider } from "@/components/ui/slider"
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

const ENDPOINT_LIST = [
  "iqs.cn-zhangjiakou.aliyuncs.com",
  "iqs-vpc.cn-beijing.aliyuncs.com",
  "iqs-vpc.cn-zhangjiakou.aliyuncs.com",
  "iqs-vpc.cn-shanghai.aliyuncs.com",
  "iqs-vpc.cn-wulanchabu.aliyuncs.com",
  "iqs-vpc.cn-chengdu.aliyuncs.com",
  "iqs-vpc.cn-guangzhou.aliyuncs.com",
  "iqs-vpc.cn-shenzhen.aliyuncs.com",
  "iqs-vpc.cn-hangzhou.aliyuncs.com",
]

const MASK_API_KEY = '******'

export default function SearchConfig() {
  const { t } = useI18n();
  const [aliyunHasKey, setAliyunHasKey] = useState(false);
  const [tavilyHasKey, setTavilyHasKey] = useState(false);

  const [aliyunAK, setAliyunAK] = useState('');
  const [aliyunSK, setAliyunSK] = useState('');
  const [endpoint, setEndpoint] = useState('')
  const [isSaving, setIsSaving] = useState(false);
  const [searchCount, setSearchCount] = useState(10);
  const [tavilyApiKey, setTavilyApiKey] = useState('');
  const [searchEngineType, setSearchEngineType] = useState('aliyun');
  const { tenantFetch } = useTenantFetch();

  // Initialize and load configuration
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
        setEndpoint(data?.endpoint || "")
        setSearchCount(data?.search_count || 10)
        setSearchEngineType(data?.type || 'aliyun')
        setTavilyApiKey(MASK_API_KEY)
      } catch (err: any) {
        toast.error(err.message);
      }
    };

    fetchConfig();
  }, []);
  // Save configuration
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
    <div id="search">
      <div
        className={
          'transition-colors rounded-lg p-4 overflow-hidden duration-200'
        }
      >
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          <h2 className="text-xl font-medium text-gray-800">{t('config.search.title')}</h2>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="endpoint" className="text-right">
                {t('config.search.selectSearchEngine')}
              </Label>
               <div className="col-span-3 flex items-center">
                <Select
                  value={searchEngineType}
                  onValueChange={(value) =>
                    setSearchEngineType(value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder={t('config.search.selectSearchEnginePlaceholder')} />
                  </SelectTrigger>
                  <SelectContent>
                      <SelectItem key='aliyun' value='aliyun'>
                        {t('config.search.aliyunUniversalSearch')}
                      </SelectItem>
                      <SelectItem key='tavily' value='tavily'>
                        {t('config.search.tavilySearch')}
                      </SelectItem>
                  </SelectContent>
                </Select>

                <a href={searchEngineType=== 'aliyun' ? "https://help.aliyun.com/document_detail/2870227.html" : "https://www.tavily.com/"} target="_blank" className="ml-4 text-sm text-blue-600 hover:underline">{t('config.search.activationGuide')} </a>
              </div>

            </div>
            { searchEngineType === 'aliyun' && (
              <div className="gap-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="endpoint" className="text-right">
                  {t('config.search.universalSearchEndpoint')}
                </Label>
                <div className="col-span-3 flex items-center">
                  <Select
                    value={endpoint}
                    onValueChange={(value) =>
                      setEndpoint(value)
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={t('config.search.selectRegionPlaceholder')} />
                    </SelectTrigger>
                    <SelectContent>
                      {ENDPOINT_LIST.map((endpoint) => (
                        <SelectItem key={endpoint} value={endpoint}>
                          {endpoint}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4 pt-4">
                <Label htmlFor="aliyun_ak" className="text-right">
                  {t('config.search.accessKeyId')}
                </Label>
                <div className="col-span-3 flex items-center">
                  <Input
                    id="aliyun_ak"
                    defaultValue={aliyunHasKey ? '******' : ''}
                    type="password"
                    onChange={(e) => setAliyunAK(e.target.value)}
                    placeholder={t('config.search.accessKeyIdPlaceholder')}
                    className="col-span-3"
                  />
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4 pt-4">
                <Label htmlFor="aliyun_sk" className="text-right">
                  {t('config.search.accessKeySecret')}
                </Label>
                <div className="col-span-3 flex items-center">
                  <Input
                    id="aliyun_sk"
                    defaultValue={aliyunHasKey ? '******' : ''}
                    type="password"
                    onChange={(e) => setAliyunSK(e.target.value)}
                    placeholder={t('config.search.accessKeySecretPlaceholder')}
                    className="col-span-3"
                  />
                </div>
              </div>
              </div>
            )}
           { searchEngineType === 'tavily' && (
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="tavily_key" className="text-right">
                {t('config.search.tavilyApiKey')}
                </Label>
                <div className="col-span-3 flex items-center">
                  <Input
                    id="tavily_key"
                    defaultValue={tavilyHasKey ? '******' : ''}
                    type="password"
                    onChange={(e) => setTavilyApiKey(e.target.value)}
                    placeholder={t('config.search.tavilyApiKeyPlaceholder')}
                    className="col-span-3"
                  />
                </div>
              </div>

           )}
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="search_count" className="text-right">
              {t('config.search.searchResultCount', { count: searchCount })}
              </Label>
              <div className="col-span-3 flex items-center">
                <Slider defaultValue={[10]} max={20} min={1} step={1} onValueChange={(value: number[]) => setSearchCount(value[0])} />
              </div>
            </div>

          </div>

          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="mt-4 px-4 py-2 text-white rounded-lg transition-colors"
          >
            {isSaving ? t('config.search.saving') : t('config.search.saveSearchConfig')}
          </Button>
        </div>
      </div>
    </div>
  );
}
