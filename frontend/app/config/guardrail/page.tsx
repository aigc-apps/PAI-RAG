'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export default function GuardrailConfig() {
  const { t, language } = useI18n();

  // Dynamic region names based on language
  const getRegionName = (region: string, network: string) => {
    const regionKey = region.toLowerCase().replace('-', '') as 'shanghai' | 'beijing' | 'hangzhou' | 'shenzhen' | 'chengdu' | 'singapore';
    const regionMap: Record<string, string> = {
      'shanghai': t('config.guardrail.regionShanghai'),
      'beijing': t('config.guardrail.regionBeijing'),
      'hangzhou': t('config.guardrail.regionHangzhou'),
      'shenzhen': t('config.guardrail.regionShenzhen'),
      'chengdu': t('config.guardrail.regionChengdu'),
      'singapore': t('config.guardrail.regionSingapore'),
    };
    const networkText = network === 'public' ? t('config.guardrail.publicNetwork') : t('config.guardrail.privateNetwork');
    return `${regionMap[regionKey]}（${networkText}）`;
  };

  const REGION_NAMES = [
    getRegionName('shanghai', 'public'),
    getRegionName('shanghai', 'private'),
    getRegionName('beijing', 'public'),
    getRegionName('beijing', 'private'),
    getRegionName('hangzhou', 'public'),
    getRegionName('hangzhou', 'private'),
    getRegionName('shenzhen', 'public'),
    getRegionName('shenzhen', 'private'),
    getRegionName('chengdu', 'public'),
    getRegionName('singapore', 'public'),
    getRegionName('singapore', 'private'),
  ];

  const REGION_ID_MAP = new Map(
    [
      [getRegionName('shanghai', 'public'), "cn-shanghai"],
      [getRegionName('shanghai', 'private'), "cn-shanghai"],
      [getRegionName('beijing', 'public'), "cn-beijing"],
      [getRegionName('beijing', 'private'), "cn-beijing"],
      [getRegionName('hangzhou', 'public'), "cn-hangzhou"],
      [getRegionName('hangzhou', 'private'), "cn-hangzhou"],
      [getRegionName('shenzhen', 'public'), "cn-shenzhen"],
      [getRegionName('shenzhen', 'private'), "cn-shenzhen"],
      [getRegionName('chengdu', 'public'), "cn-chengdu"],
      [getRegionName('singapore', 'public'), "ap-southeast-1"],
      [getRegionName('singapore', 'private'), "ap-southeast-1"],
    ]
  );

  const REGION_ENDPOINT_MAP = new Map(
    [
      [getRegionName('shanghai', 'public'), "green-cip.cn-shanghai.aliyuncs.com"],
      [getRegionName('shanghai', 'private'), "green-cip-vpc.cn-shanghai.aliyuncs.com"],
      [getRegionName('beijing', 'public'), "green-cip.cn-beijing.aliyuncs.com"],
      [getRegionName('beijing', 'private'), "green-cip-vpc.cn-beijing.aliyuncs.com"],
      [getRegionName('hangzhou', 'public'), "green-cip.cn-hangzhou.aliyuncs.com"],
      [getRegionName('hangzhou', 'private'), "green-cip-vpc.cn-hangzhou.aliyuncs.com"],
      [getRegionName('shenzhen', 'public'), "green-cip.cn-shenzhen.aliyuncs.com"],
      [getRegionName('shenzhen', 'private'), "green-cip-vpc.cn-shenzhen.aliyuncs.com"],
      [getRegionName('chengdu', 'public'), "green-cip.cn-chengdu.aliyuncs.com"],
      [getRegionName('singapore', 'public'), "green-cip.ap-southeast-1.aliyuncs.com"],
      [getRegionName('singapore', 'private'), "green-cip-vpc.ap-southeast-1.aliyuncs.com"],
    ]
  );
  const [aliyunHasKey, setAliyunHasKey] = useState(false);
  const [aliyunAK, setAliyunAK] = useState(''); // AccessKey ID
  const [aliyunSK, setAliyunSK] = useState(''); // AccessKey Secret
  const [regionName, setRegionName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');

  const { tenantFetch } = useTenantFetch();
  
  // Initialize and load configuration
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoading(true);
        setError('');

        const res = await tenantFetch(`/api/config/guardrail`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error(t('config.guardrail.loadError'));

        const data = (await res.json()).data;
        setAliyunHasKey(data.length > 0);
        setAliyunAK(data[0]?.encrypted_access_key_id || '');
        setAliyunSK(data[0]?.encrypted_access_key_secret || '');
        const savedRegionName = data[0]?.region_name || '';
        // Set default based on language if no saved value
        if (savedRegionName) {
          setRegionName(savedRegionName);
        } else {
          setRegionName(getRegionName('hangzhou', 'public'));
        }
      } catch (err: any) {
        toast.error(t('config.guardrail.configLoadFailed'));
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, [language]); // Re-fetch when language changes to update region names
  
  // Save configuration
  const handleSave = async () => {
    if (!aliyunAK || !aliyunSK) {
      toast.warning(t('config.guardrail.akSkRequired'));
      return;
    }
    try {
      setIsSaving(true);

      const update_ak = aliyunAK === '******' ? '' : aliyunAK;
      const update_sk = aliyunSK === '******' ? '' : aliyunSK;

      const res = await tenantFetch(`/api/config/guardrail`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify({
          access_key_id: update_ak,
          access_key_secret: update_sk,
          region_name: regionName,
          endpoint: REGION_ENDPOINT_MAP.get(regionName),
          region_id: REGION_ID_MAP.get(regionName)
        }),
      });

      if (!res.ok) {
        toast.error(t('config.guardrail.saveFailedToast'));
        throw new Error(t('config.guardrail.saveFailed'));
      }

      toast.success(t('config.guardrail.saveSuccess'));
    } catch (err: any) {
      toast.error(`${t('config.guardrail.saveFailed')}: ${err.message}`);
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
          <div className="flex gap-6 items-center">
            <h2 className="text-xl font-medium text-gray-800">{t('config.guardrail.aliyunGuardrailTitle')}</h2> 
            <Button variant="outline" className="h-6" asChild><a href="https://www.aliyun.com/product/content-moderation/guardrail">{t('config.guardrail.openUrl')}</a></Button>
          </div>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="region" className="text-right">
                {t('config.guardrail.selectRegion')}
              </Label>
              <div className="col-span-3 flex items-center">
                <Select
                  value={regionName}
                  onValueChange={(value) =>
                    setRegionName(value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder={t('config.guardrail.selectRegionPlaceholder')} />
                  </SelectTrigger>
                  <SelectContent>
                    {REGION_NAMES.map((region_name) => (
                      <SelectItem key={region_name} value={region_name}>
                        {region_name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="aliyun_ak" className="text-right">
                {t('config.guardrail.accessKeyId')}
              </Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="aliyun_ak"
                  defaultValue={aliyunHasKey ? '******' : ''}
                  type="password"
                  onChange={(e) => setAliyunAK(e.target.value)}
                  placeholder={t('config.guardrail.accessKeyIdPlaceholder')}
                  className="col-span-3"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="aliyun_sk" className="text-right">
                {t('config.guardrail.accessKeySecret')}
              </Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="aliyun_sk"
                  defaultValue={aliyunHasKey ? '******' : ''}
                  type="password"
                  onChange={(e) => setAliyunSK(e.target.value)}
                  placeholder={t('config.guardrail.accessKeySecretPlaceholder')}
                  className="col-span-3"
                />
              </div>
            </div>
          </div>
          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="mt-4 px-4 py-2 text-white rounded-lg transition-colors"
          >
            {isSaving ? t('common.saving') : t('config.guardrail.saveConfig')}
          </Button>
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
      </div>
    </div>
  );
}
