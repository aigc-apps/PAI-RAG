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
import { ExternalLink } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export default function GuardrailConfig() {
  const { t, language } = useI18n();

  const getRegionName = (region: string, network: string) => {
    const regionKey = region.toLowerCase().replace('-', '') as
      | 'shanghai'
      | 'beijing'
      | 'hangzhou'
      | 'shenzhen'
      | 'chengdu'
      | 'singapore';
    const regionMap: Record<string, string> = {
      shanghai: t('config.guardrail.regionShanghai'),
      beijing: t('config.guardrail.regionBeijing'),
      hangzhou: t('config.guardrail.regionHangzhou'),
      shenzhen: t('config.guardrail.regionShenzhen'),
      chengdu: t('config.guardrail.regionChengdu'),
      singapore: t('config.guardrail.regionSingapore'),
    };
    const networkText =
      network === 'public'
        ? t('config.guardrail.publicNetwork')
        : t('config.guardrail.privateNetwork');
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

  const REGION_ID_MAP = new Map([
    [getRegionName('shanghai', 'public'), 'cn-shanghai'],
    [getRegionName('shanghai', 'private'), 'cn-shanghai'],
    [getRegionName('beijing', 'public'), 'cn-beijing'],
    [getRegionName('beijing', 'private'), 'cn-beijing'],
    [getRegionName('hangzhou', 'public'), 'cn-hangzhou'],
    [getRegionName('hangzhou', 'private'), 'cn-hangzhou'],
    [getRegionName('shenzhen', 'public'), 'cn-shenzhen'],
    [getRegionName('shenzhen', 'private'), 'cn-shenzhen'],
    [getRegionName('chengdu', 'public'), 'cn-chengdu'],
    [getRegionName('singapore', 'public'), 'ap-southeast-1'],
    [getRegionName('singapore', 'private'), 'ap-southeast-1'],
  ]);

  const REGION_ENDPOINT_MAP = new Map([
    [getRegionName('shanghai', 'public'), 'green-cip.cn-shanghai.aliyuncs.com'],
    [getRegionName('shanghai', 'private'), 'green-cip-vpc.cn-shanghai.aliyuncs.com'],
    [getRegionName('beijing', 'public'), 'green-cip.cn-beijing.aliyuncs.com'],
    [getRegionName('beijing', 'private'), 'green-cip-vpc.cn-beijing.aliyuncs.com'],
    [getRegionName('hangzhou', 'public'), 'green-cip.cn-hangzhou.aliyuncs.com'],
    [getRegionName('hangzhou', 'private'), 'green-cip-vpc.cn-hangzhou.aliyuncs.com'],
    [getRegionName('shenzhen', 'public'), 'green-cip.cn-shenzhen.aliyuncs.com'],
    [getRegionName('shenzhen', 'private'), 'green-cip-vpc.cn-shenzhen.aliyuncs.com'],
    [getRegionName('chengdu', 'public'), 'green-cip.cn-chengdu.aliyuncs.com'],
    [getRegionName('singapore', 'public'), 'green-cip.ap-southeast-1.aliyuncs.com'],
    [getRegionName('singapore', 'private'), 'green-cip-vpc.ap-southeast-1.aliyuncs.com'],
  ]);

  const [aliyunHasKey, setAliyunHasKey] = useState(false);
  const [aliyunAK, setAliyunAK] = useState('');
  const [aliyunSK, setAliyunSK] = useState('');
  const [regionName, setRegionName] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchConfig = async () => {
      try {
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
        if (savedRegionName) setRegionName(savedRegionName);
        else setRegionName(getRegionName('hangzhou', 'public'));
      } catch (err: any) {
        toast.error(t('config.guardrail.configLoadFailed'));
      }
    };
    fetchConfig();
  }, [language]);

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
          region_id: REGION_ID_MAP.get(regionName),
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
    <div id="guardrail" className="settings-page">
      <div className="settings-page-header">
        <h1 className="page-title">{t('config.guardrail.aliyunGuardrailTitle')}</h1>
        <Button size="sm" onClick={handleSave} disabled={isSaving}>
          {isSaving ? t('common.saving') : t('config.guardrail.saveConfig')}
        </Button>
      </div>

      <div className="settings-page-content">
        <div className="space-y-6">
          <div>
            <a
              href="https://www.aliyun.com/product/content-moderation/guardrail"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-primary hover:underline inline-flex items-center gap-1"
            >
              {t('config.guardrail.openUrl')}
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div>
            <div className="dialog-section-title mb-2">服务区域</div>
            <div>
              <label className="form-label">{t('config.guardrail.selectRegion')}</label>
              <Select value={regionName} onValueChange={setRegionName}>
                <SelectTrigger>
                  <SelectValue placeholder={t('config.guardrail.selectRegionPlaceholder')} />
                </SelectTrigger>
                <SelectContent>
                  {REGION_NAMES.map((r) => (
                    <SelectItem key={r} value={r}>
                      {r}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div>
            <div className="dialog-section-title mb-2">Aliyun 凭证</div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="aliyun_ak" className="form-label">
                  {t('config.guardrail.accessKeyId')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="aliyun_ak"
                  defaultValue={aliyunHasKey ? '******' : ''}
                  type="password"
                  onChange={(e) => setAliyunAK(e.target.value)}
                  placeholder={t('config.guardrail.accessKeyIdPlaceholder')}
                />
              </div>
              <div>
                <label htmlFor="aliyun_sk" className="form-label">
                  {t('config.guardrail.accessKeySecret')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="aliyun_sk"
                  defaultValue={aliyunHasKey ? '******' : ''}
                  type="password"
                  onChange={(e) => setAliyunSK(e.target.value)}
                  placeholder={t('config.guardrail.accessKeySecretPlaceholder')}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
