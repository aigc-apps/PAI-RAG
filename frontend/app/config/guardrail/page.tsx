'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';

const REGION_NAMES = [
  "上海（公网）",
  "上海（内网）",
  "北京（公网）",
  "北京（内网）",
  "杭州（公网）",
  "杭州（内网）",
  "深圳（公网）",
  "深圳（内网）",
  "成都（公网）",
  "新加坡（公网）",
  "新加坡（内网）",
]

const REGION_ID_MAP = new Map(
  [
    ["上海（公网）", "cn-shanghai"],
    ["上海（内网）", "cn-shanghai"],
    ["北京（公网）", "cn-beijing"],
    ["北京（内网）", "cn-beijing"],
    ["杭州（公网）", "cn-hangzhou"],
    ["杭州（内网）", "cn-hangzhou"],
    ["深圳（公网）", "cn-shenzhen"],
    ["深圳（内网）", "cn-shenzhen"],
    ["成都（公网）", "cn-chengdu"],
    ["新加坡（公网）", "ap-southeast-1"],
    ["新加坡（内网）", "ap-southeast-1"],
  ]
)

const REGION_ENDPOINT_MAP = new Map(
  [
    ["上海（公网）", "green-cip.cn-shanghai.aliyuncs.com"],
    ["上海（内网）", "green-cip-vpc.cn-shanghai.aliyuncs.com"],
    ["北京（公网）", "green-cip.cn-beijing.aliyuncs.com"],
    ["北京（内网）", "green-cip-vpc.cn-beijing.aliyuncs.com"],
    ["杭州（公网）", "green-cip.cn-hangzhou.aliyuncs.com"],
    ["杭州（内网）", "green-cip-vpc.cn-hangzhou.aliyuncs.com"],
    ["深圳（公网）", "green-cip.cn-shenzhen.aliyuncs.com"],
    ["深圳（内网）", "green-cip-vpc.cn-shenzhen.aliyuncs.com"],
    ["成都（公网）", "green-cip.cn-chengdu.aliyuncs.com"],
    ["新加坡（公网）", "green-cip.ap-southeast-1.aliyuncs.com"],
    ["新加坡（内网）", "green-cip-vpc.ap-southeast-1.aliyuncs.com"],
  ]
)


export default function GuardrailConfig() {
  const [aliyunHasKey, setAliyunHasKey] = useState(false); // AccessKey ID
  const [aliyunAK, setAliyunAK] = useState(''); // AccessKey ID
  const [aliyunSK, setAliyunSK] = useState(''); // AccessKey Secret
  const [regionName, setRegionName] = useState('');
  const [isLoading, setIsLoading] = useState(false); // 加载状态
  const [error, setError] = useState(''); // 错误提示

  // 初始化加载配置
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoading(true);
        setError('');

        const res = await fetch(`/api/config/guardrail`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error('加载配置失败');

        const data = (await res.json()).data;
        setAliyunHasKey(data.length > 0);
        setAliyunAK(data[0]?.encrypted_access_key_id || '');
        setAliyunSK(data[0]?.encrypted_access_key_secret || '');
        setRegionName(data[0]?.region_name || '杭州（公网）');
      } catch (err: any) {
        toast.error("配置加载失败,请检查网络或重试")
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);
  // 保存配置
  const handleSave = async () => {
    if (!aliyunAK || !aliyunSK) {
      toast.warning("AK/SK必须填入。")
      return;
    }
    try {
      setIsLoading(true);

      const update_ak = aliyunAK === '******' ? '' : aliyunAK;
      const update_sk = aliyunSK === '******' ? '' : aliyunSK;

      const res = await fetch(`/api/config/guardrail`, {
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
        toast.error("AI护栏配置保存失败。");
        throw new Error("保存失败。")
      }

      toast.success("AK/AI护栏配置已成功保存。")
    } catch (err: any) {
      toast.success(`保存失败: ${err.message}`);
    } finally {
      setIsLoading(false);
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
            <h2 className="text-2xl font-bold text-gray-800">阿里云AI安全护栏配置</h2> 
            <Button variant="outline" className="h-6" asChild><a href="https://www.aliyun.com/product/content-moderation/guardrail">开通地址</a></Button>
          </div>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="region" className="text-right">
                选择服务地域
              </Label>
              <div className="col-span-3 flex items-center">
                <Select
                  value={regionName}
                  onValueChange={(value) =>
                    setRegionName(value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="请选择地域" />
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
                AccessKey ID
              </Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="aliyun_ak"
                  defaultValue={aliyunHasKey ? '******' : ''}
                  type="password" // 动态切换类型
                  onChange={(e) => setAliyunAK(e.target.value)}
                  placeholder="输入 AccessKey ID"
                  className="col-span-3"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="aliyun_sk" className="text-right">
                AccessKey Secret
              </Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="aliyun_sk"
                  defaultValue={aliyunHasKey ? '******' : ''}
                  type="password" // 动态切换类型
                  onChange={(e) => setAliyunSK(e.target.value)}
                  placeholder="输入 AccessKey Secret"
                  className="col-span-3"
                />
              </div>
            </div>
          </div>
          <Button
            onClick={handleSave}
            disabled={isLoading}
            className="mt-4 px-4 py-2 text-white rounded-lg transition-colors"
          >
            {isLoading ? '保存中...' : '保存AI护栏配置'}
          </Button>
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
      </div>
    </div>
  );
}
