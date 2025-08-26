'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import * as Toast from '@radix-ui/react-toast';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import Link from 'next/link';

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
  const [toastState, setToastState] = useState({
    open: false,
    title: '',
    description: '',
    variant: 'default' as 'default' | 'destructive',
  });

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
        setRegionName(data[0]?.region_name || '杭州（公网）')
      } catch (err: any) {
        setError(err.message || '加载失败');
        setToastState({
          open: true,
          title: '配置加载失败',
          description: err.message || '请检查网络或重试',
          variant: 'destructive',
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);
  // 保存配置
  const handleSave = async () => {
    if (!aliyunAK || !aliyunSK) {
      setError('AccessKey ID 和 Secret 不能为空');
      return;
    }
    try {
      setIsLoading(true);
      setError('');

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

      if (!res.ok) throw new Error('保存失败，请检查网络或配置');

      setToastState({
        open: true,
        title: 'AI护栏配置已成功保存',
        description: 'AI护栏配置已成功保存',
        variant: 'default',
      });
    } catch (err: any) {
      setError(err.message || '保存失败，请重试');
      setToastState({
        open: true,
        title: 'AI护栏配置保存失败',
        description: err.message || '请检查网络或重试',
        variant: 'destructive',
      });
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
        <Toast.Root
          open={toastState.open}
          onOpenChange={(open) => setToastState((prev) => ({ ...prev, open }))}
          className={`grid grid-cols-[auto_1fr] items-center gap-x-4 rounded-md border px-4 py-6 shadow-lg transition-all data-[state=open]:animate-slideIn data-[state=closed]:animate-fadeOut ${
            toastState.variant === 'destructive'
              ? 'border-red-500 bg-red-50 text-red-900'
              : 'border-gray-200 bg-white text-gray-900'
          }`}
        >
          <Toast.Description className="pl-4 text-sm font-medium">
            {toastState.description}
          </Toast.Description>
          <Toast.Action
            altText="关闭"
            onClick={() => setToastState((prev) => ({ ...prev, open: false }))}
          >
            ×
          </Toast.Action>
        </Toast.Root>

        {/* 触发 Toast 的隐藏容器 */}
        <Toast.Viewport className="fixed bottom-0 right-0 z-[100] m-0 flex w-96 flex-col gap-2 p-6" />
      </div>
    </div>
  );
}
