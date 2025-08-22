'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import * as Toast from '@radix-ui/react-toast';
import { EyeIcon, EyeOffIcon } from 'lucide-react'; // 示例图标库

export default function SearchConfig() {
  const [aliyunHasKey, setAliyunHasKey] = useState(false); // AccessKey ID
  const [aliyunAK, setAliyunAK] = useState(''); // AccessKey ID
  const [aliyunSK, setAliyunSK] = useState(''); // AccessKey Secret
  const [isLoading, setIsLoading] = useState(false); // 加载状态
  const [error, setError] = useState(''); // 错误提示
  const [showAK, setShowAK] = useState(false); // 是否显示 AK
  const [showSK, setShowSK] = useState(false); // 是否显示 SK
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

        const res = await fetch(`/api/config/websearch`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error('加载配置失败');

        const data = await res.json();
        setAliyunHasKey(data.length > 0);
        setAliyunAK(data[0]?.encrypted_access_key_id || '');
        setAliyunSK(data[0]?.encrypted_access_key_secret || '');
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

      const res = await fetch(`/api/config/websearch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify({
          access_key_id: update_ak,
          access_key_secret: update_sk,
          type: 'aliyun',
          endpoint: 'iqs.cn-zhangjiakou.aliyuncs.com',
        }),
      });

      if (!res.ok) throw new Error('保存失败，请检查网络或配置');

      setToastState({
        open: true,
        title: '阿里云搜索配置已成功保存',
        description: '阿里云搜索配置已成功保存',
        variant: 'default',
      });
    } catch (err: any) {
      setError(err.message || '保存失败，请重试');
      setToastState({
        open: true,
        title: '阿里云搜索配置保存失败',
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
          <h2 className="text-2xl font-bold text-gray-800">阿里云搜索配置</h2>
          <div className="grid gap-4 py-4">
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
            {isLoading ? '保存中...' : '保存搜索配置'}
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
