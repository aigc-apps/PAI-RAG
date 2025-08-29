'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';

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

export default function SearchConfig() {
  const [aliyunHasKey, setAliyunHasKey] = useState(false); // AccessKey ID
  const [aliyunAK, setAliyunAK] = useState(''); // AccessKey ID
  const [aliyunSK, setAliyunSK] = useState(''); // AccessKey Secret
  const [endpoint, setEndpoint] = useState('')
  const [isLoading, setIsLoading] = useState(false); // 加载状态

  // 初始化加载配置
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoading(true);

        const res = await fetch(`/api/config/websearch`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error('加载配置失败');

        const data = await res.json();
        setAliyunHasKey(data.length > 0);
        setAliyunAK(data[0]?.encrypted_access_key_id || '');
        setAliyunSK(data[0]?.encrypted_access_key_secret || '');
        setEndpoint(data[0]?.endpoint || "")
      } catch (err: any) {
        toast.error(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);
  // 保存配置
  const handleSave = async () => {
    if (!aliyunAK || !aliyunSK) {
      toast.warning(`必须填入AK和SK信息`);
      return;
    }
    try {
      setIsLoading(true);

      const update_ak = aliyunAK === '******' ? '' : aliyunAK;
      const update_sk = aliyunSK === '******' ? '' : aliyunSK;

      const res = await fetch(`/api/config/websearch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify({
          access_key_id: update_ak,
          access_key_secret: update_sk,
          type: 'aliyun',
          endpoint: endpoint,
        }),
      });

      if (!res.ok) throw new Error('保存失败，请检查网络或配置');

      toast.success('搜索配置已成功保存。');
    } catch (err: any) {
      toast.warning(err.message);
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

              <Label htmlFor="endpoint" className="text-right">
                选择Endpoint
              </Label>
              <div className="col-span-3 flex items-center">
                <Select
                  value={endpoint}
                  onValueChange={(value) =>
                    setEndpoint(value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="请选择地域" />
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
        </div>
      </div>
    </div>
  );
}
