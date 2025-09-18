'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { Slider } from "@/components/ui/slider"

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
  const [aliyunHasKey, setAliyunHasKey] = useState(false); // AccessKey ID
  const [tavilyHasKey, setTavilyHasKey] = useState(false); // AccessKey ID

  const [aliyunAK, setAliyunAK] = useState(''); // AccessKey ID
  const [aliyunSK, setAliyunSK] = useState(''); // AccessKey Secret
  const [endpoint, setEndpoint] = useState('')
  const [isLoading, setIsLoading] = useState(false); // 加载状态
  const [searchCount, setSearchCount] = useState(10); // 每次搜索返回的结果数
  const [tavilyApiKey, setTavilyApiKey] = useState(''); // Tavily API Key
  const [searchEngineType, setSearchEngineType] = useState('aliyun'); // 搜索引擎类型

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
        setAliyunHasKey(!data.is_aliyun_empty);
        setTavilyHasKey(!data.is_tavily_empty);
        setAliyunAK(MASK_API_KEY);
        setAliyunSK(MASK_API_KEY);
        setEndpoint(data[0]?.endpoint || "")
        setSearchCount(data[0]?.search_count || 10)
        setSearchEngineType(data[0]?.type || 'aliyun')
        setTavilyApiKey(MASK_API_KEY)
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

      const update_ak = aliyunAK === MASK_API_KEY ? '' : aliyunAK;
      const update_sk = aliyunSK === MASK_API_KEY ? '' : aliyunSK;
      const update_tavily_api_key = tavilyApiKey === MASK_API_KEY ? '' : tavilyApiKey;

      const res = await fetch(`/api/config/websearch`, {
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
          <h2 className="text-2xl font-bold text-gray-800">搜索配置</h2>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="endpoint" className="text-right">
                选择搜索引擎：
              </Label>
               <div className="col-span-3 flex items-center">
                <Select
                  value={searchEngineType}
                  onValueChange={(value) =>
                    setSearchEngineType(value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="请选择搜索引擎" />
                  </SelectTrigger>
                  <SelectContent>
                      <SelectItem key='aliyun' value='aliyun'>
                        阿里云通用搜索
                      </SelectItem>
                      <SelectItem key='tavily' value='tavily'>
                        Tavily 搜索
                      </SelectItem>
                  </SelectContent>
                </Select>

                <a href={searchEngineType=== 'aliyun' ? "https://help.aliyun.com/document_detail/2870227.html" : "https://www.tavily.com/"} target="_blank" className="ml-4 text-sm text-blue-600 hover:underline">开通指南 </a>
              </div>

            </div>
            { searchEngineType === 'aliyun' && (
              <div className="gap-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="endpoint" className="text-right">
                  通用搜索Endpoint
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
              <div className="grid grid-cols-4 items-center gap-4 pt-4">
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
              <div className="grid grid-cols-4 items-center gap-4 pt-4">
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
            )}
           { searchEngineType === 'tavily' && (
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="tavily_key" className="text-right">
                Tavily API Key
                </Label>
                <div className="col-span-3 flex items-center">
                  <Input
                    id="tavily_key"
                    defaultValue={tavilyHasKey ? '******' : ''}
                    type="password" // 动态切换类型
                    onChange={(e) => setTavilyApiKey(e.target.value)}
                    placeholder="输入 Tavily API Key"
                    className="col-span-3"
                  />
                </div>
              </div>

           )}
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="search_count" className="text-right">
              搜索结果条数({searchCount})
              </Label>
              <div className="col-span-3 flex items-center">
                <Slider defaultValue={[10]} max={20} min={1} step={1} onValueChange={(value: number[]) => setSearchCount(value[0])} />
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
