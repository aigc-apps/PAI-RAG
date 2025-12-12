'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { LlmConfig } from '@/app/config/model/llm/page';
import { useRouter } from 'next/navigation';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';


interface ChatDbConfig {
  dialect: string;
  host: string;
  port: number;
  username: string;
  password: string;
  db_name: string;
  model_id: string;
}

export default function ChatdbConfig() {
  const [dbConfig, setDbConfig] = useState<ChatDbConfig>({
    dialect: "mysql",
    host: "",
    port: 3306,
    username: "",
    password: "",
    db_name: "",
    model_id: "",
  });

  const [llms, setLlms] = useState<LlmConfig[]>([]);

  const [isConnecting, setIsConnecting] = useState(false); // 连接测试中
  const [isLoading, setIsLoading] = useState(false); // 加载状态
  const [isSaving, setIsSaving] = useState(false); // 加载状态
  const [error, setError] = useState(''); // 错误提示
  const { tenantFetch } = useTenantFetch();

  const router = useRouter();
  // 初始化加载配置
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoading(true);
        setError('');

        const [llmRes, dbRes] = await Promise.all([
          tenantFetch(`/api/config/llms`),
          tenantFetch(`/api/config/chatdb`)]);


        const llmResponse = await llmRes.json();
        if (llmResponse.code === 200) {
          setLlms(llmResponse.data.items);
        }
        else {
          toast.error("加载大模型列表失败: " + llmResponse.message);
        }

        const dbResponse = await dbRes.json()
        if (dbResponse.code === 200) {
          if (dbResponse.data.length > 0) {
            if (dbResponse.data[0].encrypted_password !== "" && dbResponse.data[0].encrypted_password !== undefined) {
              setDbConfig({...dbResponse.data[0], password: "******"})
            }
            else {
              setDbConfig(dbResponse.data[0]);
            }
          }
        }
        else {
          toast.error("获取ChatDB信息失败: " + dbResponse.message);
        }

      } catch (err: any) {
        toast.error("配置加载失败,请检查网络或重试")
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);

  const checkConfig = () => {
    if (!dbConfig.host) {
      toast.warning("主机地址必须填入。")
      return;
    }
      if (!dbConfig.port) {
      toast.warning("端口号必须填入。")
      return;
    }
    if (!dbConfig.username) {
      toast.warning("用户名必须填入。")
      return;
    }
    if (!dbConfig.password) {
      toast.warning("密码必须填入。")
      return;
    }
    if (!dbConfig.db_name) {
      toast.warning("数据库名称必须填入。");
      return;
    }
    if (dbConfig.dialect !== 'mysql' && dbConfig.dialect != "postgresql") {
      toast.warning("数据库只支持mysql或者postgresql.");
      return;
    }
  }
  const handleConnect = async() => {
    checkConfig();

    try {
      setIsConnecting(true);

      const password = dbConfig.password === '******' ? '' : dbConfig.password;
      const res = await tenantFetch(`/api/config/chatdb/connectiontest`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },

              body: JSON.stringify({
                ...dbConfig,
                password: password,
              }),
            });

      const response = await res.json();
      if (response.code !== 200) {
        toast.error(response.message);
      }
      else {
        toast.success("连接成功！")
      }
    } catch (err: any) {
      toast.error(`连接失败: ${err.message}`);
    } finally {
      setIsConnecting(false);
    }
  };

  // 保存配置
  const handleSave = async () => {
    checkConfig();

    try {
      setIsSaving(true);

      const password = dbConfig.password === '******' ? '' : dbConfig.password;

      const res = await tenantFetch(`/api/config/chatdb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify({
          ...dbConfig,
          password: password,
        }),
      });

      const response = await res.json();
      if (response.code !== 200) {
        toast.error(response.message);
      }
      else {
        toast.success("Chatdb配置已成功保存。")
      }
    } catch (err: any) {
      toast.error(`保存失败: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="chatdb">
        <div className="py-6 px-6">
          <h2 className="text-xl font-medium text-gray-800">ChatDB配置</h2>
            <div className="flex py-6 px-4">
              <Label htmlFor="basemodel" className="w-[90px]">
                基模型选择 <span className="text-destructive">*</span>{' '}
              </Label>
              <div className="px-6">
                {llms.length > 0 ? (
                  <Select
                    value={dbConfig.model_id}
                    onValueChange={(value) =>
                      setDbConfig((prev) => ({
                        ...prev,
                        model_id: value,
                      }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="请选择基模型" />
                    </SelectTrigger>
                    <SelectContent>
                      {llms.map((llm) => (
                        <SelectItem key={llm.id} value={llm.model_id}>
                          {llm.model_id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <div>
                    <p className="text-sm text-muted-foreground">尚未配置大模型</p>
                    <Button
                      variant="outline"
                      onClick={() => {
                        router.push('/config/model/llm');
                      }}
                    >
                      前往添加
                    </Button>
                  </div>
                )}
              </div>
            </div>
            <div className="flex py-3 px-4">
              <Label htmlFor="basemodel" className="w-[90px]">
                数据库类型 <span className="text-destructive">*</span>{' '}
              </Label>
              <div className="px-6">
                  <Select
                    value={dbConfig.dialect}
                    onValueChange={(value) =>
                      setDbConfig((prev) => ({
                        ...prev,
                        dialect: value,
                      }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="请选择数据库类型" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem key="mysql" value="mysql">
                        mysql
                      </SelectItem>
                      <SelectItem key="postgresql" value="postgresql">
                        postgresql
                      </SelectItem>
                    </SelectContent>
                  </Select>
              </div>
            </div>
            <div className="flex px-4 py-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="host">主机地址</Label>
                <Input id="host" value={dbConfig.host || ''} onChange={(e) => {
                  setDbConfig({...dbConfig, host: e.target.value});
                }} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="port">端口</Label>
                <Input id="port" type="number" value={dbConfig.port} onChange={(e) => {
                  setDbConfig({...dbConfig, port: parseInt(e.target.value)});
                }} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="database">数据库名</Label>
                <Input id="database"  value={dbConfig.db_name} onChange={(e) => {
                  setDbConfig({...dbConfig, db_name: e.target.value});
                }} />
              </div>
            </div>
            <div className="flex px-4 py-6 gap-4">
              <div className="space-y-2">
                <Label htmlFor="user">用户名</Label>
                <Input id="user" type="string" value={dbConfig.username || ''} onChange={(e) => {
                  setDbConfig({...dbConfig, username: e.target.value});
                  }} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">密码</Label>
                <Input id="password" type="password" value={dbConfig.password  || ''} onChange={(e) => {
                  setDbConfig({...dbConfig, password: e.target.value});
                  }} />
              </div>
            </div>
            <div className="flex gap-12 px-4">
              <Button
                onClick={handleSave}
                disabled={isSaving}
                className="mt-4 text-white rounded-lg transition-colors"
              >
                {isSaving ? '保存中...' : '保存ChatDB配置'}
              </Button>
              <Button
                variant="secondary"
                onClick={handleConnect}
                disabled={isConnecting}
                className="mt-4 rounded-lg transition-colors"
              >
                {isConnecting ? '连接中...' : '连接测试'}
              </Button>
            </div>
          </div>
    </div>
  );
}
