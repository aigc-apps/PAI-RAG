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
import { useI18n } from '@/app/providers/i18n';


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
  const { t } = useI18n();
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
          toast.error(t('config.chatdb.loadLlmFailed') + ': ' + llmResponse.message);
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
          toast.error(t('config.chatdb.fetchChatdbFailed') + ': ' + dbResponse.message);
        }

      } catch (err: any) {
        toast.error(t('config.chatdb.configLoadFailed'))
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);

  const checkConfig = () => {
    if (!dbConfig.host) {
      toast.warning(t('config.chatdb.hostRequired'))
      return;
    }
      if (!dbConfig.port) {
      toast.warning(t('config.chatdb.portRequired'))
      return;
    }
    if (!dbConfig.username) {
      toast.warning(t('config.chatdb.usernameRequired'))
      return;
    }
    if (!dbConfig.password) {
      toast.warning(t('config.chatdb.passwordRequired'))
      return;
    }
    if (!dbConfig.db_name) {
      toast.warning(t('config.chatdb.dbNameRequired'));
      return;
    }
    if (dbConfig.dialect !== 'mysql' && dbConfig.dialect != "postgresql") {
      toast.warning(t('config.chatdb.dialectSupported'));
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
        toast.success(t('config.chatdb.connectSuccess'))
      }
    } catch (err: any) {
      toast.error(`${t('config.chatdb.connectFailed')}: ${err.message}`);
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
        toast.success(t('config.chatdb.saveSuccess'))
      }
    } catch (err: any) {
      toast.error(`${t('config.chatdb.saveFailed')}: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="chatdb">
        <div className="py-6 px-6">
          <h2 className="text-xl font-medium text-gray-800">{t('config.chatdb.title')}</h2>
            <div className="flex py-6 px-4">
              <Label htmlFor="basemodel" className="w-[90px]">
                {t('config.chatdb.baseModel')} <span className="text-destructive">*</span>{' '}
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
                      <SelectValue placeholder={t('config.chatdb.selectBaseModel')} />
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
                    <p className="text-sm text-muted-foreground">{t('config.chatdb.noModelConfigured')}</p>
                    <Button
                      variant="outline"
                      onClick={() => {
                        router.push('/config/model/llm');
                      }}
                    >
                      {t('config.chatdb.goToAdd')}
                    </Button>
                  </div>
                )}
              </div>
            </div>
            <div className="flex py-3 px-4">
              <Label htmlFor="basemodel" className="w-[90px]">
                {t('config.chatdb.dbType')} <span className="text-destructive">*</span>{' '}
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
                      <SelectValue placeholder={t('config.chatdb.selectDbType')} />
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
                <Label htmlFor="host">{t('config.chatdb.host')}</Label>
                <Input id="host" value={dbConfig.host || ''} onChange={(e) => {
                  setDbConfig({...dbConfig, host: e.target.value});
                }} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="port">{t('config.chatdb.port')}</Label>
                <Input id="port" type="number" value={dbConfig.port} onChange={(e) => {
                  setDbConfig({...dbConfig, port: parseInt(e.target.value)});
                }} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="database">{t('config.chatdb.database')}</Label>
                <Input id="database"  value={dbConfig.db_name} onChange={(e) => {
                  setDbConfig({...dbConfig, db_name: e.target.value});
                }} />
              </div>
            </div>
            <div className="flex px-4 py-6 gap-4">
              <div className="space-y-2">
                <Label htmlFor="user">{t('config.chatdb.username')}</Label>
                <Input id="user" type="string" value={dbConfig.username || ''} onChange={(e) => {
                  setDbConfig({...dbConfig, username: e.target.value});
                  }} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">{t('config.chatdb.password')}</Label>
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
                {isSaving ? t('config.chatdb.saving') : t('config.chatdb.saveChatdbConfig')}
              </Button>
              <Button
                variant="secondary"
                onClick={handleConnect}
                disabled={isConnecting}
                className="mt-4 rounded-lg transition-colors"
              >
                {isConnecting ? t('config.chatdb.connecting') : t('config.chatdb.testConnection')}
              </Button>
            </div>
          </div>
    </div>
  );
}
