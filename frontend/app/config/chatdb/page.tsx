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
    dialect: 'mysql',
    host: '',
    port: 3306,
    username: '',
    password: '',
    db_name: '',
    model_id: '',
  });

  const [llms, setLlms] = useState<LlmConfig[]>([]);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const { tenantFetch } = useTenantFetch();
  const router = useRouter();

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const [llmRes, dbRes] = await Promise.all([
          tenantFetch(`/api/config/llms`),
          tenantFetch(`/api/config/chatdb`),
        ]);

        const llmResponse = await llmRes.json();
        if (llmResponse.code === 200) {
          setLlms(llmResponse.data.items);
        } else {
          toast.error(t('config.chatdb.loadLlmFailed') + ': ' + llmResponse.message);
        }

        const dbResponse = await dbRes.json();
        if (dbResponse.code === 200 && dbResponse.data.length > 0) {
          if (
            dbResponse.data[0].encrypted_password !== '' &&
            dbResponse.data[0].encrypted_password !== undefined
          ) {
            setDbConfig({ ...dbResponse.data[0], password: '******' });
          } else {
            setDbConfig(dbResponse.data[0]);
          }
        } else if (dbResponse.code !== 200) {
          toast.error(t('config.chatdb.fetchChatdbFailed') + ': ' + dbResponse.message);
        }
      } catch (err: any) {
        toast.error(t('config.chatdb.configLoadFailed'));
      }
    };
    fetchConfig();
  }, []);

  const checkConfig = () => {
    if (!dbConfig.host) {
      toast.warning(t('config.chatdb.hostRequired'));
      return false;
    }
    if (!dbConfig.port) {
      toast.warning(t('config.chatdb.portRequired'));
      return false;
    }
    if (!dbConfig.username) {
      toast.warning(t('config.chatdb.usernameRequired'));
      return false;
    }
    if (!dbConfig.password) {
      toast.warning(t('config.chatdb.passwordRequired'));
      return false;
    }
    if (!dbConfig.db_name) {
      toast.warning(t('config.chatdb.dbNameRequired'));
      return false;
    }
    if (dbConfig.dialect !== 'mysql' && dbConfig.dialect !== 'postgresql') {
      toast.warning(t('config.chatdb.dialectSupported'));
      return false;
    }
    return true;
  };

  const handleConnect = async () => {
    if (!checkConfig()) return;
    try {
      setIsConnecting(true);
      const password = dbConfig.password === '******' ? '' : dbConfig.password;
      const res = await tenantFetch(`/api/config/chatdb/connectiontest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...dbConfig, password }),
      });
      const response = await res.json();
      if (response.code !== 200) toast.error(response.message);
      else toast.success(t('config.chatdb.connectSuccess'));
    } catch (err: any) {
      toast.error(`${t('config.chatdb.connectFailed')}: ${err.message}`);
    } finally {
      setIsConnecting(false);
    }
  };

  const handleSave = async () => {
    if (!checkConfig()) return;
    try {
      setIsSaving(true);
      const password = dbConfig.password === '******' ? '' : dbConfig.password;
      const res = await tenantFetch(`/api/config/chatdb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...dbConfig, password }),
      });
      const response = await res.json();
      if (response.code !== 200) toast.error(response.message);
      else toast.success(t('config.chatdb.saveSuccess'));
    } catch (err: any) {
      toast.error(`${t('config.chatdb.saveFailed')}: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="chatdb" className="settings-page">
      <div className="settings-page-header">
        <h1 className="page-title">{t('config.chatdb.title')}</h1>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleConnect}
            disabled={isConnecting}
          >
            {isConnecting
              ? t('config.chatdb.connecting')
              : t('config.chatdb.testConnection')}
          </Button>
          <Button size="sm" onClick={handleSave} disabled={isSaving}>
            {isSaving
              ? t('config.chatdb.saving')
              : t('config.chatdb.saveChatdbConfig')}
          </Button>
        </div>
      </div>

      <div className="settings-page-content">
        <div className="space-y-6">
          <div>
            <div className="dialog-section-title mb-2">问答模型</div>
            <div>
              <label className="form-label">
                {t('config.chatdb.baseModel')}
                <span className="required">*</span>
              </label>
              {llms.length > 0 ? (
                <Select
                  value={dbConfig.model_id}
                  onValueChange={(v) =>
                    setDbConfig((prev) => ({ ...prev, model_id: v }))
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
                <div className="flex items-center justify-between rounded-md border border-dashed p-3">
                  <p className="text-sm text-muted-foreground">
                    {t('config.chatdb.noModelConfigured')}
                  </p>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => router.push('/config/model')}
                  >
                    {t('config.chatdb.goToAdd')}
                  </Button>
                </div>
              )}
            </div>
          </div>

          <div>
            <div className="dialog-section-title mb-2">数据库连接</div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="dialect" className="form-label">
                  {t('config.chatdb.dbType')}
                  <span className="required">*</span>
                </label>
                <Select
                  value={dbConfig.dialect}
                  onValueChange={(v) => setDbConfig((prev) => ({ ...prev, dialect: v }))}
                >
                  <SelectTrigger id="dialect">
                    <SelectValue placeholder={t('config.chatdb.selectDbType')} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mysql">MySQL</SelectItem>
                    <SelectItem value="postgresql">PostgreSQL</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label htmlFor="database" className="form-label">
                  {t('config.chatdb.database')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="database"
                  value={dbConfig.db_name}
                  onChange={(e) => setDbConfig({ ...dbConfig, db_name: e.target.value })}
                />
              </div>
              <div>
                <label htmlFor="host" className="form-label">
                  {t('config.chatdb.host')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="host"
                  value={dbConfig.host || ''}
                  onChange={(e) => setDbConfig({ ...dbConfig, host: e.target.value })}
                />
              </div>
              <div>
                <label htmlFor="port" className="form-label">
                  {t('config.chatdb.port')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="port"
                  type="number"
                  value={dbConfig.port}
                  onChange={(e) =>
                    setDbConfig({ ...dbConfig, port: parseInt(e.target.value) || 0 })
                  }
                />
              </div>
              <div>
                <label htmlFor="user" className="form-label">
                  {t('config.chatdb.username')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="user"
                  value={dbConfig.username || ''}
                  onChange={(e) => setDbConfig({ ...dbConfig, username: e.target.value })}
                />
              </div>
              <div>
                <label htmlFor="password" className="form-label">
                  {t('config.chatdb.password')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="password"
                  type="password"
                  value={dbConfig.password || ''}
                  onChange={(e) => setDbConfig({ ...dbConfig, password: e.target.value })}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
