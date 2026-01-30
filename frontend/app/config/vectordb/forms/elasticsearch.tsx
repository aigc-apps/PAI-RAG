// app/config/vectordb/forms/elasticsearch.tsx

import React, { FC, useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useI18n } from '@/app/providers/i18n';


export interface ElasticConfig {
  endpoint: string;
  user: string;
  password: string;
}

interface ElasticConfigProps {
  config: ElasticConfig;
  onValueChange: (config: ElasticConfig) => void;
}

export const ElasticsearchForm: FC<ElasticConfigProps> = ({
  config,
  onValueChange
}) => {
  const { t } = useI18n();
  // Ensure user field has default value
  const getConfigWithDefaults = (cfg: Partial<ElasticConfig>): ElasticConfig => ({
    endpoint: cfg.endpoint || '',
    user: cfg.user || 'elastic',
    password: cfg.password || '',
  });
  
  const [db, setDb] = useState<ElasticConfig>(getConfigWithDefaults(config));
  const [isInitialized, setIsInitialized] = useState(false);
  
  // On initialization, if user field does not exist, ensure default value is passed to parent component
  useEffect(() => {
    if (!isInitialized) {
      const configWithDefaults = getConfigWithDefaults(config);
      if (!config.user) {
        onValueChange(configWithDefaults);
      }
      setIsInitialized(true);
    }
  }, [isInitialized, config.user]);
  
  // Sync external config changes to internal state (but don't trigger onValueChange to avoid loop)
  useEffect(() => {
    const updatedConfig = getConfigWithDefaults(config);
    setDb(updatedConfig);
  }, [config.endpoint, config.user, config.password]);
  
  return (
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="endpoint">{t('config.vectordb.elasticsearchEndpoint')}</Label>
          <Input id="endpoint" value={db.endpoint || ''} onChange={(e) => {
            const newConfig = { ...db, endpoint: e.target.value };
            setDb(newConfig);
            onValueChange(newConfig);
          }} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="user">{t('config.vectordb.username')}</Label>
          <Input id="user" type="string" value={db.user || "elastic"} onChange={(e) => {
            const newConfig = { ...db, user: e.target.value };
            setDb(newConfig);
            onValueChange(newConfig);
          }} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">{t('config.vectordb.password')}</Label>
          <Input id="password" type="password" value={db.password || ''} onChange={(e) => {
            const newConfig = { ...db, password: e.target.value };
            setDb(newConfig);
            onValueChange(newConfig);
          }} />
        </div>
    </div>
  );
};