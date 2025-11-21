// app/config/vectordb/forms/elasticsearch.tsx

import React, { FC, useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";


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
  // 确保 user 字段有默认值
  const getConfigWithDefaults = (cfg: Partial<ElasticConfig>): ElasticConfig => ({
    endpoint: cfg.endpoint || '',
    user: cfg.user || 'elastic',
    password: cfg.password || '',
  });
  
  const [db, setDb] = useState<ElasticConfig>(getConfigWithDefaults(config));
  const [isInitialized, setIsInitialized] = useState(false);
  
  // 初始化时，如果 user 字段不存在，确保默认值被传递到父组件
  useEffect(() => {
    if (!isInitialized) {
      const configWithDefaults = getConfigWithDefaults(config);
      if (!config.user) {
        onValueChange(configWithDefaults);
      }
      setIsInitialized(true);
    }
  }, [isInitialized, config.user]);
  
  // 同步外部 config 变化到内部 state（但不触发 onValueChange 以避免循环）
  useEffect(() => {
    const updatedConfig = getConfigWithDefaults(config);
    setDb(updatedConfig);
  }, [config.endpoint, config.user, config.password]);
  
  return (
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="endpoint">Elasticsearch服务地址, 如http://xxx.com:9200</Label>
          <Input id="endpoint" value={db.endpoint || ''} onChange={(e) => {
            const newConfig = { ...db, endpoint: e.target.value };
            setDb(newConfig);
            onValueChange(newConfig);
          }} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="user">用户名</Label>
          <Input id="user" type="string" value={db.user || "elastic"} onChange={(e) => {
            const newConfig = { ...db, user: e.target.value };
            setDb(newConfig);
            onValueChange(newConfig);
          }} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">密码</Label>
          <Input id="password" type="password" value={db.password || ''} onChange={(e) => {
            const newConfig = { ...db, password: e.target.value };
            setDb(newConfig);
            onValueChange(newConfig);
          }} />
        </div>
    </div>
  );
};