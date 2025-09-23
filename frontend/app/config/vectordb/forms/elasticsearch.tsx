// app/config/vectordb/forms/milvus.tsx

import React, { FC, useState } from "react";
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
  const [db, setDb] = useState<ElasticConfig>(config);
  return (
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="endpoint">Elasticsearch服务地址, 如http://xxx.com:9200</Label>
          <Input id="endpoint" value={config.endpoint || ''} onChange={(e) => {
            setDb({...db, endpoint: e.target.value});
            onValueChange({ ...db, endpoint: e.target.value })}
            } />
        </div>
        <div className="space-y-2">
          <Label htmlFor="user">用户名</Label>
          <Input id="user" type="string" value={db.user || "elastic"} onChange={(e) => {
            setDb({...db, user: e.target.value});
            onValueChange({ ...db, user: e.target.value })}
            } />
        </div>
        <div className="space-y-2">
          <Label htmlFor="user">密码</Label>
          <Input id="password" type="password" value={db.password} onChange={(e) => {
            setDb({...db, password: e.target.value});
            onValueChange({ ...db, password: e.target.value })}
            } />
        </div>
    </div>
  );
};