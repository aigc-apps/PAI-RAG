// app/config/vectordb/forms/opensearch.tsx
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FC, useState } from "react";

export interface OpensearchConfig {
  endpoint: string;
  instance_id: string;
  username: string;
  password: string;
}

interface OpensearchConfigProps {
  config: OpensearchConfig;
  onValueChange: (config: OpensearchConfig) => void;
}

export const OpensearchForm: FC<OpensearchConfigProps> = ({
  config,
  onValueChange
}) => {
  const [db, setDb] = useState<OpensearchConfig>(config);
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="endpoint">Opensearch Endpoint</Label>
        <Input id="endpoint" value={db.endpoint || ''} onChange={(e) => {
          setDb({...db, endpoint: e.target.value});
          onValueChange({ ...db, endpoint: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="instance-id">实例ID</Label>
        <Input id="instance-id" value={db.instance_id || ''} onChange={(e) => {
          setDb({...db, instance_id: e.target.value});
          onValueChange({ ...db, instance_id: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="username">用户名</Label>
        <Input id="username"  value={db.username || ''} onChange={(e) => {
          setDb({...db, username: e.target.value});
          onValueChange({ ...db, username: e.target.value });
        }} />
      </div>
        <div className="space-y-2">
          <Label htmlFor="password">密码</Label>
          <Input id="user" type="password" value={db.password || ''} onChange={(e) => {
            setDb({...db, password: e.target.value});
            onValueChange({ ...db, password: e.target.value })}
            } />
        </div>
    </div>
  );
}