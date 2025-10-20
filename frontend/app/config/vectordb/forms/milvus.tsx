// app/config/vectordb/forms/milvus.tsx
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FC, useState } from "react";

export interface MilvusConfig {
  host: string;
  port: string;
  user: string;
  password: string;
  database: string;
}

interface MilvusConfigProps {
  config: MilvusConfig;
  onValueChange: (config: MilvusConfig) => void;
}

export const MilvusForm: FC<MilvusConfigProps> = ({
  config,
  onValueChange
}) => {
  const [db, setDb] = useState<MilvusConfig>(config);
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="host">Milvus主机地址</Label>
        <Input id="host" value={db.host || ''} onChange={(e) => {
          setDb({...db, host: e.target.value});
          onValueChange({ ...db, host: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="port">端口</Label>
        <Input id="port" type="number" value={db.port || 19530} onChange={(e) => {
          setDb({...db, port: e.target.value});
          onValueChange({ ...db, port: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="database">数据库名</Label>
        <Input id="database"  value={db.database || 'default'} onChange={(e) => {
          setDb({...db, database: e.target.value});
          onValueChange({ ...db, database: e.target.value });
        }} />
      </div>
        <div className="space-y-2">
          <Label htmlFor="user">用户名</Label>
          <Input id="user" type="string" value={db.user || ''} onChange={(e) => {
            setDb({...db, user: e.target.value});
            onValueChange({ ...db, user: e.target.value })}
            } />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">密码</Label>
          <Input id="password" type="password" value={db.password || ''} onChange={(e) => {
            setDb({...db, password: e.target.value});
            onValueChange({ ...db, password: e.target.value })}
            } />
        </div>
    </div>
  );
}