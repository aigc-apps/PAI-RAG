// app/config/vectordb/forms/milvus.tsx
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FC, useState, useEffect } from "react";
import { useI18n } from '@/app/providers/i18n';

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
  const { t } = useI18n();
  const [db, setDb] = useState<MilvusConfig>({
    ...config,
    database: config.database || 'default'
  });
  
  // Initialize to ensure database has value, and update when config changes
  useEffect(() => {
    const updatedDb = {
      ...config,
      database: config.database || 'default'
    };
    setDb(updatedDb);
    // Only notify parent component when database is missing to avoid infinite loop
    if (!config.database) {
      onValueChange(updatedDb);
    }
  }, [config]);
  
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="host">{t('config.vectordb.milvusHost')}</Label>
        <Input id="host" value={db.host || ''} onChange={(e) => {
          setDb({...db, host: e.target.value});
          onValueChange({ ...db, host: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="port">{t('config.vectordb.port')}</Label>
        <Input id="port" type="number" value={db.port || 19530} onChange={(e) => {
          setDb({...db, port: e.target.value});
          onValueChange({ ...db, port: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="database">{t('config.vectordb.database')}</Label>
        <Input id="database"  value={db.database || 'default'} onChange={(e) => {
          const value = e.target.value || 'default';
          setDb({...db, database: value});
          onValueChange({ ...db, database: value });
        }} />
      </div>
        <div className="space-y-2">
          <Label htmlFor="user">{t('config.vectordb.username')}</Label>
          <Input id="user" type="string" value={db.user || ''} onChange={(e) => {
            setDb({...db, user: e.target.value});
            onValueChange({ ...db, user: e.target.value })}
            } />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">{t('config.vectordb.password')}</Label>
          <Input id="password" type="password" value={db.password || ''} onChange={(e) => {
            setDb({...db, password: e.target.value});
            onValueChange({ ...db, password: e.target.value })}
            } />
        </div>
    </div>
  );
}