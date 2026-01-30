// app/config/vectordb/forms/hologres.tsx
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FC, useState } from "react";
import { useI18n } from '@/app/providers/i18n';

export interface HologresConfig {
  host: string;
  port: string;
  user: string;
  password: string;
  database: string;
}

interface HologresConfigProps {
  config: HologresConfig;
  onValueChange: (config: HologresConfig) => void;
}

export const HologresForm: FC<HologresConfigProps> = ({
  config,
  onValueChange
}) => {
  const { t } = useI18n();
  const [db, setDb] = useState<HologresConfig>(config);
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="host">{t('config.vectordb.hologresHost')}</Label>
        <Input id="host" value={db.host || ''} onChange={(e) => {
          setDb({...db, host: e.target.value});
          onValueChange({ ...db, host: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="port">{t('config.vectordb.port')}</Label>
        <Input id="port" type="number" value={db.port || 80} onChange={(e) => {
          setDb({...db, port: e.target.value});
          onValueChange({ ...db, port: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="database">{t('config.vectordb.database')}</Label>
        <Input id="database"  value={db.database || ''} onChange={(e) => {
          setDb({...db, database: e.target.value});
          onValueChange({ ...db, database: e.target.value });
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