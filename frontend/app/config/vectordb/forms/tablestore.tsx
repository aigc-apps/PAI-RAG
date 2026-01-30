// app/config/vectordb/forms/tablestore.tsx
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FC, useState } from "react";
import { useI18n } from '@/app/providers/i18n';

export interface TablestoreConfig {
  endpoint: string;
  instance_name: string;
  ak: string;
  sk: string;
}

interface TablestoreConfigProps {
  config: TablestoreConfig;
  onValueChange: (config: TablestoreConfig) => void;
}

export const TablestoreForm: FC<TablestoreConfigProps> = ({
  config,
  onValueChange
}) => {
  const { t } = useI18n();
  const [db, setDb] = useState<TablestoreConfig>(config);
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="endpoint">{t('config.vectordb.tablestoreEndpoint')}</Label>
        <Input id="endpoint" value={db.endpoint || ''} onChange={(e) => {
          setDb({...db, endpoint: e.target.value});
          onValueChange({ ...db, endpoint: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="instance-name">{t('config.vectordb.instanceName')}</Label>
        <Input id="instance-name" value={db.instance_name || ''} onChange={(e) => {
          setDb({...db, instance_name: e.target.value});
          onValueChange({ ...db, instance_name: e.target.value });
        }} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="access_key_id">{t('config.vectordb.accessKeyId')}</Label>
        <Input id="access_key_id"  value={db.ak || ''} onChange={(e) => {
          setDb({...db, ak: e.target.value});
          onValueChange({ ...db, ak: e.target.value });
        }} />
      </div>
        <div className="space-y-2">
          <Label htmlFor="access_key_secret">{t('config.vectordb.accessKeySecret')}</Label>
          <Input id="access_key_secret" type="password" value={db.sk || ''} onChange={(e) => {
            setDb({...db, sk: e.target.value});
            onValueChange({ ...db, sk: e.target.value })}
            } />
        </div>
    </div>
  );
}