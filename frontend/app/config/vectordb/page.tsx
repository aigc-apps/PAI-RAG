"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Key, useEffect, useState } from "react";
import { PostgresqlConfig, PostgresqlForm } from "./forms/postgresql";
import { MilvusConfig, MilvusForm } from "./forms/milvus";
import { ElasticConfig, ElasticsearchForm } from "./forms/elasticsearch";
import { HologresConfig, HologresForm } from "./forms/hologres";
import { OpensearchConfig, OpensearchForm } from "./forms/opensearch";
import { TablestoreConfig, TablestoreForm } from "./forms/tablestore";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

type DBType = "local" | "postgresql" | "milvus" | "elasticsearch" | "hologres" | "opensearch" | "tablestore";

const cache = new Map();

export default function VectorDBConsole() {
  const { t } = useI18n();
  const [dbType, setDbType] = useState<DBType>("local");
  const [db, setDb] = useState<Record<string, any>>({});

  const [loading, setLoading] = useState(false);
  const [connectionTesting, setConnectionTesting] = useState(false);
  const { tenantFetch } = useTenantFetch();
  useEffect(() => {
    const fetchConfig = async () => {
      setLoading(true);
      try {
        const res = await tenantFetch(`/api/config/vectordb`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error(t('config.loadError'));

        const data = await res.json();
        setDbType(data.data.type);
        setDb({
          ...data.data.config,
          password: data.data.config.encrypted_password ? '******': undefined,
          sk: data.data.config.encrypted_sk ? '******': undefined });
      } catch (err: any) {
        toast.error(err.message);
      }
      finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);


  const saveConnection = async () => { 
      try {
        const config: Record<string, any> = {
          ...db,
          type: dbType,
          password: db.password === '******' ? '' :  db.password,
          sk: db.sk === '******' ? '' : db.sk
        };
        // If it's milvus type and database field is missing, set default value
        if (dbType === 'milvus' && !config.database) {
          config.database = 'default';
        }
        const res = await tenantFetch(`/api/config/vectordb`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: dbType,
            config,
          }),
        });

        const response = await res.json();
        if (response.code === 200) 
        {
            toast.success(response.message);
        }
        else {
            toast.error(response.message);
        }
      } catch (err: any) {
        toast.error(err.message);
      }

 };


  const testConnection = async () => { 
      setConnectionTesting(true);
      try {
        const config: Record<string, any> = {
          ...db,
          type: dbType,
          password: db.password === '******' ? '' :  db.password,
          sk: db.sk === '******' ? '' : db.sk
        };
        // If it's milvus type and database field is missing, set default value
        if (dbType === 'milvus' && !config.database) {
          config.database = 'default';
        }
        const res = await tenantFetch(`/api/config/vectordb/connection_test`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: dbType,
            config,
          }),
        });

        const response = await res.json();
        if (response.code === 200) 
        {
            toast.success(response.message);
        }
        else {
            toast.error(response.message);
        }
      } catch (err: any) {
        toast.error(err.message);
      }
      finally {
        setConnectionTesting(false);
      }

 };
  const renderForm = () => {
    switch (dbType) {
      case "postgresql":
        return <PostgresqlForm config={db as PostgresqlConfig} onValueChange={setDb} />;
      case "milvus":
        return <MilvusForm config={db as MilvusConfig} onValueChange={setDb} />;
      case "elasticsearch":
        return <ElasticsearchForm config={db as ElasticConfig} onValueChange={setDb} />;
      case "hologres":
        return <HologresForm config={db as HologresConfig} onValueChange={setDb} />;
      case "opensearch":
        return <OpensearchForm config={db as OpensearchConfig} onValueChange={setDb} />;
      case "tablestore":
        return <TablestoreForm config={db as TablestoreConfig} onValueChange={setDb} />;
      default:
        return <div>{t('config.vectordb.localNoConfig')}</div>;
    }
  };

  return (
    <Card className="w-full max-w-3xl mx-auto">
      <CardHeader>
        <CardTitle>{t('config.vectordb.title')}</CardTitle>
      </CardHeader>
      <CardContent>
        {
            loading ? <Skeleton className="h-12 w-12 rounded-full" /> :
        <div className="space-y-6">
          <div className="space-y-2">
            <Label>{t('config.vectordb.dbType')}</Label>
            <Select value={dbType} onValueChange={(v) => {
                cache.set(dbType, db);
                setDbType(v as DBType);
                setDb(cache.get(v as DBType) || {});
            }}>
              <SelectTrigger>
                <SelectValue placeholder={t('config.vectordb.selectDbType')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="local">{t('config.vectordb.localChroma')}</SelectItem>
                <SelectItem value="postgresql">PostgreSQL</SelectItem>
                <SelectItem value="milvus">Milvus</SelectItem>
                <SelectItem value="elasticsearch">Elasticsearch</SelectItem>
                <SelectItem value="hologres">Hologres</SelectItem>
                <SelectItem value="opensearch">Opensearch</SelectItem>
                <SelectItem value="tablestore">Tablestore</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Separator />

          {renderForm()}

          <div className="flex gap-4 pt-4">
            <Button variant="outline" onClick={testConnection}> 
                {connectionTesting ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            {t('config.vectordb.testing')}
                          </>
                        ) : (
                          <>{t('config.vectordb.testConnection')}</>
                        )}</Button>
                        
            <Button onClick={saveConnection}>{t('common.save')}</Button>
          </div>
        </div>
        }
      </CardContent>
    </Card>
  );
}