// components/vector-db-console/VectorDBConsole.tsx
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
import { toast } from "sonner";
import { is } from "date-fns/locale";
import { Loader2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

type DBType = "local" | "postgresql" | "milvus" | "elasticsearch" ;

const cache = new Map();

export default function VectorDBConsole() {
  const [dbType, setDbType] = useState<DBType>("local");
  const [db, setDb] = useState<Record<string, any>>({});

  const [loading, setLoading] = useState(false);
  const [connectionTesting, setConnectionTesting] = useState(false);
  
  useEffect(() => {
    const fetchConfig = async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/config/vectordb`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error('加载配置失败');

        const data = await res.json();
        setDbType(data.data.type);
        setDb({...data.data.config, password: data.data.config.encrypted_password ? '******': undefined});
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
        const res = await fetch(`/api/config/vectordb`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: dbType,
            config: { ...db, type: dbType, password: db.password === '******' ? '' :  db.password},
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
        console.log("链接测试： ", db);
        const res = await fetch(`/api/config/vectordb/connection_test`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: dbType,
            config: { ...db, type: dbType, password: db.password === '******' ? '' :  db.password},
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
      default:
        return <div>本地存储，无需额外配置。</div>;
    }
  };

  return (
    <Card className="w-full max-w-3xl mx-auto">
      <CardHeader>
        <CardTitle>向量数据库连接管理</CardTitle>
      </CardHeader>
      <CardContent>
        {
            loading ? <Skeleton className="h-12 w-12 rounded-full" /> :
        <div className="space-y-6">
          <div className="space-y-2">
            <Label>数据库类型</Label>
            <Select value={dbType} onValueChange={(v) => {
                cache.set(dbType, db);
                setDbType(v as DBType);
                setDb(cache.get(v as DBType) || {});
            }}>
              <SelectTrigger>
                <SelectValue placeholder="选择数据库类型" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="local">本地(Chroma)</SelectItem>
                <SelectItem value="postgresql">PostgreSQL</SelectItem>
                <SelectItem value="milvus">Milvus</SelectItem>
                <SelectItem value="elasticsearch">Elasticsearch</SelectItem>
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
                            测试中...
                          </>
                        ) : (
                          <>
                            测试连接
                          </>
                        )}</Button>
                        
            <Button onClick={saveConnection}>保存配置</Button>
          </div>
        </div>
        }
      </CardContent>
    </Card>
  );
}