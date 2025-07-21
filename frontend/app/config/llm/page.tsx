"use client";

import React, { useState, useEffect } from "react";
import {
  TrashIcon,
  SettingsIcon,
  Edit,
  EyeIcon,
  EyeOffIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import * as Toast from "@radix-ui/react-toast";
import { Switch } from "@/components/ui/switch";
import { v4 as uuidv4 } from "uuid";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { ChevronRight, Plus } from "lucide-react";

export const MaskedApiKey = ({ apiKey }: { apiKey: string }) => {
  const maskApiKey = (
    apiKey: string,
    prefixLength = 4,
    suffixLength = 3,
  ): string => {
    if (apiKey.length <= prefixLength + suffixLength) return apiKey; // 如果长度不够，直接返回原值
    return `${apiKey.slice(0, prefixLength)}*****${apiKey.slice(
      -suffixLength,
    )}`;
  };
  const [showFull, setShowFull] = useState(false);

  const toggleShow = () => setShowFull((prev) => !prev);

  return (
    <div className="flex items-center space-x-2">
      <span className="text-gray-700">
        {showFull ? apiKey : maskApiKey(apiKey)}
      </span>
      <button
        onClick={toggleShow}
        className="text-sm text-black-500 hover:text-black-700"
      >
        {showFull ? (
          <EyeOffIcon className="w-4 h-4" />
        ) : (
          <EyeIcon className="w-4 h-4" />
        )}
      </button>
    </div>
  );
};

interface LlmConfig {
  id: string;
  model_id: string;
  source: string;
  model: string;
  api_key: string;
  base_url: string;
  max_context: number;
  enabled: boolean;
  vision_support: boolean;
}

interface EmbConfig {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
  api_key: string;
  endpoint: string;
  dimension: number;
  embed_batch_size: number;
}

export default function ModelConfigPage() {
  const [llmconfigs, setLlmConfigs] = useState<LlmConfig[]>([]); // 存储 LLM 配置
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(""); // 错误信息

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const modelSizePerPage = 3;

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        const res = await fetch(
          `http://localhost:${port}/v1/config/llms?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error("获取LLM模型列表失败");
        const json_data = await res.json();
        const data = json_data.data.items;
        setLlmConfigs(data); // 合并
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || "加载失败");
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, []);

  return (
    <div id="llm">
      <div className="flex flex-col h-screen p-6 space-y-6">
        {/* 顶部标题栏 */}
        <div className="flex justify-between items-center h-1/10">
          <h1 className="text-2xl font-bold">模型</h1>
        </div>

        {/* 卡片容器 */}
        <div className="h-4/5">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
            <Tabs defaultValue="llms">
              <TabsList className="py-4 bg-muted rounded-lg flex-none">
                <TabsTrigger value="llms" className="p-4">
                  LLM
                </TabsTrigger>
                <TabsTrigger value="embeddings" className="p-4">
                  Embedding
                </TabsTrigger>
              </TabsList>
              <TabsContent value="llms" className="py-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
                  <Button
                    className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 w-40"
                    // onClick={() => setActiveTab("/knowledgebase/create")}
                  >
                    <Plus className="w-6 h-6" />
                    新增LLM
                  </Button>
                  {llmconfigs.map((llm) => (
                    <Card
                      key={llm.id}
                      className="flex flex-col border rounded-lg shadow-sm h-full"
                    >
                      <CardHeader>
                        <CardTitle className="text-sm font-medium">
                          {llm.model_id}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <p>{llm.source}</p>
                      </CardContent>
                      <CardFooter className="mt-auto pt-0 flex justify-end">
                        <Button
                          variant="link"
                          // onClick={() => deleteKnowledgebase(base.id)}
                          className="text-sm text-primary text-red-600 hover:text-primary/80 underline-offset-4 hover:underline"
                        >
                          删除
                        </Button>

                        <Button
                          variant="link"
                          className="text-sm text-primary text-blue-600 hover:text-primary/80 underline-offset-4 hover:underline"
                          // onClick={() =>
                          //   setActiveTab(`/knowledgebase/details/${base.id}`)
                          // }
                        >
                          查看详情 <ChevronRight className="ml-1" size={16} />
                        </Button>
                      </CardFooter>
                    </Card>
                  ))}
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
