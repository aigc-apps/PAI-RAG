'use client';
import React from 'react';
import { useState, useEffect, use, useRef } from "react";
import { useRouter } from "next/navigation";
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { EvalConfig } from '@/app/evaluation/[evalId]/page';
import { Chatbot } from "@/app/apps/chatbot_config";
import { ChevronDownIcon, Terminal } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { toast } from 'sonner';


const default_eval_config = {
  id: '',
  name: '',
  description: '',
  chatbot_id: '',
  chatbot_config: {
    model_id: "",
    mcp_ids: [],
    kb_ids: [],
    enable_search: false,
    enable_vision: false,
    enable_agent: false,
    enable_input_guardrail: false,
    enable_output_guardrail: false,
    guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
  }
};

export default function EvalExpDetailsPage(
    { params }: { params: Promise<{ evalId : string }> }
) {
    const { evalId } = use(params);
    const router = useRouter();
    const [evalConfig, setEvalConfig] = useState<EvalConfig>(default_eval_config);
    const [chatbots, setChatbots] = useState<Chatbot[]>([]);
    const [llms, setLlms] = useState<LlmConfig[]>([]);
    const [mcps, setMcps] = useState<McpConfig[]>([]);
    const [kbs, setKbs] = useState<KbConfig[]>([]);
    const [selectedKbNames, setSelectedKbNames] = useState<string[]>([]);
    const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>([]);
    const [saveErrorMsg, setSaveErrorMsg] = useState('');
    const isCreate: boolean = evalId === undefined || evalId === '';
    console.log("isCreate", isCreate)

    useEffect(() => {
        const fetchEvalConfigs = async () => {
          try {
            const [evalRes, chatbotRes, llmRes, mcpRes, kbRes] = await Promise.all([
                fetch(`/api/config/evaluation/${evalId}`),
                fetch(`/api/config/apps`),
                fetch(`/api/config/llms`),
                fetch(`/api/config/mcps`),
                fetch(`/api/config/knowledgebases`),
            ]);
    
            if (!evalRes.ok) throw new Error('获取评估任务详情失败');
            const json_data = await evalRes.json();
            const evalData = json_data.data;
            console.log('evalData:', evalData);
            setEvalConfig(evalData);

            const chatbotData = (await chatbotRes.json())?.data.items || [];
            console.log('chatbotData', chatbotData);
            setChatbots([...chatbotData]);

            const llmData = (await llmRes.json())?.data.items || [];
            console.log('llmData', llmData);
            setLlms([...llmData]);

            const mcpData =
            ((await mcpRes.json())?.data.items as McpConfig[]) || [];
            console.log('mcpData', mcpData);
            setMcps([...mcpData]);

            const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
            console.log('kbData', kbData);
            setKbs([...kbData]);

            // const kbnames = kbData
            //     .filter((item) => evalData.chatbot_config.kb_ids.includes(item.id))
            //     .map((item) => item.name);
            // setSelectedKbNames([...kbnames]);
            // console.log('selectedKbNames', kbnames);

            // const mcpnames = mcpData
            //     .filter((item) => evalData.chatbot_config.mcp_ids.includes(item.id))
            //     .map((item) => item.name);
            // setSelectedMcpNames([...mcpnames]);
            // console.log('selectedMcpNames', mcpnames);

            // console.log('selectedKbNames', kbnames);
            // console.log('selectedMcpNames', mcpnames);
    
          } catch (err: any) {
            toast.error(err.message);
          }
        };
        fetchEvalConfigs();
      }, []);

    if(!evalConfig){
        return <div className="p-6">加载中...</div>;
    }

    const handleSaveEvalConfig = async () => {
        console.log('保存评估设置:', evalConfig);
        const submit_url = isCreate
        ? `/api/config/evaluation`
        : `/api/config/evaluation/${evalConfig.id}`;
        const updateMethod = isCreate ? 'POST' : 'PUT';
        try {
        const res = await fetch(submit_url, {
            method: updateMethod,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(evalConfig), // 包装为数组
        });

        if (!res.ok) throw new Error(`保存评估任务失败: ${await res.text()}`);
        router.push(`/evaluation/${evalConfig.id}`);
        setSaveErrorMsg('');
        } catch (err: any) {
        console.log('保存评估任务失败', err.message);
        setSaveErrorMsg(err.message);
        }
    };

    return (
        <div className="flex flex-col h-screen px-6 py-4 space-y-6">
            <div className="flex-none">
                <div className="p-2 space-y-2">
                    <div className="mb-2 flex items-center gap-2">
                        {/* 面包屑导航 */}
                        <Breadcrumb>
                            <BreadcrumbList>
                                <BreadcrumbItem>
                                    <BreadcrumbLink asChild>
                                        <Button
                                            variant="link"
                                            className="px-0"
                                            onClick={() => router.push('/evaluation')}
                                        >
                                            评估
                                        </Button>
                                    </BreadcrumbLink>
                                </BreadcrumbItem>
                                <BreadcrumbSeparator />
                                <BreadcrumbItem>
                                    <Button
                                        variant="link"
                                        className="px-0"
                                        onClick={() => router.push(`/evaluation/${evalId}`)}
                                    >
                                        {evalConfig.name}
                                    </Button>
                                </BreadcrumbItem>
                                <BreadcrumbSeparator />
                                <BreadcrumbItem>
                                    <BreadcrumbPage>settings</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                    <div className="flex justify-between items-center px-2">
                        <div>
                            <h1 className="text-2xl font-bold">任务设置</h1>
                            <div className="text-sm text-muted-foreground mt-1">
                                评估任务设置
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="px-2 max-w-6xl">
                <div className="grid gap-4 py-6 px-6">
                    <div className="space-y-2">
                        <Label htmlFor="eval-id">
                        实验名称 <span className="text-destructive">*</span>
                        </Label>
                        <Input
                        id="evalid"
                        value={evalConfig.name}
                        onChange={(e) =>
                            setEvalConfig((prev) => ({ ...prev, name: e.target.value }))
                        }
                        placeholder="请输入实验名称, 如GAIA"
                        required
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="description">描述</Label>
                        <Textarea
                        id="description"
                        value={evalConfig.description}
                        onChange={(e) =>
                            setEvalConfig((prev) => ({
                            ...prev,
                            description: e.target.value,
                            }))
                        }
                        placeholder="评估实验描述（可选）"
                        rows={3}
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="description">实验设置</Label>
                        <Tabs defaultValue="from-apps" className='space-y-2'>
                        <TabsList className="py-4 bg-muted rounded-lg flex-none">
                            <TabsTrigger value="from-apps" className="p-4">
                            从已有应用选择
                            </TabsTrigger>
                            <TabsTrigger value="customized" className="p-4">
                            自定义
                            </TabsTrigger>
                        </TabsList>
                        <TabsContent value="from-apps" className="py-4">
                            <div className="grid gap-4 py-1 px-6">
                            <div className="flex">
                                    <Label htmlFor="basemodel" className="w-[90px]">
                                      应用选择 <span className="text-destructive">*</span>{' '}
                                    </Label>
                                    <div className="px-6">
                                      {chatbots.length > 0 ? (
                                        <Select
                                          value={evalConfig.chatbot_id}
                                          onValueChange={(value) =>
                                            setEvalConfig((prev) => ({
                                              ...prev,
                                              chatbot_id: value,
                                            }))
                                          }
                                        >
                                          <SelectTrigger>
                                            <SelectValue placeholder="请选择应用" />
                                          </SelectTrigger>
                                          <SelectContent>
                                            {chatbots.map((cb) => (
                                              <SelectItem key={cb.id} value={cb.app_id}>
                                                {cb.app_id}
                                              </SelectItem>
                                            ))}
                                          </SelectContent>
                                        </Select>
                                      ) : (
                                        <div>
                                          <p className="text-sm text-muted-foreground">尚未配置应用</p>
                                          <Button
                                            variant="outline"
                                            onClick={() => {
                                              router.push('/apps/create');
                                            }}
                                          >
                                            前往添加
                                          </Button>
                                        </div>
                                      )}
                                    </div>
                                  </div>
                            </div>
                        </TabsContent>
                        {/* <TabsContent value="customized" className="py-4">
                            <div className="grid gap-4 py-1 px-6">
                            <div className="flex">
                                <Label htmlFor="basemodel" className="w-[90px]">
                                基模型选择 <span className="text-destructive">*</span>{' '}
                                </Label>
                                <div className="px-6">
                                {llms.length > 0 ? (
                                    <Select
                                    value={evalConfig?.chatbot_config?.model_id}
                                    onValueChange={(value) =>
                                        setEvalConfig((prev) => ({
                                        ...prev,
                                        model_id: value,
                                        }))
                                    }
                                    >
                                    <SelectTrigger>
                                        <SelectValue placeholder="请选择基模型" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {llms.map((llm) => (
                                        <SelectItem key={llm.id} value={llm.model_id}>
                                            {llm.model_id}
                                        </SelectItem>
                                        ))}
                                    </SelectContent>
                                    </Select>
                                ) : (
                                    <div>
                                    <p className="text-sm text-muted-foreground">尚未配置大模型</p>
                                    <Button
                                        variant="outline"
                                        onClick={() => {
                                        router.push('/config/model/llm');
                                        }}
                                    >
                                        前往添加
                                    </Button>
                                    </div>
                                )}
                                </div>
                            </div>
                            <div className="flex gap-6">
                                <Label htmlFor="enable_search" className="w-[90px]">
                                启用联网搜索
                                </Label>
                                <Switch
                                id="enable_search"
                                checked={evalConfig?.chatbot_config?.enable_search}
                                onCheckedChange={(checked) => {
                                    setEvalConfig((prev) => ({
                                        ...prev,
                                        chatbot_config: {
                                        ...prev.chatbot_config,
                                        enable_search: Boolean(checked),
                                        },
                                    }));
                                }}
                                />
                            </div>
                            <div className="flex gap-6">
                                <Label htmlFor="enable_agent" className="w-[90px]">
                                Agentic模式
                                </Label>
                                <Switch
                                id="enable_agent"
                                checked={evalConfig?.chatbot_config?.enable_agent}
                                onCheckedChange={(checked) => {
                                    setEvalConfig((prev) => ({
                                        ...prev,
                                        chatbot_config: {
                                        ...prev.chatbot_config,
                                        enable_agent: Boolean(checked),
                                        },
                                    }));
                                }}
                                />
                            </div>
                            <div className="flex">
                                <Label htmlFor="kb_selection" className="w-[90px]">
                                知识库选择
                                </Label>
                                <div className="pl-6 pr-6">
                                {kbs.length > 0 ? (
                                    <DropdownMenu modal={true}>
                                    <DropdownMenuTrigger asChild>
                                        <Button
                                        variant="outline"
                                        className="text-sm text-muted-foreground"
                                        >
                                        已选{evalConfig?.chatbot_config?.kb_ids.length || 0}个，可多选 <ChevronDownIcon />
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="w-56">
                                        <DropdownMenuLabel>知识库</DropdownMenuLabel>
                                        <DropdownMenuSeparator />
                                        {kbs.map((kb) => (
                                        <DropdownMenuCheckboxItem
                                            key={kb.id}
                                            checked={evalConfig.chatbot_config.kb_ids.includes(kb.id)}
                                            // onCheckedChange={(checked) =>
                                            //   handleKbSelect(kb.id, kb.name, checked)
                                            // }
                                            onSelect={(e) => e.preventDefault()}
                                        >
                                            {kb.name}
                                        </DropdownMenuCheckboxItem>
                                        ))}
                                    </DropdownMenuContent>
                                    </DropdownMenu>
                                ) : (
                                    <div>
                                    <p className="text-sm text-muted-foreground">尚未配置知识库</p>
                                    </div>
                                )}
                                </div>
                                {selectedKbNames.length > 0 && (
                                <div className="flex gap-1.5 items-center">
                                    {selectedKbNames.map((name) => (
                                    <Badge variant="secondary" className="h-6" key={name}>
                                        {name}
                                    </Badge>
                                    ))}
                                </div>
                                )}
                            </div>
                            <div className="flex">
                                <Label htmlFor="mcp_selection" className="w-[90px]">
                                MCP选择
                                </Label>
                                <div className="pl-6 pr-6">
                                {mcps.length > 0 ? (
                                    <DropdownMenu modal={true}>
                                    <DropdownMenuTrigger asChild>
                                        <Button
                                        variant="outline"
                                        className="text-sm text-muted-foreground"
                                        >
                                        已选{evalConfig?.chatbot_config?.mcp_ids.length}个，可多选 <ChevronDownIcon />
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="w-56">
                                        <DropdownMenuLabel>MCP</DropdownMenuLabel>
                                        <DropdownMenuSeparator />
                                        {mcps.map((mcp) => (
                                        <DropdownMenuCheckboxItem
                                            key={mcp.id}
                                            checked={evalConfig.chatbot_config.mcp_ids.includes(mcp.id)}
                                            // onCheckedChange={(checked) =>
                                            //   handleMcpSelect(mcp.id, mcp.name, checked)
                                            // }
                                            onSelect={(e) => e.preventDefault()}
                                        >
                                            {mcp.name}
                                        </DropdownMenuCheckboxItem>
                                        ))}
                                    </DropdownMenuContent>
                                    </DropdownMenu>
                                ) : (
                                    <div>
                                    <p className="text-sm text-muted-foreground">尚未配置MCP</p>
                                    </div>
                                )}
                                </div>
                                {selectedMcpNames.length > 0 && (
                                <div className="flex gap-1.5 items-center">
                                    {selectedMcpNames.map((name) => (
                                    <Badge variant="secondary" className="h-6" key={name}>
                                        {name}
                                    </Badge>
                                    ))}
                                </div>
                                )}
                            </div>
                            </div>

                        </TabsContent> */}

                        </Tabs>
                    </div>
                    {saveErrorMsg && (
                        <Alert variant="destructive">
                        <Terminal />
                        <AlertTitle>{isCreate ? '创建应用失败' : '保存应用失败'}</AlertTitle>
                        <AlertDescription>{saveErrorMsg}</AlertDescription>
                        </Alert>
                    )}
                    <div className="pt-8 flex gap-6">
                        <Button
                        variant="secondary"
                        className="w-20"
                        onClick={() => {
                            router.push(`/evaluation/${evalConfig.id}`);
                        }}
                        >
                        取消
                        </Button>

                        <Button
                        className="w-20"
                        onClick={() => {
                          handleSaveEvalConfig();
                        }}
                        >
                        {isCreate ? '创建' : '保存'}
                        </Button>
                    </div>
                    </div>    
            </div>
        </div>
    );
}
