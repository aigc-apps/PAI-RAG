'use client';
import { use, useState, useEffect, useCallback } from "react";
import { ChatbotConfigCard, Chatbot } from "../chatbot_config";
import { FAQManagement } from "../faq_management";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { useRouter } from "next/navigation";
import { useTenantFetch } from "@/hooks/use-tenant-fetch";
import { toast } from "sonner";
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { PLAN_PROMPT, ACT_PROMPT, ACT_WITH_PLAN_PROMPT, SUMMARY_PROMPT } from '@/app/common/prompts';

// Default chatbot config
const default_chat_config: Chatbot = {
  id: '',
  app_id: '',
  description: '',
  enable_search: false,
  mcp_ids: [],
  kb_ids: [],
  model_id: "",
  updated_at: "",
  enable_agent: false,
  enable_chatdb: false,
  enable_faq: false,
  faq_config: null,
  enable_input_guardrail: false,
  enable_output_guardrail: false,
  guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
  prompts: {
    plan: PLAN_PROMPT,
    act: ACT_PROMPT,
    act_with_plan: ACT_WITH_PLAN_PROMPT,
    summary: SUMMARY_PROMPT,
  }
};

export default function ViewChatApp(
    { params } : { params: Promise<{ appId: string }> }
) {
    const { appId } = use(params);
    const router = useRouter();
    const { tenantFetch } = useTenantFetch();
    
    // Centralized state management
    const [botConfig, setBotConfig] = useState<Chatbot>(default_chat_config);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    
    // Model configs
    const [llms, setLlms] = useState<LlmConfig[]>([]);
    const [mcps, setMcps] = useState<McpConfig[]>([]);
    const [kbs, setKbs] = useState<KbConfig[]>([]);

    // Fetch all configs
    const fetchAllConfigs = useCallback(async () => {
        try {
            setLoading(true);
            const [llmRes, mcpRes, kbRes, botRes] = await Promise.all([
                tenantFetch(`/api/config/llms`),
                tenantFetch(`/api/config/mcps`),
                tenantFetch(`/api/config/knowledgebases`),
                tenantFetch(`/api/config/apps?app_id=${appId}`)
            ]);

            const llmData = (await llmRes.json())?.data.items || [];
            setLlms(llmData);

            const mcpData = ((await mcpRes.json())?.data.items as McpConfig[]) || [];
            setMcps(mcpData);

            const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
            setKbs(kbData);

            const botData = await botRes.json();
            if (botData.data) {
                // Filter out invalid kb_ids and mcp_ids
                botData.data.kb_ids = botData.data.kb_ids?.filter(
                    (kb_id: string) => kbData.some((kb: any) => kb.id === kb_id)
                ) || [];
                botData.data.mcp_ids = botData.data.mcp_ids?.filter(
                    (mcp_id: string) => mcpData.some((mcp: any) => mcp.id === mcp_id)
                ) || [];
                setBotConfig(botData.data);
            }
        } catch (error: any) {
            console.error('获取配置失败:', error);
            toast.error('获取配置失败');
        } finally {
            setLoading(false);
        }
    }, [appId, tenantFetch]);

    useEffect(() => {
        fetchAllConfigs();
    }, [fetchAllConfigs]);

    // Centralized save function
    const handleSave = useCallback(async (updatedConfig?: Partial<Chatbot>, shouldToast: boolean=true) => {
        const configToSave = updatedConfig ? { ...botConfig, ...updatedConfig } : botConfig;
        
        try {
            setSaving(true);
            const res = await tenantFetch(`/api/config/apps/${configToSave.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(configToSave),
            });

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(`保存失败: ${errorText}`);
            }

            const data = await res.json();
            if (data.data) {
                setBotConfig(data.data);
            }
            if (shouldToast) {
                toast.success('保存成功');
            }
            return true;
        } catch (error: any) {
            console.error('保存失败:', error);
            toast.error(error.message || '保存失败');
            return false;
        } finally {
            setSaving(false);
        }
    }, [botConfig, tenantFetch]);

    // Handle config change from child components
    const handleConfigChange = useCallback((updates: Partial<Chatbot>) => {
        setBotConfig(prev => ({ ...prev, ...updates }));
    }, []);

    // Navigate back after save
    if (loading) {
        return <div className="flex items-center justify-center h-screen">加载中...</div>;
    }

    return (
        <div className="flex flex-col h-screen pt-0 space-y-0">
            <div className="absolute top-2 left-12 py-0 flex items-center z-10">
                <Breadcrumb>
                    <BreadcrumbList>
                        <BreadcrumbItem>
                            <BreadcrumbLink asChild>
                                <Button
                                    variant="link"
                                    className="px-0"
                                    onClick={() => router.push('/apps')}
                                >
                                    应用
                                </Button>
                            </BreadcrumbLink>
                        </BreadcrumbItem>
                        <BreadcrumbSeparator />
                        <BreadcrumbItem>
                            <BreadcrumbPage>{botConfig?.app_id || '应用编辑'}</BreadcrumbPage>
                        </BreadcrumbItem>
                    </BreadcrumbList>
                </Breadcrumb>
                <div className="flex gap-2 items-center ml-4">
                    <Badge variant="secondary" className="text-xs bg-muted text-muted-foreground">
                        ID: {botConfig?.id || ''}
                    </Badge>
                    {botConfig?.description && (
                        <Badge variant="secondary" className="text-xs bg-muted text-muted-foreground max-w-[200px] truncate">
                            {botConfig.description}
                        </Badge>
                    )}
                </div>
            </div>
            <div className="flex-1 overflow-y-auto px-2 py-6">
                <Tabs defaultValue="settings" className="h-full flex flex-col">
                    <TabsList className="py-0 bg-muted rounded-lg flex-none">
                        <TabsTrigger value="settings" className="py-1 px-2">
                            <span className="text-xs">应用设置</span>
                        </TabsTrigger>
                        <TabsTrigger value="faq" className="py-1 px-2">
                            <span className="text-xs">FAQ管理</span>
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="settings" className="py-2">
                        <ChatbotConfigCard
                            botConfig={botConfig}
                            onConfigChange={handleConfigChange}
                            onSave={handleSave}
                            saving={saving}
                            llms={llms}
                            mcps={mcps}
                            kbs={kbs}
                        />
                    </TabsContent>
                    <TabsContent value="faq" className="py-2">
                        <FAQManagement
                            appId={botConfig.app_id}
                            botConfig={botConfig}
                            onConfigChange={handleConfigChange}
                            onSave={(config) => handleSave(config, false)}
                            saving={saving}
                        />
                    </TabsContent>
                </Tabs>
            </div>
        </div>
    );
}
