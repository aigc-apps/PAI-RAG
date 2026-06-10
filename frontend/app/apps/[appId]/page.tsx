'use client';
import { use, useState, useEffect, useCallback } from "react";
import { ChatbotConfigCard, Chatbot } from "../chatbot_config";
import { FAQManagement } from "../faq_management";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { HeaderPortal } from "@/components/header-portal";
import { PageLoading } from "@/components/ui/loading";
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
import { REACT_PROMPT } from '@/app/common/prompts';
import { useI18n } from '@/app/providers/i18n';


export default function ViewChatApp(
    { params } : { params: Promise<{ appId: string }> }
) {
    const { t } = useI18n();

    // Default chatbot config
    const default_chat_config: Chatbot = {
        id: '',
        app_id: '',
        description: '',
        enable_search: false,
        mcp_ids: [],
        kb_ids: [],
        model_id: "",
        vision_model_id: null,
        updated_at: "",
        enable_agent: false,
        enable_chatdb: false,
        enable_faq: false,
        faq_config: null,
        enable_input_guardrail: false,
        enable_output_guardrail: false,
        guardrail_hint: t('apps.guardrailHint'),
        prompts: {
        react: REACT_PROMPT,
        }
    };
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
            toast.error(t('apps.fetchConfigError'));
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
                throw new Error(`${t('apps.saveFailed')}: ${errorText}`);
            }

            const data = await res.json();
            if (data.data) {
                setBotConfig(data.data);
            }
            if (shouldToast) {
                toast.success(t('messages.saveSuccess'));
            }
            return true;
        } catch (error: any) {
            console.error('保存失败:', error);
            toast.error(error.message || t('apps.saveFailed'));
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
        return <PageLoading className="h-full" />;
    }

    return (
        <div className="flex flex-col h-full min-h-0">
            <HeaderPortal>
                <Breadcrumb>
                    <BreadcrumbList>
                        <BreadcrumbItem>
                            <BreadcrumbLink asChild>
                                <Button
                                    variant="link"
                                    className="px-0 h-auto"
                                    onClick={() => router.push('/apps')}
                                >
                                    {t('sidebar.apps')}
                                </Button>
                            </BreadcrumbLink>
                        </BreadcrumbItem>
                        <BreadcrumbSeparator />
                        <BreadcrumbItem>
                            <BreadcrumbPage className="font-semibold">
                                {botConfig?.app_id || t('apps.edit')}
                            </BreadcrumbPage>
                        </BreadcrumbItem>
                    </BreadcrumbList>
                </Breadcrumb>
                {botConfig?.id && (
                    <Badge variant="secondary" className="text-[10px] font-mono bg-muted text-muted-foreground">
                        ID: {botConfig.id}
                    </Badge>
                )}
                {botConfig?.description && (
                    <span className="text-xs text-muted-foreground truncate max-w-[280px]">
                        {botConfig.description}
                    </span>
                )}
                <div className="ml-auto flex gap-2">
                    <Button variant="ghost" size="sm" onClick={() => router.push('/apps')}>
                        {t('common.cancel')}
                    </Button>
                    <Button size="sm" className="min-w-24" onClick={() => handleSave()} disabled={saving}>
                        {saving ? t('common.saving') : t('apps.saveApp')}
                    </Button>
                </div>
            </HeaderPortal>

            <div className="flex-1 min-h-0 overflow-y-auto">
                <ChatbotConfigCard
                    botConfig={botConfig}
                    onConfigChange={handleConfigChange}
                    onSave={handleSave}
                    saving={saving}
                    llms={llms}
                    mcps={mcps}
                    kbs={kbs}
                />
                <FAQManagement
                    appId={botConfig.app_id}
                    botConfig={botConfig}
                    onConfigChange={handleConfigChange}
                    onSave={(config) => handleSave(config, false)}
                    saving={saving}
                />
            </div>
        </div>
    );
}
