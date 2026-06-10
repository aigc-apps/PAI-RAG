'use client';
import { useState, useEffect, useCallback } from "react";
import { ChatbotConfigCard, Chatbot } from "../chatbot_config";
import { useTenantFetch } from "@/hooks/use-tenant-fetch";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { REACT_PROMPT } from '@/app/common/prompts';
import { useI18n } from '@/app/providers/i18n';
import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { HeaderPortal } from "@/components/header-portal";
import { PageLoading } from "@/components/ui/loading";



export default function CreateChatApp() {
    const { t } = useI18n();
// Default chatbot config for creating new app
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
    const router = useRouter();
    const { tenantFetch } = useTenantFetch();
    
    const [botConfig, setBotConfig] = useState<Chatbot>(default_chat_config);
    const [saving, setSaving] = useState(false);
    const [loading, setLoading] = useState(true);
    
    // Model configs
    const [llms, setLlms] = useState<LlmConfig[]>([]);
    const [mcps, setMcps] = useState<McpConfig[]>([]);
    const [kbs, setKbs] = useState<KbConfig[]>([]);

    // Fetch model configs
    useEffect(() => {
        const fetchConfigs = async () => {
            try {
                setLoading(true);
                const [llmRes, mcpRes, kbRes] = await Promise.all([
                    tenantFetch(`/api/config/llms`),
                    tenantFetch(`/api/config/mcps`),
                    tenantFetch(`/api/config/knowledgebases`)
                ]);

                const llmData = (await llmRes.json())?.data.items || [];
                setLlms(llmData);

                const mcpData = ((await mcpRes.json())?.data.items as McpConfig[]) || [];
                setMcps(mcpData);

                const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
                setKbs(kbData);
            } catch (error: any) {
                console.error('获取配置失败:', error);
                toast.error(t('apps.fetchConfigError'));
            } finally {
                setLoading(false);
            }
        };
        fetchConfigs();
    }, [tenantFetch]);

    // Handle config change
    const handleConfigChange = useCallback((updates: Partial<Chatbot>) => {
        setBotConfig(prev => ({ ...prev, ...updates }));
    }, []);

    // Create new chatapp
    const handleSave = useCallback(async () => {
        try {
            setSaving(true);
            const res = await tenantFetch(`/api/config/apps`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(botConfig),
            });

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(`创建失败: ${errorText}`);
            }

            toast.success(t('apps.createSuccess'));
            router.push('/apps');
            return true;
        } catch (error: any) {
            console.error('创建失败:', error);
            toast.error(error.message || t('apps.createFailedToast'));
            return false;
        } finally {
            setSaving(false);
        }
    }, [botConfig, tenantFetch, router]);

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
                                {t('apps.create')}
                            </BreadcrumbPage>
                        </BreadcrumbItem>
                    </BreadcrumbList>
                </Breadcrumb>
                <div className="ml-auto flex gap-2">
                    <Button variant="ghost" size="sm" onClick={() => router.push('/apps')}>
                        {t('common.cancel')}
                    </Button>
                    <Button size="sm" className="min-w-24" onClick={handleSave} disabled={saving}>
                        {saving ? t('common.saving') : t('apps.createApp')}
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
                    isCreate={true}
                />
            </div>
        </div>
    );
}
