'use client';
import { useState, useEffect, useCallback } from "react";
import { ChatbotConfigCard, Chatbot } from "../chatbot_config";
import { useTenantFetch } from "@/hooks/use-tenant-fetch";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { PLAN_PROMPT, ACT_PROMPT, ACT_WITH_PLAN_PROMPT, SUMMARY_PROMPT } from '@/app/common/prompts';
import { useI18n } from '@/app/providers/i18n';



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
    updated_at: "",
    enable_agent: false,
    enable_chatdb: false,
    enable_faq: false,
    faq_config: null,
    enable_input_guardrail: false,
    enable_output_guardrail: false,
    guardrail_hint: t('apps.guardrailHint'),
    prompts: {
      plan: PLAN_PROMPT,
      act: ACT_PROMPT,
      act_with_plan: ACT_WITH_PLAN_PROMPT,
      summary: SUMMARY_PROMPT,
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
        return <div className="flex items-center justify-center h-screen">{t('common.loading')}</div>;
    }

    return (
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
    );
}
