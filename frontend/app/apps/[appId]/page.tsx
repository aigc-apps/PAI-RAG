'use client';
import { use, useState, useEffect } from "react";
import { ChatbotConfigCard } from "../chatbot_config";
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
import { Chatbot } from "../chatbot_config";
import { toast } from "sonner";

export default function ViewChatApp(
    { params } : { params: Promise<{ appId: string }> }
) {
    const { appId } = use(params);
    const router = useRouter();
    const { tenantFetch } = useTenantFetch();
    const [botConfig, setBotConfig] = useState<Chatbot | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchAppConfig();
    }, [appId]);

    const fetchAppConfig = async () => {
        try {
            setLoading(true);
            const res = await tenantFetch(`/api/config/apps?app_id=${appId}`);
            if (res.ok) {
                const data = await res.json();
                setBotConfig(data.data);
            }
        } catch (error: any) {
            console.error('获取应用配置失败:', error);
            toast.error('获取应用配置失败');
        } finally {
            setLoading(false);
        }
    };

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
                        <ChatbotConfigCard chatbotId={appId} />
                    </TabsContent>
                    <TabsContent value="faq" className="py-2">
                        {botConfig && <FAQManagement appId={botConfig.app_id} botConfig={botConfig} setBotConfig={setBotConfig} />}
                    </TabsContent>
                </Tabs>
            </div>
        </div>
    );
}