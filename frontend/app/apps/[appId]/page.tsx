'use client';
import { use } from "react";
import { ChatbotConfigCard } from "../chatbot_config";


export default function ViewChatApp(
    { params } : { params: Promise<{ appId: string }> }
) {
    const { appId } = use(params);
    return (
        <ChatbotConfigCard chatbotId={appId}>
        </ChatbotConfigCard>
    )
}