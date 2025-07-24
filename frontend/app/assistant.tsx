"use client";

import React, { useState, useEffect } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
// import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import ModelSelector from "@/components/model-selector/index";
import ToolUIWrapper from "@/components/assistant-ui/tool-ui";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import ModelConfigPage from "./config/model/page";
import McpConfig from "./config/mcp/page";
import SearchConfig from "./config/search/page";
import { useMemo } from "react";
import TracingConfig from "./config/tracing/page";
import KnowledgeBase from "./knowledgebase/page";
import { usePathname } from "next/navigation";
import KnowledgeBaseDetailPage from "./knowledgebase/details/page";
import KnowledgeBaseCreatePage from "./knowledgebase/create/page";
import { UploadAttachmentAdapter } from "./attachments/upload_attachment_adapter";
import { usePaiChatThreadRuntime } from "./runtime/usePaiChatThreadRuntime";
import KnowledgeBaseFileChunksPage from "./knowledgebase/chunks/page";
export const Assistant = () => {
  // LLM 配置状态
  const [llmConfig, setLlmConfig] = useState({
    id: "",
    source: "",
    model_id: "",
  });

  // 页面加载时拉取 LLM 配置
  useEffect(() => {
    const fetchLLMConfig = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8680;
        console.log("assistant BACKEND_PORT", port);
        const res = await fetch(`http://localhost:${port}/v1/config/llms`);
        if (!res.ok) throw new Error("拉取 LLM 配置失败");
        const data = await res.json();
        const llms = data.data.items;
        if (llms.length > 0) setLlmConfig(llms[0]);
      } catch (error) {
        console.error("拉取 LLM 配置失败:", error);
      }
    };

    fetchLLMConfig();
  }, []);

  // 模型选择回调
  const handleModelChange = async (
    id: string,
    source: string,
    model_id: string,
  ) => {
    setLlmConfig({
      ...llmConfig,
      id,
      source,
      model_id,
    });
  };

  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);

  useEffect(() => {
    console.log("selectedOptions updated:", selectedOptions);
  }, [selectedOptions]);

  const extra_body = useMemo(() => {
    const mcpOptions = selectedOptions.filter((opt) => opt.startsWith("mcp:"));
    const mcp_servers = mcpOptions.map((opt) => opt.split(":")[1]);
    const kbOptions = selectedOptions.filter((opt) => opt.startsWith("kb:"));
    const kb_ids = kbOptions.map((opt) => opt.split(":")[1]);

    return {
      model: llmConfig.model_id,
      mcp_servers: mcp_servers,
      enable_search: selectedOptions.includes("search"),
      enable_thinking: selectedOptions.includes("thinking"),
      enable_mcp: mcp_servers.length > 0,
      kb_ids: kb_ids,
    };
  }, [llmConfig.model_id, selectedOptions]);

  const runtime = usePaiChatThreadRuntime({
    api: `http://localhost:${process.env.NEXT_PUBLIC_BACKEND_PORT}/v1/agent/chat`,
    body: extra_body,
    adapters: {
      attachments: new UploadAttachmentAdapter(),
    },
  });
  const pathname = usePathname();
  console.log("pathname:", pathname);
  const [activeTab, setActiveTab] = useState(pathname);

  console.log("activeTab:", activeTab);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <SidebarProvider defaultOpen={true}>
        <AppSidebar activeTab={activeTab} setActiveTab={setActiveTab} />
        <SidebarInset className="h-screen overflow-hidden">
          {activeTab === "/" && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
                <div className="flex justify-start px-20 border-none">
                  <ModelSelector
                    selectedModel={{
                      source: llmConfig.source || "",
                      model_id: llmConfig.model_id || "",
                    }}
                    onModelChange={handleModelChange}
                  />
                </div>
              </header>
              <Thread
                onToggleChange={(options) => {
                  console.log("Received options from Thread:", options); // ✅ 添加日志
                  setSelectedOptions(options); // 更新状态
                }}
              />
              <ToolUIWrapper />
            </div>
          )}
          {activeTab === "/knowledgebase" && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <KnowledgeBase setActiveTab={setActiveTab} />
            </div>
          )}
          {activeTab.startsWith("/knowledgebase/create") && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <KnowledgeBaseCreatePage setActiveTab={setActiveTab} />
            </div>
          )}
          {activeTab.startsWith("/knowledgebase/details") && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <KnowledgeBaseDetailPage
                knowledgebase_id={activeTab.split("/")[3]}
                setActiveTab={setActiveTab}
              />
            </div>
          )}
          {activeTab.startsWith("/knowledgebase/chunks") && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <KnowledgeBaseFileChunksPage
                knowledgebase_file_id={activeTab.split("/")[3]}
                setActiveTab={setActiveTab}
              />
            </div>
          )}
          {activeTab === "/config/model" && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <ModelConfigPage />
            </div>
          )}
          {activeTab === "/config/mcp" && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <McpConfig />
            </div>
          )}
          {activeTab === "/config/search" && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <SearchConfig />
            </div>
          )}
          {activeTab === "/config/tracing" && (
            <div className="flex flex-col h-full">
              <header className="flex h-12 border-b">
                <SidebarTrigger />
              </header>
              <TracingConfig />
            </div>
          )}
        </SidebarInset>
      </SidebarProvider>
    </AssistantRuntimeProvider>
  );
};
function useRef<T>(arg0: never[]) {
  throw new Error("Function not implemented.");
}
