"use client";

import React, { useState, useEffect } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import ConfigPage from "@/components/setting/config-page";
import { Button } from "@/components/ui/button";
import ModelSelector from "@/components/model-selector/index";
import ToolUIWrapper from "@/components/assistant-ui/tool-ui";
import { SettingsIcon, Settings2Icon } from "lucide-react";

export const Assistant = () => {
  // LLM 配置状态
  const [llmConfig, setLlmConfig] = useState({
    id: Date.now(),
    source: "",
    model_name: "",
    api_key: "",
    max_context: 0,
  });

  // 页面加载时拉取 LLM 配置
  useEffect(() => {
    const fetchLLMConfig = async () => {
      try {
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
        console.log("assistant BACKEND_PORT", port);
        const res = await fetch(`http://localhost:${port}/api/configs`);
        if (!res.ok) throw new Error("拉取 LLM 配置失败");
        const data = await res.json();
        if (data.llm_config?.[0]) setLlmConfig(data.llm_config[0]);
      } catch (error) {
        console.error("拉取 LLM 配置失败:", error);
      }
    };

    fetchLLMConfig();
  }, []);

  // 模型选择回调
  const handleModelChange = async (
    source: string,
    model_name: string,
    api_key: string,
  ) => {
    setLlmConfig({
      ...llmConfig,
      source,
      model_name,
      api_key,
    });
  };

  const runtime = useChatRuntime({
    // api: "/api/chat",
    api: `http://localhost:${process.env.NEXT_PUBLIC_BACKEND_PORT}/api/chat`,
    headers: {
      "X-Model-Name": llmConfig.model_name || "gpt-4o",
      "X-Api-Key": llmConfig.api_key || "",
      "X-Model-Source": llmConfig.source || "openai",
    },
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="grid h-screen grid-rows-[1fr_8fr] overflow-hidden">
        {/* 顶部栏 */}
        <div className="grid grid-cols-[200px_1fr_auto] gap-x-2 px-4 py-4">
          <p className="text-2xl font-semibold text-black tracking-tighter">
            Agent <span className="font-extrabold text-red-600"> X </span>
          </p>
          <div className="flex justify-start px-40 w-full">
            <ModelSelector
              selectedModel={{
                source: llmConfig.source || "",
                model_name: llmConfig.model_name || "",
              }}
              onModelChange={handleModelChange}
            />
          </div>
          <div>
            <Button onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
              {isSidebarOpen ? <Settings2Icon /> : <SettingsIcon />}
            </Button>
          </div>
        </div>

        {/* 主体区域 */}
        <div className="grid grid-cols-[200px_1fr_auto] gap-x-2 px-4 py-4 h-full overflow-hidden">
          <ThreadList />
          <Thread />
          <ToolUIWrapper />
          <div
            className={`transition-all duration-300 ease-in-out overflow-hidden ${
              isSidebarOpen ? "w-[500px]" : "w-0"
            } h-full`}
          >
            {isSidebarOpen && (
              <div className="h-full overflow-y-auto">
                <ConfigPage />
              </div>
            )}
          </div>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
};
