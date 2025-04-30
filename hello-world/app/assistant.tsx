"use client";

import React, { useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import MCPConfigPage from "@/components/setting/mcp-settings";
import { Button } from "@/components/ui/button";
import ModelSelector from "@/components/model-selector";
import { ALL_MODEL_NAMES } from "../constants";
import ToolUIWrapper from "@/components/assistant-ui/tool-ui";
import { SettingsIcon, Settings2Icon } from "lucide-react";

export const Assistant = () => {
  const [model, setModel] = useState<ALL_MODEL_NAMES>("gpt-4o");
  const runtime = useChatRuntime({
    // api: "http://127.0.0.1:8000/api/chat",
    api: "/api/chat",
    headers: { "X-Model-Name": model },
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="grid h-dvh grid-rows-[1fr_8fr]">
        <div className="grid grid-cols-[200px_1fr_auto] gap-x-2 px-4 py-4">
          <p className="text-2xl font-semibold text-black tracking-tighter">
            Agent <span className="font-extrabold text-red-600"> X </span>
          </p>
          <div className="flex justify-start px-40 w-full ">
            <ModelSelector modelName={model} setModelName={setModel} />
          </div>
          <div className="">
            <Button onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
              {isSidebarOpen ? <Settings2Icon /> : <SettingsIcon />}
            </Button>
          </div>
        </div>
        <div className="grid grid-cols-[200px_1fr_auto] gap-x-2 px-4 py-4">
          <ThreadList />
          <Thread />
          <ToolUIWrapper />
          {/* 动态侧边栏保持原有过渡逻辑 */}
          <div
            className={`transition-all duration-300 ease-in-out overflow-hidden ${
              isSidebarOpen ? "w-[500px]" : "w-0"
            }`}
          >
            {isSidebarOpen && <MCPConfigPage />}
          </div>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
};
