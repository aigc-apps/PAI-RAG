import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import type { FC } from "react";
import { useCallback, useMemo, useState, useEffect, useRef } from "react";
import {
  ArrowDownIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  PencilIcon,
  RefreshCwIcon,
  SendHorizontalIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { Button } from "@/components/ui/button";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
// import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { ToolFallback } from "@/components/ui/custom-tool-fallback";
import { Brain, Search, Wrench } from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { MCPConfig } from "@/app/config/mcp/page";
import { McpModal } from "@/app/config/mcp/mcpmodal";

export const Thread: FC<{ onToggleChange?: (options: string[]) => void }> = ({
  onToggleChange,
}) => {
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [mcpConfigs, setMcpConfigs] = useState<MCPConfig[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpError, setMcpError] = useState<string | null>(null);

  // 使用 ref 保存最新值
  const activeToolsRef = useRef<string[]>([]);
  useEffect(() => {
    activeToolsRef.current = activeTools;
  }, [activeTools]);

  const mcpConfigsRef = useRef<MCPConfig[]>([]);
  useEffect(() => {
    mcpConfigsRef.current = mcpConfigs;
  }, [mcpConfigs]);

  // 获取MCP配置
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        setMcpLoading(true);
        const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
        const res = await fetch(`http://localhost:${port}/api/configs`);
        if (!res.ok) throw new Error("获取配置失败");
        const data = await res.json();
        const configs = data.mcp_config.map(
          (cfg: any) =>
            new MCPConfig(
              cfg.id,
              cfg.name,
              cfg.url,
              cfg.type,
              cfg.active || false,
              cfg.enabled || true,
            ),
        );
        setMcpConfigs(
          configs.filter((item: { enabled: boolean }) => item.enabled === true),
        );
      } catch (err: any) {
        setMcpError(err.message || "加载失败");
      } finally {
        setMcpLoading(false);
      }
    };
    fetchConfigs();
  }, []);

  // 保存MCP配置到后端
  const handleSaveMcpConfig = useCallback(
    async (updatedConfigs: MCPConfig[]) => {
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
      const url = `http://localhost:${port}/api/add_mcp`;
      console.log("Sending updated MCP configs:", updatedConfigs);
      try {
        const savePromises = updatedConfigs.map((config) =>
          fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mcp_config: config }),
          }),
        );
        const responses = await Promise.all(savePromises);
        const hasError = responses.some((res) => !res.ok);
        if (hasError) {
          throw new Error("部分配置保存失败");
        }

        // 更新本地状态
        setMcpConfigs((prev) =>
          prev.map((cfg) => {
            const updated = updatedConfigs.find((u) => u.id === cfg.id);
            return updated
              ? new MCPConfig(
                  updated.id,
                  updated.name,
                  updated.url,
                  updated.type,
                  updated.active,
                  updated.enabled,
                )
              : cfg;
          }),
        );
      } catch (err: any) {
        setMcpError(err.message || "保存失败");
      } finally {
        setIsModalOpen(false);
      }
    },
    [],
  );

  const activeMcpConfigs = useMemo(
    () => mcpConfigs.filter((cfg) => cfg.active),
    [mcpConfigs],
  );

  // 处理工具切换
  const handleToolToggle = useCallback(
    (newOptions: string[]) => {
      console.log("mcpConfigs", mcpConfigsRef.current);
      let updatedOptions = [...newOptions];
      const hasMcp = updatedOptions.includes("mcp");
      const hasActiveMcp = activeMcpConfigs.length > 0;

      // 自动添加 Thinking
      if (hasMcp && !updatedOptions.includes("thinking") && hasActiveMcp) {
        updatedOptions.push("thinking");
      }

      // 确保没有激活的MCP时清除'mcp'
      if (!hasActiveMcp && updatedOptions.includes("mcp")) {
        updatedOptions = updatedOptions.filter((opt) => opt !== "mcp");
      }

      setActiveTools(updatedOptions);
      onToggleChange?.(updatedOptions);
    },
    [mcpConfigsRef],
  );

  // 清除MCP激活状态
  const clearMcpActivation = useCallback(() => {
    const updatedConfigs = mcpConfigs.map((cfg) => ({ ...cfg, active: false }));
    handleSaveMcpConfig(updatedConfigs).catch(console.error);
  }, [mcpConfigs, handleSaveMcpConfig]);

  // 监听activeTools变化处理MCP逻辑
  useEffect(() => {
    const hasMcp = activeTools.includes("mcp");
    const hasActiveMcp = activeMcpConfigs.length > 0;

    // 如果 MCP 被选中但没有激活的配置，打开模态框
    if (hasMcp && !hasActiveMcp) {
      setIsModalOpen(true);
    }

    // 当取消选择 thinking 且存在 activeMcp 时清除 MCP 激活
    if (
      !activeTools.includes("thinking") &&
      activeToolsRef.current.includes("thinking") &&
      hasActiveMcp
    ) {
      clearMcpActivation();
    }
  }, [activeTools, activeMcpConfigs, clearMcpActivation]);

  // 监听 mcpConfigs 变化更新工具状态
  useEffect(() => {
    const hasActiveMcp = activeMcpConfigs.length > 0;
    setActiveTools((prev) => {
      let newTools = [...prev];

      if (hasActiveMcp && !newTools.includes("thinking")) {
        newTools.push("thinking");
      } else if (!hasActiveMcp && newTools.includes("mcp")) {
        newTools = newTools.filter((t) => t !== "mcp");
      }

      return newTools;
    });
  }, [mcpConfigs, activeMcpConfigs]);

  const handleOpenMcpModal = () => {
    setIsModalOpen(true);
  };

  return (
    <>
      <ThreadPrimitive.Root
        className="bg-background box-border flex h-full flex-col overflow-hidden"
        style={{
          ["--thread-max-width" as string]: "60rem",
        }}
      >
        <ThreadPrimitive.Viewport className="flex h-full flex-col items-center overflow-y-scroll scroll-smooth bg-inherit px-4 pt-8">
          <ThreadWelcome />
          <ThreadPrimitive.Messages
            components={{
              UserMessage: UserMessage,
              EditComposer: EditComposer,
              AssistantMessage: AssistantMessage,
            }}
          />
          <ThreadPrimitive.If empty={false}>
            <div className="min-h-8 flex-grow" />
          </ThreadPrimitive.If>
          <div className="sticky bottom-0 mt-3 flex w-full max-w-[var(--thread-max-width)] flex-col items-center justify-end rounded-t-lg bg-inherit pb-4">
            <ThreadScrollToBottom />
            <Composer
              onToggleChange={handleToolToggle}
              value={activeTools}
              mcpConfigs={mcpConfigs}
              onOpenMcpModal={handleOpenMcpModal}
            />
          </div>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
      <McpModal
        mcpConfigs={mcpConfigs}
        isOpen={isModalOpen}
        onSave={handleSaveMcpConfig}
        onClose={() => setIsModalOpen(false)}
        isLoading={mcpLoading}
        error={mcpError}
      />
    </>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="absolute -top-8 rounded-full disabled:invisible"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  return (
    <ThreadPrimitive.Empty>
      <div className="flex w-full max-w-[var(--thread-max-width)] flex-grow flex-col">
        <div className="flex w-full flex-grow flex-col items-center justify-center">
          <p className="mt-4 font-medium">有什么我能帮您的吗？</p>
        </div>
        <ThreadWelcomeSuggestions />
      </div>
    </ThreadPrimitive.Empty>
  );
};

const ThreadWelcomeSuggestions: FC = () => {
  return (
    <div className="mt-3 flex w-full items-stretch justify-center gap-4">
      <ThreadPrimitive.Suggestion
        className="hover:bg-muted/80 flex max-w-sm grow basis-0 flex-col items-center justify-center rounded-lg border p-3 transition-colors ease-in"
        prompt="帮我规划下个月从杭州去上海旅游的一日游攻略和交通规划，两大一小，考虑天气情况。"
        method="replace"
        autoSend
      >
        <span className="line-clamp-2 text-ellipsis text-sm font-semibold">
          帮我规划下个月从杭州去上海旅游的一日游攻略和交通规划，两大一小，考虑天气情况。
        </span>
      </ThreadPrimitive.Suggestion>
      <ThreadPrimitive.Suggestion
        className="hover:bg-muted/80 flex max-w-sm grow basis-0 flex-col items-center justify-center rounded-lg border p-3 transition-colors ease-in"
        prompt="杭州有什么好玩的景点？"
        method="replace"
        autoSend
      >
        <span className="line-clamp-2 text-ellipsis text-sm font-semibold">
          杭州有什么好玩的景点？
        </span>
      </ThreadPrimitive.Suggestion>
    </div>
  );
};

interface ComposerProps {
  onToggleChange?: (options: string[]) => void;
  value?: string[];
  mcpConfigs?: MCPConfig[]; // 新增
  onOpenMcpModal?: () => void; // 新增
}

const Composer: FC<ComposerProps> = ({
  onToggleChange,
  value,
  mcpConfigs = [], // 默认值
  onOpenMcpModal,
}) => {
  return (
    <ComposerPrimitive.Root
      // className="focus-within:border-ring/20 flex w-full flex-wrap items-end rounded-lg border bg-inherit px-2.5 shadow-sm transition-colors ease-in"
      className="focus-within:border-ring/20 flex w-full flex-col rounded-lg border bg-inherit px-2.5 shadow-sm transition-colors ease-in"
    >
      {/* 第一行：输入框 */}
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="输入您的问题..."
        className="placeholder:text-muted-foreground max-h-40 w-full resize-none border-none bg-transparent px-2 py-4 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
      />

      {/* 第二行：按钮组 + ComposerAction */}
      <div className="flex flex-row items-center justify-between px-2 pb-4">
        <div>
          <ToggleGroup
            type="multiple"
            variant="outline"
            className="flex gap-x-4 overflow-visible"
            onValueChange={(value) => {
              onToggleChange?.(value); // 传递选中状态到父组件
              // 2. 如果选中了 "mcp"，则打开模态框
              if (value.includes("mcp")) {
                onOpenMcpModal?.();
              }
            }}
            value={value} // 同步 Thread 的 activeTools
          >
            <ToggleGroupItem
              value="thinking"
              aria-label="Toggle deep thinking"
              className="!rounded-full px-6 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
            >
              <Brain /> 深度思考
            </ToggleGroupItem>
            <ToggleGroupItem
              value="search"
              aria-label="Toggle web search"
              className="!rounded-full px-2 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
            >
              <Search /> 搜索
            </ToggleGroupItem>
            <ToggleGroupItem
              value="mcp"
              aria-label="Toggle mcp"
              className={cn("!rounded-full px-2 py-3", {
                "bg-black text-white":
                  value?.includes("mcp") ||
                  mcpConfigs.some((cfg) => cfg.active),
              })}
            >
              <Wrench /> MCP
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        {/* 右侧按钮：ComposerAction */}
        <div className="ml-auto">
          <ComposerAction />
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
};

const ComposerAction: FC = () => {
  return (
    <>
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send asChild>
          <TooltipIconButton
            tooltip="Send"
            variant="default"
            className="my-2.5 size-8 p-2 transition-opacity ease-in"
          >
            <SendHorizontalIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel asChild>
          <TooltipIconButton
            tooltip="Cancel"
            variant="default"
            className="my-2.5 size-8 p-2 transition-opacity ease-in"
          >
            <CircleStopIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
    </>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="grid auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] gap-y-2 [&:where(>*)]:col-start-2 w-full max-w-[var(--thread-max-width)] py-4">
      <UserActionBar />
      <div className="bg-muted text-foreground max-w-[calc(var(--thread-max-width)*0.8)] break-words rounded-3xl px-5 py-2.5 col-start-2 row-start-2">
        <MessagePrimitive.Content />
      </div>

      <BranchPicker className="col-span-full col-start-1 row-start-3 -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      className="flex flex-col items-end col-start-1 row-start-2 mr-3 mt-2.5"
    >
      <ActionBarPrimitive.Edit asChild>
        <TooltipIconButton tooltip="Edit">
          <PencilIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  return (
    <ComposerPrimitive.Root className="bg-muted my-4 flex w-full max-w-[var(--thread-max-width)] flex-col gap-2 rounded-xl">
      <ComposerPrimitive.Input className="text-foreground flex h-8 w-full resize-none bg-transparent p-4 pb-0 outline-none" />

      <div className="mx-3 mb-3 flex items-center justify-center gap-2 self-end">
        <ComposerPrimitive.Cancel asChild>
          <Button variant="ghost">Cancel</Button>
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send asChild>
          <Button>Send</Button>
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="grid grid-cols-[auto_auto_1fr] grid-rows-[auto_1fr] relative w-full max-w-[var(--thread-max-width)] py-4">
      <div className="text-foreground max-w-[calc(var(--thread-max-width)*0.8)] break-words leading-7 col-span-2 col-start-2 row-start-1 my-1.5">
        {/* <MessagePrimitive.Content components={{ Text: MarkdownText }} /> */}
        <MessagePrimitive.Content
          components={{ tools: { Fallback: ToolFallback }, Text: MarkdownText }}
        />
      </div>

      <AssistantActionBar />

      <BranchPicker className="col-start-2 row-start-2 -ml-2 mr-2" />
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="text-muted-foreground flex gap-1 col-start-3 row-start-2 -ml-1 data-[floating]:bg-background data-[floating]:absolute data-[floating]:rounded-md data-[floating]:border data-[floating]:p-1 data-[floating]:shadow-sm"
    >
      <ActionBarPrimitive.Copy asChild>
        <TooltipIconButton tooltip="Copy">
          <MessagePrimitive.If copied>
            <CheckIcon />
          </MessagePrimitive.If>
          <MessagePrimitive.If copied={false}>
            <CopyIcon />
          </MessagePrimitive.If>
        </TooltipIconButton>
      </ActionBarPrimitive.Copy>
      <ActionBarPrimitive.Reload asChild>
        <TooltipIconButton tooltip="Refresh">
          <RefreshCwIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Reload>
    </ActionBarPrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn(
        "text-muted-foreground inline-flex items-center text-xs",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton tooltip="Previous">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip="Next">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};

const CircleStopIcon = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      width="16"
      height="16"
    >
      <rect width="10" height="10" x="3" y="3" rx="2" />
    </svg>
  );
};
