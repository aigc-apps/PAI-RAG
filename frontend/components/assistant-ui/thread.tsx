import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from '@assistant-ui/react';
import type { FC } from 'react';
import { useState, useEffect } from 'react';
import {
  ArrowDownIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  PencilIcon,
  RefreshCwIcon,
  SendHorizontalIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';

import { Button } from '@/components/ui/button';
import { MarkdownText } from '@/components/assistant-ui/markdown-text';
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button';
// import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { ToolFallback } from '@/components/ui/custom-tool-fallback';
import { Brain, Search, Wrench, LibraryBig } from 'lucide-react';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { McpModal, McpEntry } from '@/app/config/mcp/mcpmodal';
import {
  ComposerAttachments,
  ComposerAddAttachment,
} from '@/components/assistant-ui/my_attachment';
import { UserMessageAttachments } from '@/components/assistant-ui/my_attachment';
import { KbModal, KbSelection } from '@/app/knowledgebases/kbmodal';
import { useChatOptions } from '@/app/providers/chat';

export const Thread: FC<{
  onToggleChange?: (options: string[]) => void;
  optionsVisible: boolean;
}> = ({ onToggleChange, optionsVisible }) => {
  // 使用useState来保存工具的选中状态
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [mcpConfigs, setMcpConfigs] = useState<McpEntry[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpError, setMcpError] = useState<string | null>(null);

  const [kbConfigs, setKbConfigs] = useState<KbSelection[]>([]);
  const [isKbModalOpen, setIsKbModalOpen] = useState(false);
  const [kbLoading, setKbLoading] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);
  const {kb_ids, mcp_ids, enable_search, enable_thinking, updateEnableSearch, updateEnableThinking, updateKbIds, updateMcpIds} = useChatOptions();

  // 获取MCP配置
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        setMcpLoading(true);
        const res = await fetch('/v1/config/mcps');
        if (!res.ok) throw new Error('获取配置失败');
        const data = await res.json();

        const configs = data.data.items.map(
          (cfg: any) =>
            new McpEntry(
              cfg.id,
              cfg.name,
              cfg.url,
              cfg.type,
              cfg.enabled ?? true,
              mcp_ids.includes(cfg.id),
            ),
        );
        const enabledConfigs = configs.filter(
          (item: { enabled: boolean }) => item.enabled === true,
        );
        console.log('all MCP configs: ', configs);
        console.log('enabled MCP configs: ', enabledConfigs);
        setMcpConfigs(enabledConfigs);
      } catch (err: any) {
        setMcpError(err.message || '加载失败');
      } finally {
        setMcpLoading(false);
      }

      try {
        setKbLoading(true);
        const res = await fetch('/v1/config/knowledgebases');
        if (!res.ok) throw new Error('获取知识库配置失败');
        const json_res = await res.json();
        console.log('Load kb.', json_res);

        const configs = json_res.data.items.map(
          (cfg: any) =>
            new KbSelection(
              cfg.id,
              cfg.name,
              cfg.description,
              kb_ids.includes(cfg.id),
              cfg.updated_at,
            ),
        );
        console.log('all kb configs: ', configs);
        setKbConfigs(configs);
      } catch (err: any) {
        setKbError(err.message || '加载知识库失败');
      } finally {
        setKbLoading(false);
      }
    };

    fetchConfigs();
    let newOptions: string[] = [];
    if (enable_search) {
      newOptions.push('search');
    }
    if (enable_thinking) {
      newOptions.push('thinking');
    }
    if (kb_ids.length > 0) {
      newOptions.push('kb');
    }
    if (mcp_ids.length > 0) {
      newOptions.push('mcp');
    }
    setActiveTools(newOptions);
    onToggleChange?.(newOptions);
  }, []);

  const handleToolUpdate = (
    value: string[]
  ) => {
    updateEnableSearch(value.includes('search'));
    updateEnableThinking(value.includes('thinking'));
    setActiveTools(value);
  }

  const handleKbUpdate = (
    updatedConfigs: KbSelection[]
  ) => {
    const new_kb_ids = updatedConfigs.filter((kb) => kb.active).map((kb) => kb.id);
    updateKbIds(new_kb_ids);

    const hasKb = activeTools.includes('kb');
    if (!hasKb && new_kb_ids.length > 0) {
        setActiveTools([...activeTools, 'kb']);
    }
    else if (hasKb && new_kb_ids.length === 0) {
      setActiveTools(activeTools.filter(v => v !== 'kb'))
    }

    // 6. 更新本地状态
    // 7. 同步到父组件
    onToggleChange?.(activeTools);
  };

  const handleMcpUpdate = (
    updatedConfigs: McpEntry[],
  ) => {
    const new_mcp_ids = updatedConfigs.filter((mcp) => mcp.active).map((mcp) => mcp.id);
    updateMcpIds(new_mcp_ids);

    const hasMcp = activeTools.includes('mcp');
    if (!hasMcp && new_mcp_ids.length > 0) {
        setActiveTools([...activeTools, 'mcp']);
    }
    else if (hasMcp && new_mcp_ids.length === 0) {
      setActiveTools(activeTools.filter(v => v !== 'mcp'))
    }

    // 6. 更新本地状态
    // 7. 同步到父组件
    onToggleChange?.(activeTools);
  };

  const handleOpenMcpModal = () => {
    setIsModalOpen(true);
  };
  const handleOpenKbModal = () => {
    setIsKbModalOpen(true);
  };

  return (
    <>
      <ThreadPrimitive.Root
        className="bg-background box-border flex h-full flex-col overflow-hidden"
        style={{
          ['--thread-max-width' as string]: '60rem',
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
              value={activeTools}
              onValueChange={(value) => handleToolUpdate(value)}
              optionsVisible={optionsVisible}
              onOpenMcpModal={handleOpenMcpModal}
              onOpenKbModal={handleOpenKbModal}
            />
            {/* 传递回调 */}
          </div>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
      <McpModal
        mcpConfigs={mcpConfigs}
        isOpen={isModalOpen}
        onSave={(updatedConfigs) => {
          // 传入当前的 activeTools 作为 newOptions
          handleMcpUpdate(updatedConfigs);
          setIsModalOpen(false);
        }}
        onClose={() => setIsModalOpen(false)}
        isLoading={mcpLoading}
        error={mcpError}
      />
      <KbModal
        kbConfigs={kbConfigs}
        isOpen={isKbModalOpen}
        onSave={(updatedKbConfigs) => {
          // 传入当前的 activeTools 作为 newOptions
          handleKbUpdate(updatedKbConfigs);
          setIsKbModalOpen(false);
        }}
        onClose={() => setIsKbModalOpen(false)}
        isLoading={kbLoading}
        error={kbError}
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
  value?: string[];
  optionsVisible: boolean;
  onOpenMcpModal?: () => void; // 新增
  onValueChange: (value: string[]) => void;
  onOpenKbModal: () => void;
}

const Composer: FC<ComposerProps> = ({
  value,
  optionsVisible,
  onValueChange,
  onOpenMcpModal,
  onOpenKbModal,
}) => {
  return (
    <ComposerPrimitive.Root
      // className="focus-within:border-ring/20 flex w-full flex-wrap items-end rounded-lg border bg-inherit px-2.5 shadow-sm transition-colors ease-in"
      className="focus-within:border-ring/20 flex w-full flex-col rounded-lg border bg-inherit px-2.5 shadow-sm transition-colors ease-in"
    >
      {/* 第一行：输入框 */}
      <div>
        <ComposerAttachments />
        <ComposerAddAttachment />
      </div>
      <div className="flex items-center justify-between px-2 pb-4">
        <div>
          <ComposerPrimitive.Input
            rows={1}
            autoFocus
            placeholder="输入您的问题..."
            className="placeholder:text-muted-foreground max-h-40 w-full resize-none border-none bg-transparent px-2 py-4 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
          />

          {/* 第二行：按钮组 + ComposerAction */}
          {optionsVisible && (
            <div className="flex flex-row items-center justify-between px-2 pb-4">
              <div>
                <ToggleGroup
                  type="multiple"
                  variant="outline"
                  className="flex gap-x-4 overflow-visible"
                  value={value} // 同步 Thread 的 activeTools
                  onValueChange={onValueChange}
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
                    className="!rounded-full px-2 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
                    onClick={() => {
                      onOpenMcpModal?.();
                    }}
                  >
                    <Wrench /> MCP
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="kb"
                    aria-label="Toggle kb"
                    className="!rounded-full px-2 py-3 data-[state=on]:bg-black data-[state=on]:text-white"
                    onClick={() => {
                      onOpenKbModal?.();
                    }}
                  >
                    <LibraryBig /> 知识库
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>

              {/* 右侧按钮：ComposerAction */}
            </div>
          )}
        </div>
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
            tooltip="发送"
            variant="default"
            className="my-2.5 w-18 h-10 p-2 transition-opacity ease-in"
          >
            <SendHorizontalIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel asChild>
          <TooltipIconButton
            tooltip="取消"
            variant="default"
            className="my-2.5 w-18 h-10 p-2 transition-opacity ease-in"
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
        <UserMessageAttachments />
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

      <BranchPicker className="flex flex-row gap-x-6 col-start-2 row-start-2 mr-10" />
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="flex flex-row items-center gap-x-10  text-muted-foreground flex gap-1 col-start-3 row-start-2 -ml-1 data-[floating]:bg-background data-[floating]:absolute data-[floating]:rounded-md data-[floating]:border data-[floating]:p-1 data-[floating]:shadow-sm"
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
        'text-muted-foreground inline-flex items-center text-xs',
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
