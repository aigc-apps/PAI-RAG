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
  DatabaseIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from "@/components/ui/button";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { CollapsibleReasoning } from "@/components/assistant-ui/reasoning-ui";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
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
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export const Thread: FC<{
  onToggleChange?: (options: string[]) => void;
  optionsVisible: boolean;
}> = ({ onToggleChange, optionsVisible }) => {
  const { t } = useI18n();

  // Use useState to save tool selection state
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const [mcpConfigs, setMcpConfigs] = useState<McpEntry[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpError, setMcpError] = useState<string | null>(null);

  const [kbConfigs, setKbConfigs] = useState<KbSelection[]>([]);
  const [isKbModalOpen, setIsKbModalOpen] = useState(false);
  const [kbLoading, setKbLoading] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);
  const {kb_ids, mcp_ids, enable_search, enable_agent, enable_chatdb, 
    updateEnableSearch, updateEnablePlanning, updateKbIds, updateMcpIds, updateEnableChatdb} = useChatOptions();
  const { tenantFetch } = useTenantFetch();
  // 获取MCP配置
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        setMcpLoading(true);
        setKbLoading(true);
        const [mcpRes, kbRes] = await Promise.all(
          [
            tenantFetch(`/api/config/mcps`),
            tenantFetch(`/api/config/knowledgebases`),
          ]
        )
        if (!mcpRes.ok) setMcpError(t('chat.thread.mcpLoadFailed'));
        else {
          const data = await mcpRes.json();

          const items = data?.data?.items || [];
          const configs = items
            .filter((cfg: any) => cfg != null)
            .map(
              (cfg: any) =>
                new McpEntry(
                  cfg.id || '',
                  cfg.name || '',
                  cfg.url || '',
                  cfg.type || '',
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
          setMcpLoading(false);
        }

        if (!kbRes.ok) setKbError(t('chat.thread.kbLoadFailed'));
        else {
          const json_res = await kbRes.json();
          console.log('Load kb.', json_res);

          // Ensure json_res.data.items exists and is an array
          const items = json_res?.data?.items || [];
          const configs = items
            .filter((cfg: any) => cfg != null)
            .map(
              (cfg: any) =>
                new KbSelection(
                  cfg.id || '',
                  cfg.name || '',
                  cfg.description || '',
                  kb_ids.includes(cfg.id),
                  cfg.updated_at || '',
                ),
            );
          console.log('all kb configs: ', configs);
          setKbConfigs(configs);
          setKbLoading(false);
        }
      }
      catch (error) {
        console.error('Error fetching configs:', error);
        setKbLoading(false);
        setMcpLoading(false);
      }
    };

    fetchConfigs();
    let newOptions: string[] = [];
    if (enable_search) {
      newOptions.push('search');
    }
    if (enable_agent) {
      newOptions.push('planning');
    }
    if (enable_chatdb) {
      newOptions.push('chatdb');
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
    updateEnablePlanning(value.includes('planning'));
    updateEnableChatdb(value.includes('chatdb'));
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

  const [tempActiveTools, setTempActiveTools] = useState<string[]>([]);

  const handleOpenMcpModal = () => {
    setTempActiveTools([...activeTools]);
    setIsModalOpen(true);
  };
  const handleCancelMcpModal = () => {
    // Restore previous state
    setActiveTools(tempActiveTools);
    setIsModalOpen(false);
  };
  const handleOpenKbModal = () => {
    setTempActiveTools([...activeTools]);
    setIsKbModalOpen(true);
  };

  const handleCancelKbModal = () => {
    // Restore previous state
    setActiveTools([...tempActiveTools]);
    setIsKbModalOpen(false);
  };

  return (
    <>
      <ThreadPrimitive.Root
        className="bg-background box-border flex h-full flex-col overflow-hidden"
        style={{
          ['--thread-max-width' as string]: '60rem',
        }}
      >
        <ThreadPrimitive.Viewport className="flex h-full flex-col items-center overflow-y-scroll scroll-smooth bg-inherit px-2">
          <ThreadWelcome />

          <ThreadPrimitive.Messages
            components={{
              UserMessage: UserMessage,
              EditComposer: EditComposer,
              AssistantMessage: AssistantMessage,
            }}
          />

          <ThreadPrimitive.If empty={false}>
            <div className="min-h-2 flex-grow" />
          </ThreadPrimitive.If>

          <div className="sticky bottom-6 mt-12 flex w-full max-w-[var(--thread-max-width)] flex-col items-center justify-end rounded-t-lg bg-inherit pb-2">
            <ThreadScrollToBottom />
            <Composer
              value={activeTools}
              onValueChange={(value) => handleToolUpdate(value)}
              optionsVisible={optionsVisible}
              onOpenMcpModal={handleOpenMcpModal}
              onOpenKbModal={handleOpenKbModal}
            />
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
        onClose={handleCancelMcpModal}
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
        onClose={handleCancelKbModal}
        isLoading={kbLoading}
        error={kbError}
      />
    </>
  );
};

const ThreadScrollToBottom: FC = () => {
  const { t } = useI18n();
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip={t('chat.thread.scrollToBottom')}
        variant="outline"
        className="absolute -top-8 rounded-full disabled:invisible"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  const { t } = useI18n();
  return (
    <ThreadPrimitive.Empty>
      <div className="flex w-full max-w-[var(--thread-max-width)] flex-grow flex-col">
        <div className="flex w-full flex-grow flex-col items-center justify-center">
          <p className="mt-4 font-medium">{t('chat.thread.welcomeMessage')}</p>
        </div>
        <ThreadWelcomeSuggestions />
      </div>
    </ThreadPrimitive.Empty>
  );
};

const ThreadWelcomeSuggestions: FC = () => {
  const { t } = useI18n();
  return (
    <div className="mt-3 flex w-full items-stretch justify-center gap-4 pb-8">
      <ThreadPrimitive.Suggestion
        className="hover:bg-muted/80 flex max-w-sm grow basis-0 flex-col items-center justify-center rounded-lg border p-3 transition-colors ease-in"
        prompt={t('chat.thread.suggestion1')}
        method="replace"
        autoSend
      >
        <span className="line-clamp-2 text-gray-800 text-xs text-ellipsis text-sm font-semibold">
          {t('chat.thread.suggestion1')}
        </span>
      </ThreadPrimitive.Suggestion>
      <ThreadPrimitive.Suggestion
        className="hover:bg-muted/80 flex max-w-sm grow basis-0 flex-col items-center justify-center rounded-lg border p-3 transition-colors ease-in"
        prompt={t('chat.thread.suggestion2')}
        method="replace"
        autoSend
      >
        <span className="line-clamp-2 text-gray-800 text-xs text-ellipsis text-sm font-semibold">
          {t('chat.thread.suggestion2')}
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
  const { t } = useI18n();
  return (
    <ComposerPrimitive.Root
      // className="focus-within:border-ring/20 flex w-full flex-wrap items-end rounded-lg border bg-inherit px-2.5 shadow-sm transition-colors ease-in"
      className="focus-within:border-ring/20 flex w-full flex-col rounded-lg border bg-inherit px-2.5 shadow-sm transition-colors ease-in"
    >
      {/* First row: input box */}
      <div className="flex items-center justify-between px-2 pb-1">
        <div className="w-full">
          <div className="flex gap-3 pt-2">
            <ComposerAttachments />
            <ComposerPrimitive.Input
              rows={1}
              autoFocus
              placeholder={t('chat.thread.inputQuestion')}
              className="flex-1 placeholder:text-muted-foreground max-h-40 resize-none border-none bg-transparent px-2 py-3 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
            />
          </div>
          {/* Second row: button group + ComposerAction */}
          {optionsVisible && (
            <div className="flex flex-row items-center px-2 pb-1">
                <ComposerAddAttachment />
                <ToggleGroup
                  type="multiple"
                  variant="outline"
                  className="flex gap-x-4 px-6 overflow-visible"
                  value={value} // 同步 Thread 的 activeTools
                  onValueChange={onValueChange}
                >
                  <ToggleGroupItem
                    value="planning"
                    aria-label="Toggle deep planning"
                    className="!rounded-full px-4 py-1 data-[state=on]:bg-black data-[state=on]:text-white h-8"
                  >
                    <Brain /> {t('chat.thread.deepThinking')}
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="search"
                    aria-label="Toggle web search"
                    className="!rounded-full px-2 py-1 data-[state=on]:bg-black data-[state=on]:text-white h-8"
                  >
                    <Search /> {t('chat.thread.search')}
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="mcp"
                    aria-label="Toggle mcp"
                    className="!rounded-full px-2 py-1 data-[state=on]:bg-black data-[state=on]:text-white h-8"
                    onClick={() => {
                      onOpenMcpModal?.();
                    }}
                  >
                    <Wrench /> MCP
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="kb"
                    aria-label="Toggle kb"
                    className="!rounded-full px-2 py-1 data-[state=on]:bg-black data-[state=on]:text-white h-8"
                    onClick={() => {
                      onOpenKbModal?.();
                    }}
                  >
                    <LibraryBig /> {t('chat.thread.knowledgeBase')}
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="chatdb"
                    aria-label="Toggle chatdb"
                    className="!rounded-full px-6 py-1 data-[state=on]:bg-black data-[state=on]:text-white h-8"
                  >
                    <DatabaseIcon /> ChatDB
                  </ToggleGroupItem>
                </ToggleGroup>

              {/* Right button: ComposerAction */}
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
  const { t } = useI18n();
  return (
    <>
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send asChild>
          <TooltipIconButton
            tooltip={t('chat.thread.send')}
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
            tooltip={t('chat.thread.cancel')}
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
      <div className="bg-muted text-foreground max-w-[calc(var(--thread-max-width)*0.8)] break-words rounded-2xl px-5 py-2 col-start-2 row-start-2 text-sm">
        <UserMessageAttachments />
        <MessagePrimitive.Content />
      </div>

      <BranchPicker className="col-span-full col-start-1 row-start-3 -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  const { t } = useI18n();
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      className="flex flex-col items-end col-start-1 row-start-2 mr-3 mt-2.5"
    >
      <ActionBarPrimitive.Edit asChild>
        <TooltipIconButton tooltip={t('chat.thread.edit')}>
          <PencilIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  const { t } = useI18n();
  return (
    <ComposerPrimitive.Root className="bg-muted my-4 flex w-full max-w-[var(--thread-max-width)] flex-col gap-2 rounded-xl">
      <ComposerPrimitive.Input className="text-foreground flex h-8 w-full resize-none bg-transparent p-2 pb-0 outline-none" />

      <div className="mx-2 mb-2 flex items-center justify-center gap-2 self-end">
        <ComposerPrimitive.Cancel asChild>
          <Button variant="ghost">{t('chat.thread.cancel')}</Button>
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send asChild>
          <Button>{t('chat.thread.send')}</Button>
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  );
};

const AssistantMessage: FC = () => {

  return (
    <MessagePrimitive.Root className="grid grid-cols-[auto_auto_1fr] grid-rows-[auto_1fr] relative w-full max-w-[var(--thread-max-width)] py-1">
      <div className="gap-2 text-foreground max-w-[calc(var(--thread-max-width)*0.8)] break-words leading-5 col-span-2 col-start-2 row-start-1 my-1.5 text-sm">
        {/* <MessagePrimitive.Content components={{ Text: MarkdownText }} /> */}
        <MessagePrimitive.Content
          components={{
            tools: { Fallback: ToolFallback },
            Text: MarkdownText,
            Reasoning: CollapsibleReasoning,
          }}
        />
      </div>

      <AssistantActionBar />

      <BranchPicker className="flex flex-row gap-x-6 col-start-2 row-start-2 mr-10" />
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  const { t } = useI18n();
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="flex flex-row items-center text-muted-foreground gap-1 col-start-3 row-start-2 -ml-1 data-[floating]:bg-background data-[floating]:absolute data-[floating]:rounded-md data-[floating]:border data-[floating]:p-1 data-[floating]:shadow-sm"
    >
      <ActionBarPrimitive.Copy asChild>
        <TooltipIconButton tooltip={t('chat.thread.copy')}>
          <MessagePrimitive.If copied>
            <CheckIcon />
          </MessagePrimitive.If>
          <MessagePrimitive.If copied={false}>
            <CopyIcon />
          </MessagePrimitive.If>
        </TooltipIconButton>
      </ActionBarPrimitive.Copy>
      <ActionBarPrimitive.Reload asChild>
        <TooltipIconButton tooltip={t('chat.thread.refresh')}>
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
  const { t } = useI18n();
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
        <TooltipIconButton tooltip={t('chat.thread.previous')}>
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip={t('chat.thread.next')}>
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
