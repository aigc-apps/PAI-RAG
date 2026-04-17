'use client';

import { type FC, createContext, useContext, useEffect, useState, useCallback } from 'react';
import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useThreadListItem,
} from '@assistant-ui/react';
import { PlusIcon, TrashIcon } from 'lucide-react';

import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button';
import { useRouter } from 'next/navigation';
import { useI18n } from '@/app/providers/i18n';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { formatFriendlyTime } from '@/lib/time-format';

/**
 * Provide a map of {remoteThreadId -> created_at} to children.
 * assistant-ui's ThreadListItemState doesn't expose timestamps,
 * so we fetch /api/threads separately once and refresh when the list mutates.
 */
type ThreadTimestampMap = Record<string, string>;
const ThreadTimestampContext = createContext<ThreadTimestampMap>({});

function ThreadTimestampProvider({ children }: { children: React.ReactNode }) {
  const { tenantFetch } = useTenantFetch();
  const [map, setMap] = useState<ThreadTimestampMap>({});

  const refresh = useCallback(async () => {
    try {
      const res = await tenantFetch(`/api/threads`);
      if (!res.ok) return;
      const response = await res.json();
      const next: ThreadTimestampMap = {};
      for (const t of response.data || []) {
        if (t.id && t.created_at) next[t.id] = t.created_at;
      }
      setMap(next);
    } catch {
      // ignore
    }
  }, [tenantFetch]);

  useEffect(() => {
    refresh();
    // Refresh periodically so new threads / deletions reconcile
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <ThreadTimestampContext.Provider value={map}>
      {children}
    </ThreadTimestampContext.Provider>
  );
}

function useThreadCreatedAt(remoteId: string | undefined): string | undefined {
  const map = useContext(ThreadTimestampContext);
  if (!remoteId) return undefined;
  return map[remoteId];
}

export const ThreadList: FC = () => {
  return (
    <ThreadTimestampProvider>
      <ThreadListPrimitive.Root className="flex flex-col items-stretch gap-0.5">
        <ThreadListNew />
        <ThreadListItems />
      </ThreadListPrimitive.Root>
    </ThreadTimestampProvider>
  );
};

const ThreadListNew: FC = () => {
  const { t } = useI18n();
  const router = useRouter();
  return (
    <ThreadListPrimitive.New asChild>
      <button className="new-chat-btn mb-1" onClick={() => router.push('/')}>
        <span className="flex items-center gap-2">
          <PlusIcon className="w-3.5 h-3.5" />
          <span suppressHydrationWarning>{t('chat.threadList.newConversation')}</span>
        </span>
      </button>
    </ThreadListPrimitive.New>
  );
};

const ThreadListItems: FC = () => {
  return <ThreadListPrimitive.Items components={{ ThreadListItem }} />;
};

const ThreadListItem: FC = () => {
  const router = useRouter();
  const remoteId = useThreadListItem((i) => i.remoteId);
  const createdAt = useThreadCreatedAt(remoteId);

  return (
    <ThreadListItemPrimitive.Root
      className="group/thread data-[active]:bg-muted hover:bg-muted focus-visible:bg-muted focus-visible:ring-ring flex items-center rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2"
      title={createdAt ? formatFriendlyTime(createdAt) : undefined}
    >
      <ThreadListItemPrimitive.Trigger
        className="flex-1 min-w-0 pl-1.5 pr-1 py-1 text-start"
        onClick={() => router.push('/')}
      >
        <ThreadListItemTitle />
      </ThreadListItemPrimitive.Trigger>
      <ThreadListItemDelete />
    </ThreadListItemPrimitive.Root>
  );
};

const ThreadListItemTitle: FC = () => {
  const { t } = useI18n();
  return (
    <p
      className="text-[12px] truncate leading-tight text-sidebar-foreground/90"
      suppressHydrationWarning
    >
      <ThreadListItemPrimitive.Title fallback={t('chat.threadList.newSession')} />
    </p>
  );
};

const ThreadListItemDelete: FC = () => {
  const { t } = useI18n();
  return (
    <ThreadListItemPrimitive.Delete asChild>
      <TooltipIconButton
        className="shrink-0 h-5 w-5 mr-0.5 p-0 text-sidebar-foreground/40 hover:text-destructive opacity-0 group-hover/thread:opacity-100 transition-opacity"
        variant="ghost"
        tooltip={t('chat.threadList.deleteThread')}
      >
        <TrashIcon className="h-3 w-3" />
      </TooltipIconButton>
    </ThreadListItemPrimitive.Delete>
  );
};
