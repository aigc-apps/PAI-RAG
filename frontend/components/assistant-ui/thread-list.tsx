import type { FC } from 'react';
import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
} from '@assistant-ui/react';
import { ArchiveIcon, PlusIcon, TrashIcon } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button';
import { useRouter } from 'next/navigation';
import { useI18n } from '@/app/providers/i18n';

export const ThreadList: FC = () => {
  return (
    <ThreadListPrimitive.Root className="flex flex-col items-stretch gap-1.5">
      <ThreadListNew />
      <ThreadListItems />
    </ThreadListPrimitive.Root>
  );
};

const ThreadListNew: FC = () => {
  const { t } = useI18n();
  const router = useRouter();
  return (
    <ThreadListPrimitive.New asChild>
      <Button
        className="data-[active]:bg-muted hover:bg-muted flex items-center justify-start gap-1 rounded-lg px-2.5 py-2 text-start"
        variant="ghost"
        onClick={() => {router.push('/')}}
      >
        <PlusIcon />
        <span suppressHydrationWarning>{t('chat.threadList.newConversation')}</span>
      </Button>
    </ThreadListPrimitive.New>
  );
};

const ThreadListItems: FC = () => {
  return <ThreadListPrimitive.Items components={{ ThreadListItem }} />;
};

const ThreadListItem: FC = () => {
  const router = useRouter();
  return (
    <ThreadListItemPrimitive.Root className="data-[active]:bg-muted hover:bg-muted focus-visible:bg-muted focus-visible:ring-ring flex items-center gap-2 rounded-lg transition-all focus-visible:outline-none focus-visible:ring-2">
      <ThreadListItemPrimitive.Trigger className="flex-grow px-3 py-2 text-start" onClick={() => {router.push('/')}}>
        <ThreadListItemTitle />
      </ThreadListItemPrimitive.Trigger>
      {/* <ThreadListItemArchive /> */}
      <ThreadListItemDelete />
    </ThreadListItemPrimitive.Root>
  );
};

const ThreadListItemTitle: FC = () => {
  const { t } = useI18n();
  return (
    <p className="text-sm" suppressHydrationWarning>
      <ThreadListItemPrimitive.Title fallback={t('chat.threadList.newSession')} />
    </p>
  );
};

const ThreadListItemArchive: FC = () => {
  const { t } = useI18n();
  return (
    <ThreadListItemPrimitive.Archive asChild>
      <TooltipIconButton
        className="hover:text-primary text-foreground ml-auto mr-3 size-4 p-0"
        variant="ghost"
        tooltip={t('chat.threadList.archiveThread')}
      >
        <ArchiveIcon />
      </TooltipIconButton>
    </ThreadListItemPrimitive.Archive>
  );
};

const ThreadListItemDelete: FC = () => {
  const { t } = useI18n();
  return (
    <ThreadListItemPrimitive.Delete asChild>
      <TooltipIconButton
        className="hover:text-primary text-foreground ml-auto mr-3 size-4 p-0"
        variant="ghost"
        tooltip={t('chat.threadList.deleteThread')}
      >
        <TrashIcon />
      </TooltipIconButton>
    </ThreadListItemPrimitive.Delete>
  );
};
