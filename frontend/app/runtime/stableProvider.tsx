'use client';
import React from 'react';

import { useMemo } from 'react';
import { ExportedMessageRepository, ThreadHistoryAdapter, ThreadMessage } from '@assistant-ui/react';
import {
  useThreadListItem,
  RuntimeAdapterProvider
} from '@assistant-ui/react';
import {  } from '@assistant-ui/react';


export const StableProvider: React.ComponentType<{ children?: React.ReactNode }> = ({
  children,
}) => {
  // This runs in the context of each thread
  const threadListItem = useThreadListItem();
  const remoteId = threadListItem.remoteId;
  // Create thread-specific history adapter
  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      async load() {
        if (!remoteId) return { headId: null, messages: [] };
        // 模拟从后端获取数据
        try {
          const res = await fetch(`/v1/agent/threads/${remoteId}/messages`);

          if (!res.ok) throw new Error('获取配置失败');
          const messages = await res.json();
          if (messages.length === 0) {
            return { headId: null, messages: [] };
          }
          const response = ExportedMessageRepository.fromArray(
            messages.map((m: any) => ({
              role: m.role as ThreadMessage['role'],
              content: m.content,
              attachments: m.attachments,
              id: m.id,
              createdAt: new Date(m.createdAt),
            })),
          );
          return response;
        } catch (error) {
          console.error('Error fetching threads:', error);
          return { headId: null, messages: [] };
        }
      },
      async append(message) {
        if (!remoteId) {
          console.warn('Cannot save message - thread not initialized');
          return;
        }
        try {
          const url = `/v1/agent/threads/${remoteId}/messages`;
          console.log('append message', message);
          
          const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              thread_id: remoteId,
              role: message.message.role,
              attachments: message.message.attachments,
              content: message.message.content,
            }),
          });

          if (!response.ok) {
            throw new Error(`Failed to create thread: ${response.statusText}`);
          }
        } catch (error) {
          console.error('Error creating thread:', error);
          throw error;
        }
      },
    }),
    [remoteId],
  );
  const adapters = useMemo(() => ({ history }), [history]);
  return (
    <RuntimeAdapterProvider adapters={adapters}>
      {children}
    </RuntimeAdapterProvider>
  );
};