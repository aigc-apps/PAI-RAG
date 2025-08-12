"use client";

import {
  ChatModelAdapter,
  ChatModelRunOptions,
  ThreadMessage,
  ChatModelRunResult,
  AssistantRuntimeProvider,
} from '@assistant-ui/react';
import { INTERNAL } from '@assistant-ui/react';

import { EdgeRuntimeOptions } from '@assistant-ui/react-edge';
const { splitLocalRuntimeOptions } = INTERNAL;
import { jsonrepair } from 'jsonrepair';
import {
  useLocalThreadRuntime,
  unstable_useRemoteThreadListRuntime as useRemoteThreadListRuntime,
  type unstable_RemoteThreadListAdapter,
} from '@assistant-ui/react';
import { ReactNode } from 'react'; // ✅ 添加这一行以导入 ReactNode
import { useChatOptions } from '../providers/chat';
import { StableProvider } from './stableProvider';
import { UploadAttachmentAdapter } from '../attachments/upload_attachment_adapter';

interface Props {
  children?: ReactNode;
}

type HeadersValue = Record<string, string> | Headers;

export type EdgeModelAdapterOptions = {
  api: string;
  /**
   * Callback function to be called when the API response is received.
   */
  onResponse?: (response: Response) => void | Promise<void>;
  /**
   * Optional callback function that is called when the assistant message is finished streaming.
   */
  onFinish?: (message: ThreadMessage) => void;
  /**
   * Callback function to be called when an error is encountered.
   */
  onError?: (error: Error) => void;

  credentials?: RequestCredentials;

  /**
   * Headers to be sent with the request.
   * Can be a static headers object or a function that returns a Promise of headers.
   */
  headers?: HeadersValue | (() => Promise<HeadersValue>);

  body?: object;
};

// This adapter connects LocalRuntime to your AI backend
function AddOrMergeToolCall(
  eventQueue: Array<{ type: string; data: any }>,
  toolCall: any,
): void {
  const existingIndex = eventQueue.findIndex(
    (item) => item.type === 'tool-call' && item.data.id === toolCall.id,
  );

  if (existingIndex !== -1) {
    // 更新已有条目
    const existing = eventQueue[existingIndex];
    existing.data = toolCall; // 合并逻辑
  } else {
    // 新增条目
    eventQueue.push({
      type: 'tool-call',
      data: toolCall,
    });
  }
}

export class MyModelAdapter implements ChatModelAdapter {
  constructor(private options: EdgeModelAdapterOptions) {}
  async *run({
    messages,
    runConfig,
    abortSignal,
    context,
    unstable_getMessage,
  }: ChatModelRunOptions) {
    const headersValue =
      typeof this.options.headers === 'function'
        ? await this.options.headers()
        : this.options.headers;

    const headers = new Headers(headersValue);
    headers.set('Content-Type', 'application/json');
    const enableAttachments = messages.some(
      (m) => (m.attachments ?? []).length > 0,
    );

    // load chat options
    const { enable_thinking, enable_search, mcp_ids, kb_ids } = useChatOptions();

    const result = await fetch(this.options.api, {
      method: 'POST',
      headers,
      credentials: this.options.credentials ?? 'same-origin',
      body: JSON.stringify({
        system: context.system,
        messages: messages,
        tools: [],
        runConfig,
        ...context.callSettings,
        ...context.config,

        enable_thinking: enable_thinking,
        enable_search: enable_search,
        mcp_ids: mcp_ids,
        kb_ids: kb_ids,
        enable_attachments: enableAttachments,
      }),
      signal: abortSignal,
    });

    await this.options.onResponse?.(result);
    if (!result.ok) {
      throw new Error(`Status ${result.status}: ${await result.text()}`);
    }
    if (!result.body) {
      throw new Error('Response body is null');
    }

    const reader = result.body.getReader();
    const decoder = new TextDecoder();
    let content = '';
    // let toolCalls: { [key: string]: any } = {};
    let buffer = '';

    const currentToolCallMap: {
      [key: string]: {
        id: string;
        type: string;
        function: { name: string; arguments: string };
        state?: string;
        result?: string;
      };
    } = {};

    const eventQueue: Array<{
      type: 'text' | 'tool-call';
      data: any;
    }> = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop()!; // 保留未闭合的行
      for (const line of lines) {
        if (line.startsWith('data:')) {
          const chunk = JSON.parse(line.slice(5));
          // 处理单条数据
          const delta = chunk.choices[0]?.delta;

          if (delta?.role === 'assistant' && delta?.content) {
            content += delta.content;
            if (
              eventQueue.length === 0 ||
              eventQueue[eventQueue.length - 1].type !== 'text'
            ) {
              eventQueue.push({
                type: 'text',
                data: content,
              });
            } else {
              // 更新最后一条文本内容
              eventQueue[eventQueue.length - 1].data = content;
            }
          }
          if (delta?.tool_calls) {
            for (const toolCall of delta.tool_calls) {
              const toolCallId = toolCall.id;
              if (!currentToolCallMap[toolCallId]) {
                currentToolCallMap[toolCallId] = {
                  id: toolCallId,
                  type: 'function',
                  function: {
                    name: toolCall.function?.name || '',
                    arguments: JSON.parse(
                      jsonrepair(toolCall.function?.arguments || '{}'),
                    ),
                  },
                  state: 'running',
                  result: undefined,
                };
              }
              // 更新 tool call
              if (toolCall.function?.name) {
                currentToolCallMap[toolCallId].function.name =
                  toolCall.function.name;
              }
              if (toolCall.function?.arguments) {
                // 使用 jsonrepair 修复 JSON 格式
                try {
                  const jsonr = jsonrepair(toolCall.function.arguments);
                  currentToolCallMap[toolCallId].function.arguments =
                    JSON.parse(jsonr || '{}');
                } catch (e) {
                  console.error('JSON parse error:', e);
                }
              }
              AddOrMergeToolCall(eventQueue, currentToolCallMap[toolCallId]);
            }
          }
          if (delta?.role === 'tool') {
            // 处理工具调用结果
            const keys = Object.keys(currentToolCallMap);
            const lastKey = keys[keys.length - 1];
            if (currentToolCallMap[lastKey]) {
              currentToolCallMap[lastKey].state = 'complete';
              currentToolCallMap[lastKey].result = delta.content;
            } else {
              console.warn(`Tool call with ID ${lastKey} not found.`);
            }
            AddOrMergeToolCall(eventQueue, currentToolCallMap[lastKey]);
          }

          // 生成结果
          yield {
            content: eventQueue
              .map((event) => {
                if (event.type === 'text') {
                  return {
                    type: 'text' as const,
                    text: event.data,
                  };
                } else if (event.type === 'tool-call') {
                  const toolCall = event.data;
                  return {
                    type: 'tool-call' as const,
                    toolCallId: toolCall.id,
                    toolName: toolCall.function.name,
                    args: toolCall.function.arguments,
                    state: toolCall.state,
                    result: toolCall.result,
                    isError: false,
                  };
                }
                return null;
              })
              .filter(Boolean),
          } as ChatModelRunResult;
        }
      }
    }

    this.options.onFinish?.(unstable_getMessage());
  }
}

let isInitializing = false;
let initializedThreadId = '';

function delay(ms: any) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Implement your custom adapter with proper message persistence
const myDatabaseAdapter: unstable_RemoteThreadListAdapter = {
  async list() {
    try {
      const res = await fetch('/v1/agent/threads');
      if (!res.ok) throw new Error('获取配置失败');
      const response = await res.json();
      return {
        threads: response.map((t: any) => ({
          status: t.archived ? 'archived' : 'regular',
          remoteId: t.id,
          title: t.title,
        })),
      };
    } catch (error) {
      console.error('Error fetching threads:', error);
      return { threads: [] };
    }
  },
  async initialize(threadId: string) {
    isInitializing = true;

    try {
      const url = '/v1/agent/threads';
      const now = new Date();
      const formattedTime = `${now.getFullYear()}-${String(
        now.getMonth() + 1,
      ).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(
        now.getHours(),
      ).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'PAI-RAG Assistant',
          title: `会话 - ${formattedTime}`, // 动态插入时间
          archived: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to create thread: ${response.statusText}`);
      }

      const data = await response.json();
      initializedThreadId = data.id;
      isInitializing = false;
      return {
        remoteId: data.id,
        externalId: data.id,
      };
    } catch (error) {
      console.error('Error creating thread:', error);
      throw error;
    }
  },
  async rename(remoteId, newTitle) {
    // await db.threads.update(remoteId, { title: newTitle });
    // const thread = mockThreads.find((t) => t.id === remoteId);
    // if (thread) thread.title = newTitle;
  },
  async archive(remoteId) {},
  async unarchive(remoteId) {},
  async delete(remoteId) {
    try {
      const res = await fetch(`/v1/agent/threads/${remoteId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!res.ok) {
        throw new Error('删除失败，请检查网络或配置');
      }
    } catch (err: any) {
      // 显示错误提示
      throw new Error('删除失败，请检查网络或配置');
    }
  },
  async generateTitle(remoteId, messages) {
    // Generate title from messages using your AI
    // const title = await generateTitle(messages);
    // await db.threads.update(remoteId, { title });
    // return new ReadableStream(); // Return empty stream

    // TODO:generateTitle
    // const thread = mockThreads.find((t) => t.id === remoteId);
    // if (thread) {
    //   thread.title = `AI 生成标题: ${messages[0]?.content.slice(0, 10) || "..."}`;
    // }
    return new ReadableStream(); // 返回空流
  },
};

export const usePaiChatThreadRuntime = (options: EdgeRuntimeOptions) => {
  const { localRuntimeOptions, otherOptions } =
    splitLocalRuntimeOptions(options);

  const runtime = useRemoteThreadListRuntime({
    runtimeHook: () => {
      return useLocalThreadRuntime(
        new MyModelAdapter(otherOptions),
        localRuntimeOptions,
      );
    },
    adapter: {
      ...myDatabaseAdapter,
      // The Provider component adds thread-specific adapters
      unstable_Provider: StableProvider,
    },
  });
  return runtime;
};


export function MyChatRuntimeProvider({ children }: { children: ReactNode }) {
  const runtime = usePaiChatThreadRuntime({
    api: '/v1/chat/completions',
    adapters: {
      attachments: new UploadAttachmentAdapter(),
    },
  });
  
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  )
}