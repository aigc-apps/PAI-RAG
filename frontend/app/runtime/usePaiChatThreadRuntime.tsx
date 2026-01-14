"use client";

import {
  ChatModelAdapter,
  ChatModelRunOptions,
  ThreadMessage,
  ChatModelRunResult,
  AssistantRuntimeProvider,
  useThreadListItem,
  ThreadHistoryAdapter,
  ExportedMessageRepository,
  RuntimeAdapterProvider,
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
import { ReactNode, useMemo } from 'react'; // ✅ 添加这一行以导入 ReactNode
import { useChatOptions } from '../providers/chat';
import { UploadAttachmentAdapter } from '../attachments/upload_attachment_adapter';
import { v4 as uuidv4 } from 'uuid';
import { AssistantStream, PlainTextDecoder } from "assistant-stream";
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

interface Props {
  children?: ReactNode;
}

type HeadersValue = Record<string, string> | Headers;
let isInitializing = false;
let initializedThreadId = "";
let msgParentIdMap = new Map<string, string>();


function toByteStream(data: string | Buffer): ReadableStream<Uint8Array> {
  let uint8: Uint8Array;

  if (typeof data === 'string') {
    uint8 = new TextEncoder().encode(data);
  } else {
    uint8 = data; // Buffer is Uint8Array
  }

  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(uint8);
      controller.close();
    }
  });
}

function delay(ms: any) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
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

  /**
   * Tenant-aware fetch function for API calls
   */
  tenantFetch?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
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

    const fetchFn = this.options.tenantFetch ?? fetch;
    const result = await fetchFn(this.options.api, {
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
        ...this.options.body,
        stream: true,
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
    let content = "";
    let reasoning_content = "";
    // let toolCalls: { [key: string]: any } = {};
    let buffer = '';

    const currentToolCallMap: {
      [key: string]: {
        id: string;
        type: string;
        function: { name: string; arguments: string };
        state?: string;
        result?: string;
        isError: boolean;
      };
    } = {};

    const eventQueue: Array<{
      type: "text" | "tool-call" | "reasoning";
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

          if (delta.reasoning_completed) {
            // 思考完成，清空 reasoning_content
            reasoning_content = "";
          }

          if (delta?.role === "assistant" && delta?.reasoning_content) {
            reasoning_content += delta.reasoning_content;
            if (eventQueue.length === 0 || eventQueue[eventQueue.length - 1].type !== "reasoning") {
              eventQueue.push({
                type: "reasoning",
                data: reasoning_content,
              })
            } else {
              // 更新最后一条思考内容
              eventQueue[eventQueue.length - 1].data = reasoning_content;
            }
          }

          if (delta?.role === 'assistant' && delta?.content) {
            if (chunk.safety_violation) {
              content = delta.content
            }
            else {
              content += delta.content;
            }

            if (eventQueue.length === 0 || eventQueue[eventQueue.length - 1].type !== 'text'
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
          if (chunk?.actions) {
            reasoning_content = ""; // 清空思考内容
            for (const toolCall of chunk.actions) {
              const toolCallId = toolCall.id;
              const toolCallName = toolCall.function?.name?.replace(/search-knowledgebase.*/, "search-knowledgebase");
              if (!currentToolCallMap[toolCallId]) {
                currentToolCallMap[toolCallId] = {
                  id: toolCallId,
                  type: 'function',
                  function: {
                    name: toolCallName || '',
                    arguments: JSON.parse(
                      jsonrepair(toolCall.function?.arguments || '{}'),
                    ),
                  },
                  state: 'running',
                  result: undefined,
                  isError: false,
                };
              }
              // 更新 tool call
              if (toolCallName) {
                currentToolCallMap[toolCallId].function.name =
                  toolCallName;
              }
              if (toolCall.function?.arguments) {
                // 使用 jsonrepair 修复 JSON 格式
                try {
                  const argumentsStr = toolCall.function.arguments;
                  if (argumentsStr && typeof argumentsStr === 'string') {
                    try {
                      // 尝试使用 jsonrepair 修复 JSON
                      const jsonr = jsonrepair(argumentsStr);
                      currentToolCallMap[toolCallId].function.arguments =
                        JSON.parse(jsonr || '{}');
                    } catch (jsonrepairError) {
                      console.warn('jsonrepair failed, trying direct JSON parse:', jsonrepairError);
                      // 如果 jsonrepair 失败，尝试直接解析
                      try {
                        currentToolCallMap[toolCallId].function.arguments = JSON.parse(argumentsStr);
                      } catch (parseError) {
                        console.warn('Direct JSON parse also failed, using empty object string:', parseError);
                        currentToolCallMap[toolCallId].function.arguments = '{}';
                      }
                    }
                  } else {
                    // 如果 arguments 不是字符串，转换为字符串
                    currentToolCallMap[toolCallId].function.arguments = 
                      typeof argumentsStr === 'object' ? JSON.stringify(argumentsStr) : String(argumentsStr || '{}');
                  }
                } catch (e) {
                  console.error('Unexpected error processing arguments:', e);
                  // 如果解析失败，使用空对象字符串作为默认值
                  currentToolCallMap[toolCallId].function.arguments = '{}';
                }
              }
              AddOrMergeToolCall(eventQueue, currentToolCallMap[toolCallId]);
            }
          }
          if (chunk?.observation) {
            // 处理工具调用结果
            // 使用 observation.tool.id 来正确匹配工具调用，而不是假设是最后一个
            const toolCallId = chunk?.observation?.tool?.id;
            if (toolCallId && currentToolCallMap[toolCallId]) {
              currentToolCallMap[toolCallId].state = 'complete';
              currentToolCallMap[toolCallId].result = chunk?.observation?.result || chunk?.observation?.error;
              currentToolCallMap[toolCallId].isError = chunk?.observation?.error != null && chunk?.observation?.error.trim() !== '';
              AddOrMergeToolCall(eventQueue, currentToolCallMap[toolCallId]);
            } else {
              console.warn(`Tool call with ID ${toolCallId} not found in observation.`, {
                toolCallId,
                availableIds: Object.keys(currentToolCallMap),
                observation: chunk?.observation
              });
            }
            content = ""; // 重置content
          }

          // 生成结果
          yield {
            content: eventQueue
              .map((event) => {
                if (event.type === "reasoning") {
                  return {
                    type: "reasoning" as const,
                    text: event.data,
                  };
                } else if (event.type === "text") {
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
                    argsText: JSON.stringify(toolCall.function.arguments),
                    args: toolCall.function.arguments,
                    state: toolCall.state,
                    result: toolCall.result,
                    isError: toolCall.isError,
                  };
                }
                return null;
              })
              .filter(Boolean),
          } as ChatModelRunResult;
        }
      }
      
      // 流结束时，检查是否有未完成的工具调用，将它们从eventQueue中移除（隐藏）
      const cancelledToolCallIds = new Set<string>();
      for (const toolCallId in currentToolCallMap) {
        const toolCall = currentToolCallMap[toolCallId];
        if (toolCall.state === 'running') {
          // 标记为已取消，但不添加到eventQueue，这样前端就不会显示
          cancelledToolCallIds.add(toolCallId);
        }
      }
      
      // 从eventQueue中移除已取消的工具调用
      const filteredEventQueue = eventQueue.filter((event) => {
        if (event.type === 'tool-call') {
          return !cancelledToolCallIds.has(event.data.id);
        }
        return true;
      });
      
      // 发送最终结果，已取消的工具调用已被过滤掉
      if (filteredEventQueue.length > 0) {
        yield {
          content: filteredEventQueue
            .map((event) => {
              if (event.type === "reasoning") {
                return {
                  type: "reasoning" as const,
                  text: event.data,
                };
              } else if (event.type === "text") {
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
                  argsText: JSON.stringify(toolCall.function.arguments),
                  args: toolCall.function.arguments,
                  state: toolCall.state,
                  result: toolCall.result,
                  isError: toolCall.isError,
                };
              }
              return null;
            })
            .filter(Boolean),
        } as ChatModelRunResult;
      }
    }

    this.options.onFinish?.(unstable_getMessage());
  }
}

// 创建 adapter 的工厂函数，接收 tenantFetch 作为参数
function createDatabaseAdapter(tenantFetch: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>): unstable_RemoteThreadListAdapter {
  return {
    async list() {
      try {
        const res = await tenantFetch(`/api/threads`);
        if (!res.ok) throw new Error('获取配置失败');
        const response = await res.json();
        return {
          threads: response.data.map((t: any) => ({
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
        const url = `/api/threads`;
        const now = new Date();
        const formattedTime = `${now.getFullYear()}-${String(
          now.getMonth() + 1,
        ).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(
          now.getHours(),
        ).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

        const response = await tenantFetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: 'PAI-RAG Assistant',
            title: `会话 - ${formattedTime}`, // 动态插入时间
            archived: false,
          }),
        });

        if (!response.ok) {
          isInitializing = false;
          throw new Error(`Failed to create thread: ${response.statusText}`);
        }

        const result = await response.json();
        initializedThreadId = result.data.id;

        return {
          remoteId: result.data.id,
          externalId: result.data.id,
        };
      } catch (error) {
        console.error('Error creating thread:', error);
        throw error;
      }
      finally {
        isInitializing = false;
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
        const res = await tenantFetch(`/api/threads/${remoteId}`, {
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
      try {
        const res = await tenantFetch(`/api/threads/${remoteId}/title`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(messages), 
        });
        if (!res.ok) {
          throw new Error('生成标题失败，请检查网络或配置');
        }
        const data = await res.json();
        return AssistantStream.fromByteStream(toByteStream(data.data.title) as ReadableStream<Uint8Array<ArrayBuffer>>, new PlainTextDecoder());

      } catch (err: any) {
        // 显示错误提示
        throw new Error('生成标题失败，请检查网络或配置');
      }
    },
  };
}


export const StableProvider: React.ComponentType<{ children?: React.ReactNode }> = ({
  children,
}) => {
  // This runs in the context of each thread
  const threadListItem = useThreadListItem();
  const remoteId = threadListItem.remoteId;
  const { tenantFetch } = useTenantFetch();

  // Create thread-specific history adapter
  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      async load() {
        if (!remoteId) return { headId: null, messages: [] };
        // 模拟从后端获取数据
        try {
          const res = await tenantFetch(`/api/threads/${remoteId}/messages`);

          if (!res.ok) throw new Error('获取配置失败');
          const result = await res.json();
          const messages = result.data;
          if (messages.length === 0) {
            return { headId: null, messages: [] };
          }
          msgParentIdMap.clear();

          let parentId = "";
          for (let i = 0; i < messages.length; i ++) {
            if (messages[i].local_id !== undefined && messages[i].local_id !== '')
            {
              msgParentIdMap.set(parentId, messages[i].local_id);
            } 
            parentId = messages[i].id;
          }
          console.log("load messages", messages, msgParentIdMap);

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
          while (isInitializing) {
            console.log(
              "thread isInitializing",
              isInitializing,
              initializedThreadId,
            );
            await delay(100);
          }
          console.log("initialized remoteId", initializedThreadId);
        }

        const remoteThreadId = remoteId ? remoteId : initializedThreadId;

        try {
          const url = `/api/threads/${remoteThreadId}/messages`;
          console.log('append message', message);
          
          let msgId = message.message.id;
          const pid = message.parentId || "";
          if (msgParentIdMap.has(pid)) {
            msgId = msgParentIdMap.get(pid) || "";
          }
          else {
            msgParentIdMap.set(pid, msgId);
          }
          const response = await tenantFetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              thread_id: remoteThreadId,
              role: message.message.role,
              attachments: message.message.attachments,
              content: message.message.content,
              local_id: msgId,
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
    [remoteId, tenantFetch],
  );
  const adapters = useMemo(() => ({ history }), [history]);
  return (
    <RuntimeAdapterProvider adapters={adapters}>
      {children}
    </RuntimeAdapterProvider>
  );
};

export const usePaiChatThreadRuntime = (options: EdgeRuntimeOptions) => {
  const { localRuntimeOptions, otherOptions } =
    splitLocalRuntimeOptions(options);

  // load chat options
  const { model, enable_agent, enable_search, enable_chatdb, mcp_ids, kb_ids, user_id } = useChatOptions();
  const { tenantFetch } = useTenantFetch();

  // 创建带有 tenantFetch 的 adapter
  const databaseAdapter = useMemo(() => createDatabaseAdapter(tenantFetch), [tenantFetch]);

  const runtime = useRemoteThreadListRuntime({
    runtimeHook: () => {
      return useLocalThreadRuntime(
        new MyModelAdapter({...otherOptions, tenantFetch, body: { model, enable_agent, enable_search, enable_chatdb, mcp_ids, kb_ids, user_id }}),
        localRuntimeOptions,
      );
    },
    adapter: {
      ...databaseAdapter,
      // The Provider component adds thread-specific adapters
      unstable_Provider: StableProvider,
    },
  });
  return runtime;
};


export function MyChatRuntimeProvider({ children }: { children: ReactNode }) {
  const { tenantFetch } = useTenantFetch();
  
  const runtime = usePaiChatThreadRuntime({
    api: `/api/chat/completions`,
    adapters: {
      attachments: useMemo(() => new UploadAttachmentAdapter(tenantFetch), [tenantFetch]),
    },
  });
  
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  )
}