import {
  ChatModelAdapter,
  ChatModelRunOptions,
  ThreadMessage,
  ChatModelRunResult,
} from "@assistant-ui/react";
import {
  AssistantRuntime,
  INTERNAL,
  useLocalRuntime,
} from "@assistant-ui/react";
import { EdgeRuntimeOptions } from "@assistant-ui/react-edge";
const { splitLocalRuntimeOptions } = INTERNAL;
import { jsonrepair } from "jsonrepair";

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
  console.log("Current eventQueue:", eventQueue);
  const existingIndex = eventQueue.findIndex(
    (item) => item.type === "tool-call" && item.data.id === toolCall.id,
  );

  if (existingIndex !== -1) {
    // 更新已有条目
    const existing = eventQueue[existingIndex];
    existing.data = toolCall; // 合并逻辑
  } else {
    // 新增条目
    eventQueue.push({
      type: "tool-call",
      data: toolCall,
    });
  }
  console.log("Updated eventQueue:", eventQueue);
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
      typeof this.options.headers === "function"
        ? await this.options.headers()
        : this.options.headers;

    const headers = new Headers(headersValue);
    headers.set("Content-Type", "application/json");
    console.log("messages", messages);
    const enableAttachments = messages.some(
      (m) => (m.attachments ?? []).length > 0,
    );

    const lastMsg = messages[messages.length - 1];
    const extractedAttachments =
      lastMsg.attachments?.map((attachment) => ({
        id: attachment.id,
        type: attachment.type,
        name: attachment.name,
      })) ?? [];
    const newLastMessage = {
      role: "user",
      id: lastMsg.id,
      content: lastMsg.content,
      metadata: lastMsg.metadata,
      createdAt: lastMsg.createdAt,
      status: lastMsg.status,
    };
    const formattedMessages = [...messages.slice(0, -1), newLastMessage];

    const result = await fetch(this.options.api, {
      method: "POST",
      headers,
      credentials: this.options.credentials ?? "same-origin",
      body: JSON.stringify({
        system: context.system,
        messages: formattedMessages,
        tools: [],
        runConfig,
        ...context.callSettings,
        ...context.config,

        ...this.options.body,
        enable_attachments: enableAttachments,
        attachments: extractedAttachments,
      }),
      signal: abortSignal,
    });

    await this.options.onResponse?.(result);
    if (!result.ok) {
      throw new Error(`Status ${result.status}: ${await result.text()}`);
    }
    if (!result.body) {
      throw new Error("Response body is null");
    }

    const reader = result.body.getReader();
    const decoder = new TextDecoder();
    let content = "";
    // let toolCalls: { [key: string]: any } = {};
    let buffer = "";

    let currentToolCallMap: {
      [key: string]: {
        id: string;
        type: string;
        function: { name: string; arguments: string };
        state?: string;
        result?: string;
      };
    } = {};

    const eventQueue: Array<{
      type: "text" | "tool-call";
      data: any;
    }> = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!; // 保留未闭合的行
      for (const line of lines) {
        if (line.startsWith("data:")) {
          const chunk = JSON.parse(line.slice(5));
          // 处理单条数据
          console.log("chunk", chunk);
          const delta = chunk.choices[0]?.delta;

          if (delta?.role === "assistant" && delta?.content) {
            content += delta.content;
            if (
              eventQueue.length === 0 ||
              eventQueue[eventQueue.length - 1].type !== "text"
            ) {
              eventQueue.push({
                type: "text",
                data: content,
              });
            } else {
              // 更新最后一条文本内容
              eventQueue[eventQueue.length - 1].data = content;
            }
          }
          if (delta?.tool_calls) {
            for (const toolCall of delta.tool_calls) {
              console.log("toolCall", toolCall);
              const toolCallId = toolCall.id;
              if (!currentToolCallMap[toolCallId]) {
                currentToolCallMap[toolCallId] = {
                  id: toolCallId,
                  type: "function",
                  function: {
                    name: toolCall.function?.name || "",
                    arguments: JSON.parse(
                      jsonrepair(toolCall.function?.arguments || "{}"),
                    ),
                  },
                  state: "running",
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
                    JSON.parse(jsonr || "{}");
                } catch (e) {
                  console.error("JSON parse error:", e);
                }
              }
              AddOrMergeToolCall(eventQueue, currentToolCallMap[toolCallId]);
            }
          }
          if (delta?.role === "tool") {
            // 处理工具调用结果
            console.log("Tool call result:", delta);
            const keys = Object.keys(currentToolCallMap);
            const lastKey = keys[keys.length - 1];
            if (currentToolCallMap[lastKey]) {
              currentToolCallMap[lastKey].state = "complete";
              currentToolCallMap[lastKey].result = delta.content;
            } else {
              console.warn(`Tool call with ID ${lastKey} not found.`);
            }
            AddOrMergeToolCall(eventQueue, currentToolCallMap[lastKey]);
          }
          console.log("Current tool call list:", currentToolCallMap);

          // 生成结果
          yield {
            content: eventQueue
              .map((event) => {
                if (event.type === "text") {
                  return {
                    type: "text" as const,
                    text: event.data,
                  };
                } else if (event.type === "tool-call") {
                  const toolCall = event.data;
                  return {
                    type: "tool-call" as const,
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
export const usePaiChatRuntime = (
  options: EdgeRuntimeOptions,
): AssistantRuntime => {
  const { localRuntimeOptions, otherOptions } =
    splitLocalRuntimeOptions(options);

  return useLocalRuntime(new MyModelAdapter(otherOptions), localRuntimeOptions);
};
