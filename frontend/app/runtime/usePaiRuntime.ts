import {
  ChatModelAdapter,
  ChatModelRunOptions,
  ThreadMessage,
} from "@assistant-ui/react";
import {
  AssistantMessageAccumulator,
  DataStreamDecoder,
  unstable_toolResultStream,
} from "assistant-stream";
import { asAsyncIterableStream } from "assistant-stream/utils";
import {
  AssistantRuntime,
  INTERNAL,
  useLocalRuntime,
} from "@assistant-ui/react";
import { EdgeRuntimeOptions } from "@assistant-ui/react-edge";
const { splitLocalRuntimeOptions } = INTERNAL;

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

    try {
      if (!result.ok) {
        throw new Error(`Status ${result.status}: ${await result.text()}`);
      }
      if (!result.body) {
        throw new Error("Response body is null");
      }

      const stream = result.body
        .pipeThrough(new DataStreamDecoder())
        .pipeThrough(unstable_toolResultStream(context.tools, abortSignal))
        .pipeThrough(new AssistantMessageAccumulator());

      yield* asAsyncIterableStream(stream);

      this.options.onFinish?.(unstable_getMessage());
    } catch (error: unknown) {
      this.options.onError?.(error as Error);
      throw error;
    }
  }
}
export const usePaiRuntime = (
  options: EdgeRuntimeOptions,
): AssistantRuntime => {
  const { localRuntimeOptions, otherOptions } =
    splitLocalRuntimeOptions(options);

  return useLocalRuntime(new MyModelAdapter(otherOptions), localRuntimeOptions);
};
