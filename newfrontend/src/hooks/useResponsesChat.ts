import { useCallback, useRef, useState } from "react";
import { streamResponse } from "../api/client";
import { cancelResponse, streamResume } from "../api/responses";
import { getUserId } from "../lib/user";
import { useChatStore } from "../store/chat";
import { useConversationsStore } from "../store/conversations";
import {
  initialStreamState,
  reduceStreamEvent,
  type StreamState,
} from "../stream/reducer";
import type { ChatMessage } from "../types";

function tempId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2)}`;
}

function patchFromState(s: StreamState): Partial<ChatMessage> {
  return {
    text: s.message.text,
    reasoning: s.message.reasoning,
    reasoningStatus: s.message.reasoningStatus,
    toolCalls: s.message.toolCalls,
    status: s.message.status,
    responseId: s.message.responseId,
    usage: s.message.usage,
    error: s.message.error,
    lastSequenceNumber: s.message.lastSequenceNumber,
  };
}

export function useResponsesChat() {
  const [isStreaming, setIsStreaming] = useState(false);
  const currentResponseId = useRef<string | undefined>(undefined);
  const cancelPending = useRef(false);
  const resuming = useRef(false);
  const sendInFlight = useRef(false);
  const lastInput = useRef("");

  // Shared loop: fold each event into the store's last message; fire a deferred
  // cancel the moment the response id is known.
  const consume = useCallback(
    async (stream: AsyncIterable<unknown>, seed: StreamState) => {
      let state = seed;
      for await (const event of stream) {
        state = reduceStreamEvent(state, event as never);
        if (state.responseId) {
          currentResponseId.current = state.responseId;
          if (cancelPending.current) {
            cancelPending.current = false;
            void cancelResponse(state.responseId);
          }
        }
        useChatStore.getState().updateLast(patchFromState(state));
      }
      return state;
    },
    []
  );

  const finalize = useCallback((state: StreamState) => {
    const s = state.message.status;
    // A cancelled turn is a real, persisted, continuable response — advance
    // anchors exactly as for completed.
    if (s === "completed" || s === "cancelled") {
      useChatStore.getState().setAnchors({
        conversationId: state.conversationId,
        lastResponseId: state.responseId,
      });
      void useConversationsStore.getState().refresh();
    }
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const chat = useChatStore.getState();
      lastInput.current = trimmed;
      currentResponseId.current = undefined;
      cancelPending.current = false;

      const userMsg: ChatMessage = {
        id: tempId("user"),
        role: "user",
        text: trimmed,
        reasoning: "",
        reasoningStatus: "idle",
        status: "completed",
        toolCalls: [],
      };
      const assistantMsg: ChatMessage = {
        ...initialStreamState(tempId("assistant")).message,
      };
      chat.appendMessage(userMsg);
      chat.appendMessage(assistantMsg);
      chat.setStatus("streaming");
      setIsStreaming(true);
      sendInFlight.current = true;

      const controller = new AbortController();
      let state = initialStreamState(assistantMsg.id);
      try {
        const stream = streamResponse(
          {
            model: chat.model,
            input: trimmed,
            user_id: getUserId(),
            conversation: chat.conversationId,
            previous_response_id: chat.lastResponseId,
            background: true,
          } as never,
          controller.signal
        );
        state = await consume(stream, state);
      } catch (err) {
        setIsStreaming(false);
        useChatStore.getState().setStatus("idle");
        sendInFlight.current = false;
        if (!currentResponseId.current) {
          // The run never started server-side -> a real failure.
          useChatStore.getState().updateLast({
            status: "failed",
            error: err instanceof Error ? err.message : "stream error",
          });
        }
        // else: the run IS server-owned and resumable; leave the bubble "streaming"
        // so resumeIfInterrupted() recovers it on the next reconnect signal.
        return;
      }
      setIsStreaming(false);
      useChatStore.getState().setStatus("idle");
      sendInFlight.current = false;
      finalize(state);
    },
    [consume, finalize]
  );

  const stop = useCallback(() => {
    if (currentResponseId.current) {
      void cancelResponse(currentResponseId.current);
    } else {
      // response id not known yet — fire the cancel as soon as it arrives.
      cancelPending.current = true;
    }
  }, []);

  const regenerate = useCallback(async () => {
    if (lastInput.current) await send(lastInput.current);
  }, [send]);

  const resumeIfInterrupted = useCallback(async () => {
    if (resuming.current || sendInFlight.current) return;
    const chat = useChatStore.getState();
    const last = chat.messages[chat.messages.length - 1];
    if (
      !last ||
      last.role !== "assistant" ||
      last.status !== "streaming" ||
      !last.responseId
    ) {
      return;
    }
    resuming.current = true;
    setIsStreaming(true);
    currentResponseId.current = last.responseId;
    const controller = new AbortController();
    const seed: StreamState = {
      message: last,
      responseId: last.responseId,
      conversationId: chat.conversationId,
      lastSequenceNumber: last.lastSequenceNumber ?? 0,
    };
    let state = seed;
    try {
      const stream = streamResume(
        last.responseId,
        seed.lastSequenceNumber,
        controller.signal
      );
      state = await consume(stream, state);
    } catch {
      // 409 (evicted/finished) or a network error: leave the bubble as-is.
      // The run finished server-side; a conversation reload shows the final.
      resuming.current = false;
      setIsStreaming(false);
      return;
    }
    resuming.current = false;
    setIsStreaming(false);
    useChatStore.getState().setStatus("idle");
    finalize(state);
  }, [consume, finalize]);

  return { send, stop, regenerate, isStreaming, resumeIfInterrupted };
}
