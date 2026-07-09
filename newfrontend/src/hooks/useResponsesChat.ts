import { useCallback, useRef, useState } from "react";
import { streamResponse } from "../api/client";
import { cancelResponse, streamResume } from "../api/responses";
import { truncateLastTurn } from "../api/conversations";
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
    steps: s.message.steps,
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
            agent_id: chat.agentId || undefined,
            input: trimmed,
            conversation: chat.conversationId,
            previous_response_id: chat.lastResponseId,
            background: true,
          },
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
    if (sendInFlight.current || resuming.current) return;
    // Re-run the most recent user turn. Source the prompt from the store's last
    // user message rather than the in-memory `lastInput` ref — the ref is only
    // populated by an in-session send(), so after a reload or opening a saved
    // conversation it is empty and the button would silently no-op.
    const chat = useChatStore.getState();
    const msgs = chat.messages;
    let userIdx = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user" && msgs[i].text.trim()) {
        userIdx = i;
        break;
      }
    }
    const text = (userIdx >= 0 ? msgs[userIdx].text : lastInput.current).trim();
    if (!text) return;

    // The server keeps the full transcript, so to truly regenerate (replace the
    // last answer, not append a duplicate turn) we drop the last turn on the
    // server first, then re-run the same prompt. If that fails (turn already
    // gone / not the tail / never persisted) we fall back to a plain resend.
    let respId: string | undefined;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant" && msgs[i].responseId) {
        respId = msgs[i].responseId;
        break;
      }
    }
    if (chat.conversationId && respId && userIdx >= 0) {
      try {
        const { previous_response_id } = await truncateLastTurn(
          chat.conversationId,
          respId
        );
        useChatStore
          .getState()
          .dropLastTurn(userIdx, previous_response_id ?? undefined);
      } catch {
        // couldn't truncate — leave messages as-is and just resend (appends)
      }
    }
    await send(text);
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
