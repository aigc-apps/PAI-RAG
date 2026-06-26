import { useCallback, useRef, useState } from "react";
import { streamResponse } from "../api/client";
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

export function useResponsesChat() {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const lastInputRef = useRef<string>("");

  const runTurn = useCallback(
    async (
      text: string,
      anchors: { conversation?: string; previousResponseId?: string }
    ) => {
      const chat = useChatStore.getState();
      lastInputRef.current = text;

      const userMsg: ChatMessage = {
        id: tempId("user"),
        role: "user",
        text,
        reasoning: "",
        reasoningStatus: "idle",
        status: "completed",
      };
      const assistantMsg: ChatMessage = {
        ...initialStreamState(tempId("assistant")).message,
      };
      chat.appendMessage(userMsg);
      chat.appendMessage(assistantMsg);
      chat.setStatus("streaming");
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;
      let state: StreamState = initialStreamState(assistantMsg.id);
      let aborted = false;

      try {
        const stream = streamResponse(
          {
            model: chat.model,
            input: text,
            user_id: getUserId(),
            conversation: anchors.conversation,
            previous_response_id: anchors.previousResponseId,
          },
          controller.signal
        );
        for await (const event of stream) {
          state = reduceStreamEvent(state, event);
          useChatStore.getState().updateLast({
            text: state.message.text,
            reasoning: state.message.reasoning,
            reasoningStatus: state.message.reasoningStatus,
            status: state.message.status,
            responseId: state.message.responseId,
            usage: state.message.usage,
            error: state.message.error,
          });
        }
      } catch (err) {
        if (controller.signal.aborted) {
          aborted = true;
        } else {
          useChatStore.getState().updateLast({
            status: "failed",
            error: err instanceof Error ? err.message : "stream error",
          });
        }
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
        useChatStore.getState().setStatus("idle");
      }

      // Also handle the case where the generator completed without throwing
      // (e.g., a mock that doesn't check the abort signal itself).
      if (!aborted && controller.signal.aborted) {
        aborted = true;
      }

      if (aborted) {
        // Stop is local-only: keep the partial bubble, do NOT advance anchors.
        useChatStore.getState().updateLast({ status: "stopped" });
        return;
      }

      if (state.message.status === "completed") {
        useChatStore.getState().setAnchors({
          conversationId: state.conversationId,
          lastResponseId: state.responseId,
        });
        void useConversationsStore.getState().refresh();
      }
    },
    []
  );

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const chat = useChatStore.getState();
      await runTurn(trimmed, {
        conversation: chat.conversationId,
        previousResponseId: chat.lastResponseId,
      });
    },
    [runTurn]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const regenerate = useCallback(async () => {
    const text = lastInputRef.current;
    if (!text) return;
    const chat = useChatStore.getState();
    // Re-send the last input with the anchors that produced the prior turn.
    await runTurn(text, {
      conversation: chat.conversationId,
      previousResponseId: chat.lastResponseId,
    });
  }, [runTurn]);

  return { send, stop, regenerate, isStreaming };
}
