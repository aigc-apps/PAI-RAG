import { useCallback, useEffect, useRef } from "react";
import { streamResponse } from "../api/client";
import { cancelResponse, streamResume } from "../api/responses";
import { getConversation, truncateLastTurn } from "../api/conversations";
import { activeRuntime, useChatStore } from "../store/chat";
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

/**
 * Per-conversation local stream control, keyed by the runtime's stable local id.
 * An entry exists exactly while a local loop (send OR resume) is folding events
 * into that conversation — so `streams.has(key)` is the single source of truth
 * for "a loop is running here", and `controller.abort()` stops the local reader
 * on switch-away (the run keeps going server-side because it's `background`).
 * Kept in a module Map, not in zustand, because AbortController isn't state.
 */
interface StreamCtl {
  controller: AbortController;
  responseId?: string;
  cancelPending: boolean;
}
const streams = new Map<string, StreamCtl>();

export function useResponsesChat() {
  const activeKey = useChatStore((s) => s.activeKey);
  const isStreaming = useChatStore(
    (s) => activeRuntime(s)?.status === "streaming"
  );
  const prevKeyRef = useRef(activeKey);
  const lastInput = useRef("");

  // Shared loop: fold each event into runtime `key`'s last message; fire a
  // deferred cancel once the response id is known; learn the server
  // conversationId as soon as it arrives so the runtime is findable on a later
  // switch-back and the sidebar can highlight it.
  const consume = useCallback(
    async (stream: AsyncIterable<unknown>, seed: StreamState, key: string) => {
      let state = seed;
      for await (const event of stream) {
        state = reduceStreamEvent(state, event as never);
        const ctl = streams.get(key);
        if (state.responseId && ctl) {
          ctl.responseId = state.responseId;
          if (ctl.cancelPending) {
            ctl.cancelPending = false;
            void cancelResponse(state.responseId);
          }
        }
        if (state.conversationId) {
          const rt = useChatStore.getState().runtimes[key];
          if (rt && rt.conversationId !== state.conversationId) {
            useChatStore
              .getState()
              .setAnchorsOf(key, { conversationId: state.conversationId });
            if (useChatStore.getState().activeKey === key) {
              useConversationsStore.getState().select(state.conversationId);
            }
          }
        }
        useChatStore.getState().updateLastOf(key, patchFromState(state));
      }
      return state;
    },
    []
  );

  const finalize = useCallback((state: StreamState, key: string) => {
    const s = state.message.status;
    // A cancelled turn is a real, persisted, continuable response — advance
    // anchors exactly as for completed.
    if (s === "completed" || s === "cancelled") {
      useChatStore.getState().setAnchorsOf(key, {
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
      const store = useChatStore.getState();
      const key = store.activeKey;
      const rt = store.runtimes[key];
      lastInput.current = trimmed;

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
      store.appendMessage(key, userMsg);
      store.appendMessage(key, assistantMsg);
      store.setStatusOf(key, "streaming");

      const controller = new AbortController();
      streams.set(key, { controller, responseId: undefined, cancelPending: false });

      let state = initialStreamState(assistantMsg.id);
      try {
        const stream = streamResponse(
          {
            model: store.model,
            agent_id: store.agentId || undefined,
            input: trimmed,
            conversation: rt?.conversationId,
            previous_response_id: rt?.lastResponseId,
            background: true,
          },
          controller.signal
        );
        state = await consume(stream, state, key);
      } catch (err) {
        const hadId = !!streams.get(key)?.responseId;
        const aborted = controller.signal.aborted;
        streams.delete(key);
        useChatStore.getState().setStatusOf(key, "idle");
        if (!hadId && !aborted) {
          // The run never started server-side -> a real failure.
          useChatStore.getState().updateLastOf(key, {
            status: "failed",
            error: err instanceof Error ? err.message : "stream error",
          });
        }
        // Otherwise the run IS server-owned and resumable (or we aborted to switch
        // away): leave the bubble "streaming" so a switch-back / reconnect resumes.
        return;
      }
      streams.delete(key);
      useChatStore.getState().setStatusOf(key, "idle");
      finalize(state, key);
    },
    [consume, finalize]
  );

  const stop = useCallback(() => {
    const key = useChatStore.getState().activeKey;
    const ctl = streams.get(key);
    if (!ctl) return;
    if (ctl.responseId) {
      void cancelResponse(ctl.responseId);
    } else {
      // response id not known yet — fire the cancel as soon as it arrives.
      ctl.cancelPending = true;
    }
  }, []);

  const regenerate = useCallback(async () => {
    const key = useChatStore.getState().activeKey;
    if (streams.has(key)) return; // a loop is already running for this conversation
    const rt = useChatStore.getState().runtimes[key];
    if (!rt) return;
    // Re-run the most recent user turn. Source the prompt from the runtime's last
    // user message rather than the in-memory `lastInput` ref — the ref is only
    // populated by an in-session send(), so after a reload or opening a saved
    // conversation it is empty and the button would silently no-op.
    const msgs = rt.messages;
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
    if (rt.conversationId && respId && userIdx >= 0) {
      try {
        const { previous_response_id } = await truncateLastTurn(
          rt.conversationId,
          respId
        );
        useChatStore
          .getState()
          .dropLastTurnOf(key, userIdx, previous_response_id ?? undefined);
      } catch {
        // couldn't truncate — leave messages as-is and just resend (appends)
      }
    }
    await send(text);
  }, [send]);

  const resumeIfInterrupted = useCallback(async () => {
    const store = useChatStore.getState();
    const key = store.activeKey;
    if (streams.has(key)) return; // a loop is already running for this conversation
    const rt = store.runtimes[key];
    const last = rt?.messages[rt.messages.length - 1];
    if (
      !rt ||
      !last ||
      last.role !== "assistant" ||
      last.status !== "streaming" ||
      !last.responseId
    ) {
      return;
    }
    const controller = new AbortController();
    streams.set(key, {
      controller,
      responseId: last.responseId,
      cancelPending: false,
    });
    store.setStatusOf(key, "streaming");
    const seed: StreamState = {
      message: last,
      responseId: last.responseId,
      conversationId: rt.conversationId,
      lastSequenceNumber: last.lastSequenceNumber ?? 0,
    };
    let state = seed;
    try {
      const stream = streamResume(
        last.responseId,
        seed.lastSequenceNumber,
        controller.signal
      );
      state = await consume(stream, state, key);
    } catch {
      const aborted = controller.signal.aborted;
      streams.delete(key);
      useChatStore.getState().setStatusOf(key, "idle");
      // Aborted == we switched away on purpose; leave the bubble as-is.
      if (aborted) return;
      // 409 (evicted/finished) or a transport drop: if we know the server
      // conversation, reload its final persisted state so a finished-in-
      // background run shows its answer instead of a stuck "streaming" bubble.
      // Only hydrate if it's still the conversation on screen.
      const cid = useChatStore.getState().runtimes[key]?.conversationId;
      if (cid && useChatStore.getState().activeKey === key) {
        try {
          useChatStore.getState().hydrate(await getConversation(cid));
        } catch {
          /* leave as-is; a manual reload recovers it */
        }
      }
      return;
    }
    streams.delete(key);
    useChatStore.getState().setStatusOf(key, "idle");
    finalize(state, key);
  }, [consume, finalize]);

  // On switching the active conversation: abort the outgoing conversation's local
  // loop (its run keeps going server-side), then try to resume the new one from
  // its cursor (续传). Fires only on an actual switch — resume on first mount /
  // tab re-focus is owned by ChatView's visibility effect.
  useEffect(() => {
    const prev = prevKeyRef.current;
    if (prev === activeKey) return;
    prevKeyRef.current = activeKey;
    streams.get(prev)?.controller.abort();
    void resumeIfInterrupted();
  }, [activeKey, resumeIfInterrupted]);

  return { send, stop, regenerate, isStreaming, resumeIfInterrupted };
}
