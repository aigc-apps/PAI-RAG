import { create } from "zustand";

/**
 * A global slot for "send a message as the user" so deep components — the
 * inline HITL authorization card and its dialog — can resume a paused agent
 * turn without threading the chat hook through the tree. {@link ChatView}
 * registers `useResponsesChat().send` here on mount; the card's "继续" button
 * calls `submit(...)`.
 */
interface ComposerState {
  submit: ((text: string) => void) | null;
  setSubmit: (fn: ((text: string) => void) | null) => void;
}

export const useComposer = create<ComposerState>((set) => ({
  submit: null,
  setSubmit: (fn) => set({ submit: fn }),
}));
