import { create } from "zustand";
import type { ConversationSummary } from "../types";
import { listConversations, deleteConversation } from "../api/conversations";
import { getUserId } from "../lib/user";

interface ConversationsState {
  items: ConversationSummary[];
  selectedId?: string;
  refresh: () => Promise<void>;
  select: (id: string) => void;
  clearSelection: () => void;
  remove: (id: string) => Promise<void>;
}

export const useConversationsStore = create<ConversationsState>((set) => ({
  items: [],
  selectedId: undefined,

  refresh: async () => {
    const items = await listConversations(getUserId());
    set({ items });
  },

  select: (id) => set({ selectedId: id }),
  clearSelection: () => set({ selectedId: undefined }),

  remove: async (id) => {
    await deleteConversation(id);
    set((s) => ({
      items: s.items.filter((c) => c.id !== id),
      selectedId: s.selectedId === id ? undefined : s.selectedId,
    }));
  },
}));
