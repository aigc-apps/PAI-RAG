import { create } from "zustand";
import type { FileArtifact } from "../types";

interface PreviewState {
  /** The artifacts available in the right-side panel (a whole message's files),
   * empty when the panel is closed. */
  items: FileArtifact[];
  /** Which artifact is currently shown; falls back to items[0] when stale. */
  activeId: string | null;
  /** Maximized: the panel fills the area right of the sidebar, hiding the chat. */
  expanded: boolean;
  /** Open the panel on a set of artifacts, focusing `activeId` (or the first). */
  open: (items: FileArtifact[], activeId?: string) => void;
  /** Switch the shown artifact without changing the set. */
  setActive: (id: string) => void;
  /** Toggle maximize/restore. */
  toggleExpanded: () => void;
  close: () => void;
}

export const usePreviewStore = create<PreviewState>((set) => ({
  items: [],
  activeId: null,
  expanded: false,
  open: (items, activeId) =>
    set({ items, activeId: activeId ?? items[0]?.id ?? null }),
  setActive: (id) => set({ activeId: id }),
  toggleExpanded: () => set((s) => ({ expanded: !s.expanded })),
  close: () => set({ items: [], activeId: null, expanded: false }),
}));
