import { create } from "zustand";
import type { FileArtifact } from "../types";

interface PreviewState {
  /** The artifact shown in the right-side preview panel, or null when closed. */
  active: FileArtifact | null;
  open: (artifact: FileArtifact) => void;
  close: () => void;
}

export const usePreviewStore = create<PreviewState>((set) => ({
  active: null,
  open: (artifact) => set({ active: artifact }),
  close: () => set({ active: null }),
}));
