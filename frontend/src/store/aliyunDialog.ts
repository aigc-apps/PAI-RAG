import { create } from "zustand";

/**
 * Single-instance control for the Aliyun authorization dialog. Several entry
 * points open the same dialog — the avatar menu, Settings → Tools, and the
 * inline "去授权" tool card in chat — so it's mounted once in {@link App} and
 * toggled through this store rather than a `useState` per call site.
 */
interface AliyunDialogState {
  open: boolean;
  /**
   * True when opened from a paused agent turn (the inline "去授权" card): the
   * dialog then offers a "继续" button after a successful authorization to
   * resume the agent. Opening from the avatar menu / Settings leaves it false.
   */
  resumeAfter: boolean;
  show: (opts?: { resumeAfter?: boolean }) => void;
  close: () => void;
}

export const useAliyunDialog = create<AliyunDialogState>((set) => ({
  open: false,
  resumeAfter: false,
  show: (opts) => set({ open: true, resumeAfter: opts?.resumeAfter ?? false }),
  close: () => set({ open: false, resumeAfter: false }),
}));
