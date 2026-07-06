import { describe, it, expect, beforeEach } from "vitest";
import { usePreviewStore } from "../preview";
import type { FileArtifact } from "../../types";

const A: FileArtifact = { id: "a", name: "a.md", mime: "text/markdown", size: 1, kind: "markdown" };
const B: FileArtifact = { id: "b", name: "b.png", mime: "image/png", size: 2, kind: "image" };

beforeEach(() => usePreviewStore.getState().close());

describe("preview store", () => {
  it("open sets the item set and defaults active to the first", () => {
    usePreviewStore.getState().open([A, B]);
    const s = usePreviewStore.getState();
    expect(s.items).toHaveLength(2);
    expect(s.activeId).toBe("a");
  });

  it("open honors an explicit activeId", () => {
    usePreviewStore.getState().open([A, B], "b");
    expect(usePreviewStore.getState().activeId).toBe("b");
  });

  it("setActive switches within the set; close clears it", () => {
    usePreviewStore.getState().open([A, B]);
    usePreviewStore.getState().setActive("b");
    expect(usePreviewStore.getState().activeId).toBe("b");
    usePreviewStore.getState().close();
    expect(usePreviewStore.getState().items).toEqual([]);
    expect(usePreviewStore.getState().activeId).toBeNull();
  });

  it("open on an empty set leaves activeId null", () => {
    usePreviewStore.getState().open([]);
    expect(usePreviewStore.getState().activeId).toBeNull();
  });

  it("expanded defaults false, toggles, and resets on close", () => {
    usePreviewStore.getState().open([A, B]);
    expect(usePreviewStore.getState().expanded).toBe(false);
    usePreviewStore.getState().toggleExpanded();
    expect(usePreviewStore.getState().expanded).toBe(true);
    usePreviewStore.getState().close();
    expect(usePreviewStore.getState().expanded).toBe(false);
  });
});
