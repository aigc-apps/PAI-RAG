import { describe, it, expect } from "vitest";
import { deriveAssistantView } from "../assistantView";
import type { ChatMessage, ToolUse } from "../../types";

function msg(over: Partial<ChatMessage>): ChatMessage {
  return {
    id: "m",
    role: "assistant",
    text: "",
    reasoning: "",
    reasoningStatus: "idle",
    status: "completed",
    toolCalls: [],
    ...over,
  };
}

const tool = (id: string): ToolUse => ({
  id,
  name: "load_skill",
  arguments: "{}",
  status: "done",
  output: "OK",
});

describe("deriveAssistantView", () => {
  it("splits the final text run into the body and routes narration+tools to activity", () => {
    const v = deriveAssistantView(
      msg({
        text: "final answer",
        toolCalls: [tool("c1")],
        steps: [
          { kind: "text", text: "先加载技能。" },
          { kind: "tool", id: "c1" },
          { kind: "text", text: "final answer" },
        ],
      })
    );
    expect(v.bodyText).toBe("final answer");
    expect(v.activitySteps).toEqual([
      { kind: "text", text: "先加载技能。" },
      { kind: "tool", tool: tool("c1") },
    ]);
  });

  it("keeps the body empty while the turn ends on a running tool", () => {
    const v = deriveAssistantView(
      msg({
        text: "",
        toolCalls: [{ ...tool("c1"), status: "running", output: undefined }],
        steps: [
          { kind: "text", text: "working…" },
          { kind: "tool", id: "c1" },
        ],
      })
    );
    expect(v.bodyText).toBe("");
    expect(v.activitySteps).toEqual([
      { kind: "text", text: "working…" },
      { kind: "tool", tool: { ...tool("c1"), status: "running", output: undefined } },
    ]);
  });

  it("drops whitespace-only narration runs so they don't render as empty blocks", () => {
    const v = deriveAssistantView(
      msg({
        text: "answer",
        toolCalls: [tool("c1")],
        steps: [
          { kind: "text", text: "\n\n" },
          { kind: "tool", id: "c1" },
          { kind: "text", text: "answer" },
        ],
      })
    );
    expect(v.bodyText).toBe("answer");
    expect(v.activitySteps).toEqual([{ kind: "tool", tool: tool("c1") }]);
  });

  it("falls back to legacy layout for reloaded history without steps", () => {
    const v = deriveAssistantView(
      msg({ text: "narration + answer merged", toolCalls: [tool("c1")] })
    );
    // Whole blob is the body; every tool sits in the activity panel.
    expect(v.bodyText).toBe("narration + answer merged");
    expect(v.activitySteps).toEqual([{ kind: "tool", tool: tool("c1") }]);
  });

  it("a plain answer with no tools has no activity", () => {
    const v = deriveAssistantView(
      msg({ text: "hi", steps: [{ kind: "text", text: "hi" }] })
    );
    expect(v.bodyText).toBe("hi");
    expect(v.activitySteps).toEqual([]);
  });
});
