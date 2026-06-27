import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { AssistantMessage } from "../AssistantMessage";
import type { ChatMessage } from "../../types";

function msg(over: Partial<ChatMessage>): ChatMessage {
  return {
    id: "a",
    role: "assistant",
    text: "",
    reasoning: "",
    reasoningStatus: "idle",
    status: "completed",
    toolCalls: [],
    ...over,
  };
}

describe("AssistantMessage toolCalls", () => {
  it("renders tool cards from toolCalls", () => {
    render(<AssistantMessage message={msg({ status: "completed", text: "ans",
      toolCalls: [{ id: "c1", name: "web_fetch", arguments: "{}", status: "done", output: "PAGE" }] })} />);
    expect(screen.getByText("web_fetch")).toBeInTheDocument();
    expect(screen.getByText("ans")).toBeInTheDocument();
  });
});

describe("AssistantMessage cancelled", () => {
  beforeEach(() => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  });

  it("shows a cancelled note and the partial text", () => {
    render(<AssistantMessage message={msg({ status: "cancelled", text: "partial answer" })} />);
    expect(screen.getByText("partial answer")).toBeInTheDocument();
    expect(screen.getByText(/cancelled/i)).toBeInTheDocument();
  });

  it("still offers copy/regenerate for a cancelled (continuable) turn", () => {
    render(<AssistantMessage message={msg({ status: "cancelled", text: "x" })} onRegenerate={() => {}} />);
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /regenerate/i })).toBeInTheDocument();
  });
});
