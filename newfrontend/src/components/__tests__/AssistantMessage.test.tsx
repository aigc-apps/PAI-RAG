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
    render(<AssistantMessage message={msg({ status: "completed", text: "answer",
      toolCalls: [{ id: "c1", name: "web_fetch", arguments: "{}", status: "done", output: "PAGE" }] })} />);
    expect(screen.getByText("answer")).toBeInTheDocument();
    expect(screen.getByText("Activity")).toBeInTheDocument();
    expect(screen.getByText("1 tool")).toBeInTheDocument();
  });

  it("keeps reasoning and tools in activity before the final answer", () => {
    render(<AssistantMessage message={msg({
      status: "streaming",
      text: "final answer",
      reasoning: "checking sources",
      reasoningStatus: "streaming",
      toolCalls: [{ id: "c1", name: "web_search", arguments: "{}", status: "running" }],
    })} />);
    expect(screen.getByText("Working")).toBeInTheDocument();
    expect(screen.getByText("checking sources")).toBeVisible();
    expect(screen.getAllByText("web_search").length).toBeGreaterThan(0);
    expect(screen.getByText("final answer")).toBeInTheDocument();
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
