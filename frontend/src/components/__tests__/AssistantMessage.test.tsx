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
    expect(screen.getByText("执行记录")).toBeInTheDocument();
    expect(screen.getByText("1 个工具")).toBeInTheDocument();
  });

  it("keeps reasoning and tools in activity before the final answer", () => {
    render(<AssistantMessage message={msg({
      status: "streaming",
      text: "final answer",
      reasoning: "checking sources",
      reasoningStatus: "streaming",
      toolCalls: [{ id: "c1", name: "web_search", arguments: "{}", status: "running" }],
    })} />);
    expect(screen.getByText("工作中")).toBeInTheDocument();
    expect(screen.getByText("checking sources")).toBeVisible();
    expect(screen.getAllByText("(web_search)").length).toBeGreaterThan(0);
    expect(screen.getByText("final answer")).toBeInTheDocument();
  });
});

describe("AssistantMessage aliyun authorization notice (HITL)", () => {
  const hitlMsg = () => msg({
    status: "completed", // activity panel is inactive → collapsed/unmounted
    text: "已暂停,等待授权",
    toolCalls: [{
      id: "c1", name: "shell", arguments: '{"command":"aliyun sts x"}', status: "error",
      output: "{}", notice: { kind: "aliyun_authorization", bound: false, interrupt: true },
    }],
  });

  it("renders an actionable card at the message level when it is the latest turn (paused, awaiting auth)", () => {
    render(<AssistantMessage message={hitlMsg()} isLast />);
    // The card stays visible (message level), not buried in the collapsed panel.
    expect(screen.getByText("需要阿里云授权才能继续")).toBeInTheDocument();
    expect(screen.getByText("去授权")).toBeInTheDocument();
    expect(screen.queryByText("已处理")).not.toBeInTheDocument();
    // The persisted pause explanation is shown as the answer body.
    expect(screen.getByText("已暂停,等待授权")).toBeInTheDocument();
  });

  it("renders a read-only record when a later turn exists (reloaded, already handled)", () => {
    render(<AssistantMessage message={hitlMsg()} isLast={false} />);
    // Still visible so the interaction stays in the transcript…
    expect(screen.getByText("需要阿里云授权才能继续")).toBeInTheDocument();
    expect(screen.getByText("已处理")).toBeInTheDocument();
    // …but the actions are gone (it was already resolved by a following turn).
    expect(screen.queryByText("去授权")).not.toBeInTheDocument();
  });

  it("does not render the card for a notice lacking interrupt", () => {
    render(<AssistantMessage message={msg({
      status: "completed",
      toolCalls: [{
        id: "c1", name: "shell", arguments: "{}", status: "error",
        notice: { kind: "aliyun_authorization", bound: false },
      }],
    })} />);
    expect(screen.queryByText("需要阿里云授权才能继续")).not.toBeInTheDocument();
  });
});

describe("AssistantMessage cancelled", () => {
  beforeEach(() => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  });

  it("shows a cancelled note and the partial text", () => {
    render(<AssistantMessage message={msg({ status: "cancelled", text: "partial answer" })} />);
    expect(screen.getByText("partial answer")).toBeInTheDocument();
    expect(screen.getByText(/已取消/)).toBeInTheDocument();
  });

  it("still offers copy/regenerate for a cancelled (continuable) turn", () => {
    render(<AssistantMessage message={msg({ status: "cancelled", text: "x" })} onRegenerate={() => {}} />);
    expect(screen.getByRole("button", { name: /复制/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /重新生成/ })).toBeInTheDocument();
  });
});
