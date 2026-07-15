import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useI18nStore } from "../../i18n";
import { AgentActivity } from "../AgentActivity";

beforeEach(() => useI18nStore.getState().setLang("en"));

describe("AgentActivity", () => {
  it("renders timeline reasoning around a tool without duplicating aggregate reasoning", () => {
    const { container } = render(
      <AgentActivity
        reasoning="first thoughtsecond thought"
        reasoningStatus="streaming"
        messageStatus="streaming"
        steps={[
          { kind: "reasoning", text: "first thought" },
          {
            kind: "tool",
            tool: {
              id: "c1",
              name: "custom_tool",
              arguments: "{}",
              status: "running",
            },
          },
          { kind: "reasoning", text: "second thought" },
        ]}
      />
    );

    const content = container.textContent ?? "";
    expect(content.indexOf("first thought")).toBeLessThan(content.indexOf("custom_tool"));
    expect(content.indexOf("custom_tool")).toBeLessThan(content.indexOf("second thought"));
    expect(screen.getAllByText("first thought")).toHaveLength(1);
    expect(screen.queryByText("first thoughtsecond thought")).not.toBeInTheDocument();
  });

  it("still renders aggregate reasoning for legacy messages without timeline reasoning", () => {
    render(
      <AgentActivity
        reasoning="legacy thought"
        reasoningStatus="streaming"
        messageStatus="streaming"
        steps={[]}
      />
    );

    expect(screen.getByText("legacy thought")).toBeInTheDocument();
  });
});
