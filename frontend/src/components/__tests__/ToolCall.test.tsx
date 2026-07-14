import { beforeEach, describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useI18nStore } from "../../i18n";
import { ToolCall } from "../ToolCall";

describe("ToolCall", () => {
  beforeEach(() => {
    useI18nStore.getState().setLang("zh");
  });

  it("shows name + result and expands details", async () => {
    render(<ToolCall tool={{ id: "c1", name: "web_fetch", arguments: '{"url":"x"}', status: "done", output: "PAGE TEXT" }} />);
    expect(screen.getByText("(web_fetch)")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByText(/PAGE TEXT/)).toBeInTheDocument();
    expect(screen.getByText(/"url"/)).toBeInTheDocument();
  });
  it("shows an error status", () => {
    render(<ToolCall tool={{ id: "c1", name: "web_fetch", arguments: "{}", status: "error", error: "boom" }} />);
    expect(screen.getByText("(web_fetch)")).toBeInTheDocument();
    expect(screen.getByText("工具执行失败")).toBeInTheDocument();
  });

  it("shows localized name, raw name, and hoverable summary", () => {
    render(
      <ToolCall
        tool={{
          id: "c1",
          name: "shell",
          arguments: '{"command":"ls  /opt/code\\n--color=auto"}',
          status: "done",
          output: "",
        }}
      />,
    );

    expect(screen.getByText("执行命令")).toBeInTheDocument();
    expect(screen.getByText("(shell)")).toBeInTheDocument();
    const summary = screen.getByTitle("ls /opt/code --color=auto");
    expect(summary).toHaveTextContent("ls /opt/code --color=auto");
    expect(summary).toHaveClass("truncate");
  });

  it("uses the current language for the localized tool name", () => {
    useI18nStore.getState().setLang("en");
    render(
      <ToolCall
        tool={{
          id: "c1",
          name: "web_search",
          arguments: '{"query":"PAI-RAG"}',
          status: "running",
        }}
      />,
    );

    expect(screen.getByText("Search the web")).toBeInTheDocument();
    expect(screen.getByText("(web_search)")).toBeInTheDocument();
  });

  it("shows a known no-argument tool without a summary separator", () => {
    render(
      <ToolCall
        tool={{
          id: "c1",
          name: "current_datetime",
          arguments: "{}",
          status: "done",
          output: "now",
        }}
      />,
    );

    expect(screen.getByText("获取当前时间")).toBeInTheDocument();
    expect(screen.getByText("(current_datetime)")).toBeInTheDocument();
    expect(screen.queryByText("·")).not.toBeInTheDocument();
  });

  it("keeps unknown tools as a raw name only", () => {
    render(
      <ToolCall
        tool={{
          id: "c1",
          name: "custom_tool",
          arguments: '{"query":"x"}',
          status: "done",
          output: "ok",
        }}
      />,
    );

    expect(screen.getByText("custom_tool")).toBeInTheDocument();
    expect(screen.queryByText("(custom_tool)")).not.toBeInTheDocument();
    expect(screen.queryByTitle("x")).not.toBeInTheDocument();
  });
});
