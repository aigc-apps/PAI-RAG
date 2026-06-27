import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ToolCall } from "../ToolCall";

describe("ToolCall", () => {
  it("shows name + result and expands details", async () => {
    render(<ToolCall tool={{ id: "c1", name: "web_fetch", arguments: '{"url":"x"}', status: "done", output: "PAGE TEXT" }} />);
    expect(screen.getByText("web_fetch")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByText(/PAGE TEXT/)).toBeInTheDocument();
    expect(screen.getByText(/"url"/)).toBeInTheDocument();
  });
  it("shows an error status", () => {
    render(<ToolCall tool={{ id: "c1", name: "web_fetch", arguments: "{}", status: "error", error: "boom" }} />);
    expect(screen.getByText("web_fetch")).toBeInTheDocument();
    expect(screen.getByText(/error/i)).toBeInTheDocument();
  });
});
