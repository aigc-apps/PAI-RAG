import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Composer } from "../Composer";

describe("Composer", () => {
  it("sends on click and clears the input", async () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} onStop={() => {}} isStreaming={false} />);
    const box = screen.getByRole("textbox");
    await userEvent.type(box, "hi there");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(onSend).toHaveBeenCalledWith("hi there");
    expect((box as HTMLTextAreaElement).value).toBe("");
  });

  it("sends on plain Enter", async () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} onStop={() => {}} isStreaming={false} />);
    const box = screen.getByRole("textbox") as HTMLTextAreaElement;
    await userEvent.type(box, "hello");
    fireEvent.keyDown(box, { key: "Enter" });
    expect(onSend).toHaveBeenCalledWith("hello");
  });

  it("does not send on Enter while an IME is composing", async () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} onStop={() => {}} isStreaming={false} />);
    const box = screen.getByRole("textbox") as HTMLTextAreaElement;
    await userEvent.type(box, "中文");
    // Enter that confirms an IME candidate carries isComposing / keyCode 229.
    fireEvent.keyDown(box, { key: "Enter", isComposing: true, keyCode: 229 });
    expect(onSend).not.toHaveBeenCalled();
    expect(box.value).toBe("中文");
  });

  it("does not send on Shift+Enter", async () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} onStop={() => {}} isStreaming={false} />);
    const box = screen.getByRole("textbox") as HTMLTextAreaElement;
    await userEvent.type(box, "line one");
    fireEvent.keyDown(box, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("shows a stop button while streaming", async () => {
    const onStop = vi.fn();
    render(<Composer onSend={() => {}} onStop={onStop} isStreaming={true} />);
    await userEvent.click(screen.getByRole("button", { name: /stop/i }));
    expect(onStop).toHaveBeenCalled();
  });
});
