import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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

  it("shows a stop button while streaming", async () => {
    const onStop = vi.fn();
    render(<Composer onSend={() => {}} onStop={onStop} isStreaming={true} />);
    await userEvent.click(screen.getByRole("button", { name: /stop/i }));
    expect(onStop).toHaveBeenCalled();
  });
});
