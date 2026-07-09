import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MessageControls } from "../MessageControls";

describe("MessageControls", () => {
  beforeEach(() => {
    // copyText() only uses navigator.clipboard in a secure context; jsdom
    // reports isSecureContext=false, so force it on to exercise that path.
    Object.defineProperty(window, "isSecureContext", {
      value: true,
      configurable: true,
    });
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  it("copies text to the clipboard", async () => {
    render(<MessageControls text="hello world" />);
    await userEvent.click(screen.getByRole("button", { name: /复制/ }));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("hello world");
  });

  it("calls onRegenerate when provided", async () => {
    const onRegenerate = vi.fn();
    render(<MessageControls text="x" onRegenerate={onRegenerate} />);
    await userEvent.click(screen.getByRole("button", { name: /重新生成/ }));
    expect(onRegenerate).toHaveBeenCalled();
  });
});
