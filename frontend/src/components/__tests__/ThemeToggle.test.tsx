import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeToggle } from "../ThemeToggle";

describe("ThemeToggle", () => {
  beforeEach(() => { localStorage.clear(); document.documentElement.removeAttribute("data-theme"); });
  it("toggles the document theme on click", async () => {
    render(<ThemeToggle />);
    const btn = screen.getByRole("button", { name: /toggle theme/i });
    const before = document.documentElement.getAttribute("data-theme");
    await userEvent.click(btn);
    expect(document.documentElement.getAttribute("data-theme")).not.toBe(before);
  });
});
