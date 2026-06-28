import { describe, it, expect, beforeEach, vi } from "vitest";
import { getTheme, setTheme, toggleTheme } from "../theme";

describe("theme", () => {
  beforeEach(() => { localStorage.clear(); document.documentElement.removeAttribute("data-theme"); });
  it("setTheme persists and applies data-theme", () => {
    setTheme("dark");
    expect(localStorage.getItem("agent-chat:theme")).toBe("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(getTheme()).toBe("dark");
  });
  it("toggleTheme flips light<->dark", () => {
    setTheme("light");
    expect(toggleTheme()).toBe("dark");
    expect(getTheme()).toBe("dark");
    expect(toggleTheme()).toBe("light");
  });
  it("defaults from system when unset", () => {
    vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("dark"), media: q, addEventListener() {}, removeEventListener() {} }));
    expect(getTheme()).toBe("dark");
  });
});
