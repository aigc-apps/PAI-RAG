import { describe, it, expect, beforeEach } from "vitest";
import { getUserId } from "../user";

describe("getUserId", () => {
  beforeEach(() => localStorage.clear());

  it("generates and persists a stable id", () => {
    const a = getUserId();
    const b = getUserId();
    expect(a).toBe(b);
    expect(a.length).toBeGreaterThan(0);
    expect(localStorage.getItem("agent-chat:user_id")).toBe(a);
  });
});
