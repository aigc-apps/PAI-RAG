import { describe, it, expect } from "vitest";
import { cn } from "../cn";

describe("cn", () => {
  it("joins truthy class names and drops falsy", () => {
    expect(cn("a", false, "b", null, undefined, "c")).toBe("a b c");
  });
});
