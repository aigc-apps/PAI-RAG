import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CollapsibleReasoning } from "../CollapsibleReasoning";

describe("CollapsibleReasoning", () => {
  it("renders nothing when reasoning is empty", () => {
    const { container } = render(
      <CollapsibleReasoning reasoning="" status="idle" />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows reasoning text while streaming (expanded)", () => {
    render(<CollapsibleReasoning reasoning="thinking..." status="streaming" />);
    expect(screen.getByText("thinking...")).toBeVisible();
  });

  it("renders a toggle once done", () => {
    render(<CollapsibleReasoning reasoning="done thinking" status="done" />);
    expect(screen.getByRole("button")).toBeInTheDocument();
  });
});
