import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("../../api/models", () => ({ listModels: vi.fn() }));
import { ModelSelector } from "../ModelSelector";
import * as api from "../../api/models";

beforeEach(() => vi.clearAllMocks());

describe("ModelSelector", () => {
  it("renders fetched models", async () => {
    (api.listModels as any).mockResolvedValue({ ids: ["fast", "smart"], default: "fast" });
    render(<ModelSelector model="fast" onChange={() => {}} />);
    await waitFor(() => expect(screen.getByRole("option", { name: "smart" })).toBeInTheDocument());
    expect(screen.getByRole("option", { name: "fast" })).toBeInTheDocument();
  });

  it("falls back to the current model when the fetch fails", async () => {
    (api.listModels as any).mockRejectedValue(new Error("boom"));
    render(<ModelSelector model="gpt-4o-mini" onChange={() => {}} />);
    await waitFor(() =>
      expect(screen.getByRole("option", { name: "gpt-4o-mini" })).toBeInTheDocument()
    );
  });

  it("adopts the backend default when the current model is not in the catalog", async () => {
    const onChange = vi.fn();
    (api.listModels as any).mockResolvedValue({ ids: ["fast", "smart"], default: "fast" });
    render(<ModelSelector model="stale/old" onChange={onChange} />);
    await waitFor(() => expect(onChange).toHaveBeenCalledWith("fast"));
  });
});
