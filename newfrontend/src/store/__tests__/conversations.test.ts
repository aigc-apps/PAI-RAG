import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));

import { useConversationsStore } from "../conversations";
import * as api from "../../api/conversations";

beforeEach(() => {
  vi.clearAllMocks();
  useConversationsStore.setState({ items: [], selectedId: undefined });
});

describe("conversations store", () => {
  it("refresh loads the list for the current user", async () => {
    (api.listConversations as any).mockResolvedValue([
      { id: "c1", title: "a", created_at: null, updated_at: null, last_response_id: "r1" },
    ]);
    await useConversationsStore.getState().refresh();
    expect(api.listConversations).toHaveBeenCalledWith("u1");
    expect(useConversationsStore.getState().items).toHaveLength(1);
  });

  it("select / clearSelection update selectedId", () => {
    useConversationsStore.getState().select("c1");
    expect(useConversationsStore.getState().selectedId).toBe("c1");
    useConversationsStore.getState().clearSelection();
    expect(useConversationsStore.getState().selectedId).toBeUndefined();
  });

  it("remove deletes via api and drops from the list", async () => {
    useConversationsStore.setState({
      items: [
        { id: "c1", title: "a", created_at: null, updated_at: null, last_response_id: "r1" },
      ],
      selectedId: "c1",
    });
    (api.deleteConversation as any).mockResolvedValue(undefined);
    await useConversationsStore.getState().remove("c1");
    expect(api.deleteConversation).toHaveBeenCalledWith("c1");
    expect(useConversationsStore.getState().items).toHaveLength(0);
    expect(useConversationsStore.getState().selectedId).toBeUndefined();
  });
});
