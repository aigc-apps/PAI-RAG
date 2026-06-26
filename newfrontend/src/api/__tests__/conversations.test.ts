import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  listConversations,
  getConversation,
  deleteConversation,
} from "../conversations";

function mockFetch(json: unknown, ok = true, status = 200) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => json,
  } as Response);
}

describe("conversations api", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("lists conversations for a user", async () => {
    const fetchMock = mockFetch({
      data: [
        {
          id: "c1",
          title: "hi",
          created_at: "t",
          updated_at: "t",
          last_response_id: "r1",
        },
      ],
    });
    vi.stubGlobal("fetch", fetchMock);
    const out = await listConversations("u1");
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("c1");
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/v1/conversations");
    expect(url).toContain("user_id=u1");
  });

  it("gets a conversation detail", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch({
        id: "c1",
        title: "hi",
        created_at: "t",
        updated_at: "t",
        latest_response_id: "r2",
        messages: [],
      })
    );
    const out = await getConversation("c1");
    expect(out.latest_response_id).toBe("r2");
  });

  it("throws on a 404 detail", async () => {
    vi.stubGlobal("fetch", mockFetch({}, false, 404));
    await expect(getConversation("nope")).rejects.toThrow();
  });

  it("deletes a conversation", async () => {
    const fetchMock = mockFetch({ deleted: true });
    vi.stubGlobal("fetch", fetchMock);
    await deleteConversation("c1");
    expect(String(fetchMock.mock.calls[0][0])).toContain("/v1/conversations/c1");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "DELETE" });
  });
});
