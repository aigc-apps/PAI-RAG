import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AliyunAuthToolCard } from "../AliyunAuthToolCard";
import type { ToolUse } from "../../types";
import { useAliyunDialog } from "../../store/aliyunDialog";
import { useComposer } from "../../store/composer";
import { verifyAliyun } from "../../api/agentConfig";

vi.mock("../../api/agentConfig", () => ({ verifyAliyun: vi.fn() }));

function tool(over: Partial<ToolUse>): ToolUse {
  return { id: "c1", name: "shell", arguments: "{}", status: "done", ...over };
}

describe("AliyunAuthToolCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAliyunDialog.setState({ open: false, resumeAfter: false });
    useComposer.setState({ submit: null });
  });

  it("offers 去授权 when unbound and opens the dialog in resume mode", () => {
    render(
      <AliyunAuthToolCard
        tool={tool({ notice: { kind: "aliyun_authorization", bound: false } })}
      />
    );
    expect(screen.getByText("需要阿里云授权才能继续")).toBeInTheDocument();
    // Unbound: no re-verify button, only 去授权.
    expect(screen.queryByText("重新校验")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("去授权"));
    expect(useAliyunDialog.getState().open).toBe(true);
    // Opened from a paused turn → dialog offers the "继续" affordance on success.
    expect(useAliyunDialog.getState().resumeAfter).toBe(true);
  });

  it("shows a 继续 button after a successful re-verify that resumes the agent", async () => {
    (verifyAliyun as any).mockResolvedValue({ ok: true, external_id: "x", verdict: { ok: true } });
    const submit = vi.fn();
    useComposer.setState({ submit });
    render(
      <AliyunAuthToolCard
        tool={tool({ notice: { kind: "aliyun_authorization", bound: true } })}
      />
    );
    fireEvent.click(screen.getByText("重新校验"));
    const cont = await screen.findByText("继续");
    fireEvent.click(cont);
    expect(submit).toHaveBeenCalledOnce();
    expect(submit.mock.calls[0][0]).toMatch(/继续/);
  });

  it("offers 重新校验 when bound and shows a success result", async () => {
    (verifyAliyun as any).mockResolvedValue({ ok: true, external_id: "x", verdict: { ok: true } });
    render(
      <AliyunAuthToolCard
        tool={tool({
          notice: { kind: "aliyun_authorization", bound: true, error_code: "InvalidSecurityToken.Expired" },
        })}
      />
    );
    expect(screen.getByText("阿里云凭证可能已失效")).toBeInTheDocument();
    expect(screen.getByText("(InvalidSecurityToken.Expired)")).toBeInTheDocument();
    fireEvent.click(screen.getByText("重新校验"));
    await waitFor(() => expect(verifyAliyun).toHaveBeenCalledOnce());
    expect(await screen.findByText(/凭证有效/)).toBeInTheDocument();
  });

  it("surfaces a failed re-verify", async () => {
    (verifyAliyun as any).mockResolvedValue({
      ok: false,
      external_id: "x",
      verdict: { ok: false, error_message: "token expired" },
    });
    render(
      <AliyunAuthToolCard
        tool={tool({ notice: { kind: "aliyun_authorization", bound: true } })}
      />
    );
    fireEvent.click(screen.getByText("重新校验"));
    expect(await screen.findByText("token expired")).toBeInTheDocument();
  });

  it("renders nothing without an aliyun notice", () => {
    const { container } = render(<AliyunAuthToolCard tool={tool({})} />);
    expect(container).toBeEmptyDOMElement();
  });
});
