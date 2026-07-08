import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("../../api/agentConfig", () => ({
  getAliyunStatus: vi.fn().mockResolvedValue({ configured: true, bound: false }),
}));
vi.mock("../../api/auth", () => ({
  changePassword: vi.fn().mockResolvedValue(undefined),
}));

import { UserMenu } from "../UserMenu";
import { useAuthStore } from "../../store/auth";

beforeEach(() => {
  useAuthStore.setState({
    user: { id: "u1", email: "amy@example.com", role: "user", status: "active", display_name: null },
    logout: vi.fn().mockResolvedValue(undefined),
  });
});

describe("UserMenu", () => {
  it("shows the account initial and opens the menu with all actions", async () => {
    render(<UserMenu />);
    // Avatar shows the first letter of the email.
    const avatar = screen.getByRole("button", { name: /account menu/i });
    expect(avatar.textContent).toBe("A");
    fireEvent.click(avatar);
    expect(await screen.findByText(/aliyun authorization/i)).toBeInTheDocument();
    expect(screen.getByText(/change password/i)).toBeInTheDocument();
    expect(screen.getByText(/sign out/i)).toBeInTheDocument();
  });

  it("calls logout from the menu", async () => {
    const logout = vi.fn().mockResolvedValue(undefined);
    useAuthStore.setState({ logout });
    render(<UserMenu />);
    fireEvent.click(screen.getByRole("button", { name: /account menu/i }));
    fireEvent.click(await screen.findByText(/sign out/i));
    await waitFor(() => expect(logout).toHaveBeenCalledTimes(1));
  });

  it("enables the Aliyun item only when the deployment is configured", async () => {
    render(<UserMenu />);
    fireEvent.click(screen.getByRole("button", { name: /account menu/i }));
    const item = (await screen.findByText(/aliyun authorization/i)).closest("button")!;
    // getAliyunStatus resolves configured:true, so the item is actionable.
    await waitFor(() => expect(item).not.toBeDisabled());
  });
});
