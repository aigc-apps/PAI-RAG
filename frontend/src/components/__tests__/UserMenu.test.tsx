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
  it("shows the account row and opens the menu with all actions", async () => {
    render(<UserMenu />);
    // The whole row is the trigger: avatar initial + email are both inside it.
    const trigger = screen.getByRole("button", { name: /account menu/i });
    expect(trigger).toHaveTextContent("A");
    expect(trigger).toHaveTextContent("amy@example.com");
    fireEvent.click(trigger);
    expect(await screen.findByText(/aliyun authorization/i)).toBeInTheDocument();
    expect(screen.getByText(/change password/i)).toBeInTheDocument();
    expect(screen.getByText(/sign out/i)).toBeInTheDocument();
  });

  it("shows Settings only when onOpenSettings is provided, and invokes it", async () => {
    const { unmount } = render(<UserMenu />);
    fireEvent.click(screen.getByRole("button", { name: /account menu/i }));
    await screen.findByText(/sign out/i);
    expect(screen.queryByText(/^settings$/i)).not.toBeInTheDocument();
    unmount();

    const onOpenSettings = vi.fn();
    render(<UserMenu onOpenSettings={onOpenSettings} />);
    fireEvent.click(screen.getByRole("button", { name: /account menu/i }));
    fireEvent.click(await screen.findByText(/^settings$/i));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  it("shows Users only when onOpenUsers is provided, and invokes it", async () => {
    const { unmount } = render(<UserMenu />);
    fireEvent.click(screen.getByRole("button", { name: /account menu/i }));
    await screen.findByText(/sign out/i);
    expect(screen.queryByText(/^users$/i)).not.toBeInTheDocument();
    unmount();

    const onOpenUsers = vi.fn();
    render(<UserMenu onOpenUsers={onOpenUsers} />);
    fireEvent.click(screen.getByRole("button", { name: /account menu/i }));
    fireEvent.click(await screen.findByText(/^users$/i));
    expect(onOpenUsers).toHaveBeenCalledTimes(1);
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
