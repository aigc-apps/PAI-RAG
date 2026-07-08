import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LoginView, CreateAdminView } from "../AuthViews";
import { useAuthStore } from "../../store/auth";

beforeEach(() => {
  useAuthStore.setState({
    login: vi.fn().mockResolvedValue(undefined),
    createAdmin: vi.fn().mockResolvedValue(undefined),
  });
});

describe("LoginView", () => {
  it("submits email + password to the store", async () => {
    const login = vi.fn().mockResolvedValue(undefined);
    useAuthStore.setState({ login });
    render(<LoginView />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));
    await waitFor(() => expect(login).toHaveBeenCalledWith("a@b.com", "password123"));
  });

  it("surfaces a login error", async () => {
    useAuthStore.setState({ login: vi.fn().mockRejectedValue(new Error("invalid email or password")) });
    render(<LoginView />);
    fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: /sign in/i }));
    expect(await screen.findByText(/invalid email or password/i)).toBeInTheDocument();
  });
});

describe("CreateAdminView", () => {
  it("rejects mismatched passwords before calling the store", async () => {
    const createAdmin = vi.fn().mockResolvedValue(undefined);
    useAuthStore.setState({ createAdmin });
    render(<CreateAdminView />);
    fireEvent.change(screen.getByLabelText(/admin email/i), { target: { value: "admin@b.com" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "password123" } });
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: "different1" } });
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));
    expect(await screen.findByText(/do not match/i)).toBeInTheDocument();
    expect(createAdmin).not.toHaveBeenCalled();
  });

  it("creates the admin when the form is valid", async () => {
    const createAdmin = vi.fn().mockResolvedValue(undefined);
    useAuthStore.setState({ createAdmin });
    render(<CreateAdminView />);
    fireEvent.change(screen.getByLabelText(/admin email/i), { target: { value: "admin@b.com" } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: "password123" } });
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));
    await waitFor(() =>
      expect(createAdmin).toHaveBeenCalledWith("admin@b.com", "password123", undefined)
    );
  });
});
