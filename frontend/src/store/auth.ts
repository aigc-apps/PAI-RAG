import { create } from "zustand";
import {
  acceptInvite as apiAcceptInvite,
  bootstrapStatus,
  createAdmin as apiCreateAdmin,
  login as apiLogin,
  logout as apiLogout,
  me as apiMe,
  type AuthUser,
} from "../api/auth";
import { setUnauthorizedHandler } from "../lib/apiFetch";

// "loading" until the initial /me hydration resolves; then either the user is
// known ("authenticated") or not ("anonymous"). bootstrapNeeded is a separate
// axis: when true, no account exists yet and the app shows Create-Admin.
type AuthPhase = "loading" | "authenticated" | "anonymous";

interface AuthState {
  user: AuthUser | null;
  phase: AuthPhase;
  bootstrapNeeded: boolean;
  isAdmin: boolean;

  hydrate: () => Promise<void>;
  createAdmin: (email: string, password: string, token?: string) => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  acceptInvite: (token: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

function admin(u: AuthUser | null): boolean {
  return u?.role === "admin";
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  phase: "loading",
  bootstrapNeeded: false,
  isAdmin: false,

  hydrate: async () => {
    // A live session (cookie) short-circuits the bootstrap probe.
    const user = await apiMe().catch(() => null);
    if (user) {
      set({ user, isAdmin: admin(user), phase: "authenticated", bootstrapNeeded: false });
      return;
    }
    let needed = false;
    try {
      needed = (await bootstrapStatus()).needed;
    } catch {
      needed = false;
    }
    set({ user: null, isAdmin: false, phase: "anonymous", bootstrapNeeded: needed });
  },

  createAdmin: async (email, password, token) => {
    const user = await apiCreateAdmin(email, password, token);
    set({ user, isAdmin: admin(user), phase: "authenticated", bootstrapNeeded: false });
  },

  login: async (email, password) => {
    const user = await apiLogin(email, password);
    set({ user, isAdmin: admin(user), phase: "authenticated", bootstrapNeeded: false });
  },

  acceptInvite: async (token, password) => {
    const user = await apiAcceptInvite(token, password);
    set({ user, isAdmin: admin(user), phase: "authenticated", bootstrapNeeded: false });
  },

  logout: async () => {
    await apiLogout().catch(() => undefined);
    set({ user: null, isAdmin: false, phase: "anonymous", bootstrapNeeded: false });
  },
}));

// Any 401 from an authenticated call drops the app to logged-out so the guards
// re-render the login screen. Guard against clobbering the initial "loading".
setUnauthorizedHandler(() => {
  const s = useAuthStore.getState();
  if (s.phase === "authenticated") {
    useAuthStore.setState({ user: null, isAdmin: false, phase: "anonymous" });
  }
});
