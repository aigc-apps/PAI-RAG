import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { ChatView } from "./ChatView";
import { SetupWizard } from "./SetupWizard";
import { SettingsView } from "./SettingsView";
import { PreviewPanel } from "./PreviewPanel";
import { AliyunAuthDialog } from "./AliyunAuthDialog";
import { LoginView, CreateAdminView, AcceptInviteView } from "./AuthViews";
import { useAgentConfigStore } from "../store/agentConfig";
import { useAuthStore } from "../store/auth";
import { usePreviewStore } from "../store/preview";
import { useAliyunDialog } from "../store/aliyunDialog";
import { cn } from "../lib/cn";

function Splash() {
  return (
    <div className="grid h-full place-items-center bg-[var(--bg)] text-[var(--text-muted)]">
      <Loader2 className="h-5 w-5 animate-spin" />
    </div>
  );
}

/** Read (and remember) an `?invite=<token>` deep-link. The value is captured
 * once at module init so clearing the URL after activation doesn't lose it. */
function readInviteToken(): string {
  if (typeof window === "undefined") return "";
  return new URLSearchParams(window.location.search).get("invite") ?? "";
}

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [view, setView] = useState<"chat" | "settings">("chat");
  const [inviteToken, setInviteToken] = useState(readInviteToken);

  const phase = useAuthStore((s) => s.phase);
  const bootstrapNeeded = useAuthStore((s) => s.bootstrapNeeded);
  const isAdmin = useAuthStore((s) => s.isAdmin);
  const hydrate = useAuthStore((s) => s.hydrate);

  const doc = useAgentConfigStore((s) => s.doc);
  const configLoading = useAgentConfigStore((s) => s.loading);
  const loadSetup = useAgentConfigStore((s) => s.loadSetup);
  const previewExpanded = usePreviewStore((s) => s.expanded);
  const aliyunDialogOpen = useAliyunDialog((s) => s.open);
  const closeAliyunDialog = useAliyunDialog((s) => s.close);

  // Hydrate auth on mount (cookie → /me, else bootstrap probe).
  useEffect(() => {
    void hydrate();
  }, [hydrate]);

  // The agent-config document is an admin-only resource (GET /v1/setup is gated),
  // so only fetch it once we know the session is an authenticated admin.
  useEffect(() => {
    if (phase === "authenticated" && isAdmin) {
      void loadSetup();
    }
  }, [phase, isAdmin, loadSetup]);

  const clearInvite = () => {
    setInviteToken("");
    if (typeof window !== "undefined") {
      window.history.replaceState({}, "", window.location.pathname);
    }
  };

  // --- unauthenticated guards, in priority order ---
  if (phase === "loading") return <Splash />;
  if (bootstrapNeeded) return <CreateAdminView />;
  if (inviteToken) return <AcceptInviteView token={inviteToken} onDone={clearInvite} />;
  if (phase !== "authenticated") return <LoginView />;

  // --- authenticated ---
  // Admins go through first-time setup; regular users chat directly and never
  // touch the control plane.
  if (isAdmin) {
    if (configLoading && !doc) return <Splash />;
    if (doc && !doc.setup.completed) {
      return <SetupWizard doc={doc} onDone={() => setView("chat")} />;
    }
    if (doc && view === "settings") {
      return <SettingsView doc={doc} onBack={() => setView("chat")} />;
    }
  }

  return (
    <div className="flex h-full bg-[var(--bg)]">
      {sidebarOpen && (
        <Sidebar onOpenSettings={isAdmin ? () => setView("settings") : undefined} />
      )}
      <main className={cn("flex flex-1 flex-col min-w-0", previewExpanded && "hidden")}>
        <ChatView onToggleSidebar={() => setSidebarOpen((o) => !o)} />
      </main>
      <PreviewPanel />
      {aliyunDialogOpen && <AliyunAuthDialog onClose={closeAliyunDialog} />}
    </div>
  );
}
