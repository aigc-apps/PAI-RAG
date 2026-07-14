import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { ChatView } from "./ChatView";
import { SetupWizard } from "./SetupWizard";
import { SettingsView, type SettingsSection } from "./SettingsView";
import { UsersView } from "./UsersView";
import { PreviewPanel } from "./PreviewPanel";
import { AliyunAuthDialog } from "./AliyunAuthDialog";
import { LoginView, CreateAdminView, AcceptInviteView } from "./AuthViews";
import { useAgentConfigStore } from "../store/agentConfig";
import { useAuthStore } from "../store/auth";
import { usePreviewStore } from "../store/preview";
import { useAliyunDialog } from "../store/aliyunDialog";
import { useChatStore } from "../store/chat";
import { useConversationsStore } from "../store/conversations";
import { getConversation } from "../api/conversations";
import { translate, useI18nStore } from "../i18n";
import { cn } from "../lib/cn";
import type { AgentConfigDocument } from "../api/agentConfig";

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

const conversationLoads = new Map<string, ReturnType<typeof getConversation>>();

function loadConversationOnce(id: string) {
  const existing = conversationLoads.get(id);
  if (existing) return existing;
  const request = getConversation(id);
  conversationLoads.set(id, request);
  const cleanup = () => {
    if (conversationLoads.get(id) === request) conversationLoads.delete(id);
  };
  void request.then(cleanup, cleanup);
  return request;
}

function ChatPage({
  isAdmin,
}: {
  isAdmin: boolean;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { conversationId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const rootEntryToSkip = useRef<string | null>(null);
  const requestedConversationId = useRef<string | null>(null);
  const activateByConversationId = useChatStore((s) => s.activateByConversationId);
  const hydrateConversation = useChatStore((s) => s.hydrate);
  const newDraft = useChatStore((s) => s.newDraft);
  const activeConversationId = useChatStore(
    (s) => s.runtimes[s.activeKey]?.conversationId,
  );
  const select = useConversationsStore((s) => s.select);
  const clearSelection = useConversationsStore((s) => s.clearSelection);
  const previewExpanded = usePreviewStore((s) => s.expanded);
  const aliyunDialogOpen = useAliyunDialog((s) => s.open);
  const closeAliyunDialog = useAliyunDialog((s) => s.close);

  useEffect(() => {
    if (conversationId) return;
    requestedConversationId.current = null;
    if (activeConversationId) {
      rootEntryToSkip.current = location.key;
      newDraft();
    }
    clearSelection();
    // A new history entry at `/` always represents a fresh draft. The skip
    // marker prevents the previous runtime id from immediately restoring its URL.
  }, [conversationId, location.key]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!conversationId) return;
    if (requestedConversationId.current === conversationId) return;
    requestedConversationId.current = conversationId;
    let cancelled = false;

    if (activateByConversationId(conversationId)) {
      select(conversationId);
      return;
    }

    void loadConversationOnce(conversationId)
      .then((detail) => {
        if (cancelled) return;
        // A live runtime may have appeared while the request was in flight. It
        // is authoritative because hydrating stale server history could erase
        // partial streaming output.
        if (!activateByConversationId(conversationId)) {
          hydrateConversation(detail);
        }
        select(conversationId);
      })
      .catch(() => {
        if (cancelled) return;
        clearSelection();
        toast.error(translate(useI18nStore.getState().lang, "sidebar.loadFailed"));
        navigate("/", { replace: true });
      });

    return () => {
      cancelled = true;
    };
  }, [activateByConversationId, clearSelection, conversationId, hydrateConversation, navigate, select]);

  useEffect(() => {
    if (conversationId || location.pathname !== "/" || !activeConversationId) return;
    if (rootEntryToSkip.current === location.key) {
      rootEntryToSkip.current = null;
      return;
    }
    navigate(`/chat/${encodeURIComponent(activeConversationId)}`, { replace: true });
  }, [activeConversationId, conversationId, location.key, location.pathname, navigate]);

  return (
    <div className="app-shell flex h-full bg-[var(--bg)]">
      {sidebarOpen && (
        <Sidebar
          onOpenSettings={isAdmin ? () => navigate("/settings/agents") : undefined}
          onOpenUsers={isAdmin ? () => navigate("/users") : undefined}
        />
      )}
      <main className={cn("flex flex-1 flex-col min-w-0", previewExpanded && "hidden")}>
        <ChatView onToggleSidebar={() => setSidebarOpen((open) => !open)} />
      </main>
      <PreviewPanel />
      {aliyunDialogOpen && <AliyunAuthDialog onClose={closeAliyunDialog} />}
    </div>
  );
}

const SETTINGS_SECTIONS = new Set<SettingsSection>([
  "agents",
  "connections",
  "tools",
  "knowledge",
  "skills",
  "org-persona",
  "yaml",
]);

function SettingsPage({ doc }: { doc: AgentConfigDocument }) {
  const { section } = useParams();
  const navigate = useNavigate();
  if (!section || !SETTINGS_SECTIONS.has(section as SettingsSection)) {
    return <Navigate to="/settings/agents" replace />;
  }
  const activeSection = section as SettingsSection;
  return (
    <SettingsView
      doc={doc}
      initialSection={activeSection}
      onSectionChange={(next) => navigate(`/settings/${next}`)}
      onBack={() => navigate("/")}
    />
  );
}

export function App() {
  const [inviteToken, setInviteToken] = useState(readInviteToken);
  const navigate = useNavigate();

  const phase = useAuthStore((s) => s.phase);
  const bootstrapNeeded = useAuthStore((s) => s.bootstrapNeeded);
  const isAdmin = useAuthStore((s) => s.isAdmin);
  const hydrate = useAuthStore((s) => s.hydrate);

  const doc = useAgentConfigStore((s) => s.doc);
  const configLoading = useAgentConfigStore((s) => s.loading);
  const loadSetup = useAgentConfigStore((s) => s.loadSetup);

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
      return <SetupWizard doc={doc} onDone={() => navigate("/", { replace: true })} />;
    }
  }

  const admin = (element: React.ReactNode) =>
    isAdmin ? element : <Navigate to="/" replace />;

  return (
    <Routes>
      <Route path="/" element={<ChatPage isAdmin={isAdmin} />} />
      <Route path="/chat/:conversationId" element={<ChatPage isAdmin={isAdmin} />} />
      <Route
        path="/settings/:section?"
        element={admin(
          doc ? (
            <SettingsPage doc={doc} />
          ) : (
            <Splash />
          ),
        )}
      />
      {/* Knowledge base management now lives inside Settings; keep the old
          top-level links working by redirecting to the Settings tab. */}
      <Route path="/knowledge" element={<Navigate to="/settings/knowledge" replace />} />
      <Route path="/knowledge/:kbId/:tab?" element={<Navigate to="/settings/knowledge" replace />} />
      <Route path="/users" element={admin(<UsersView onBack={() => navigate("/")} />)} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
