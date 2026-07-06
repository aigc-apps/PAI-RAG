import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { ChatView } from "./ChatView";
import { SetupWizard } from "./SetupWizard";
import { SettingsView } from "./SettingsView";
import { PreviewPanel } from "./PreviewPanel";
import { useAgentConfigStore } from "../store/agentConfig";
import { usePreviewStore } from "../store/preview";
import { cn } from "../lib/cn";

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [view, setView] = useState<"chat" | "settings">("chat");
  const doc = useAgentConfigStore((s) => s.doc);
  const loading = useAgentConfigStore((s) => s.loading);
  const loadSetup = useAgentConfigStore((s) => s.loadSetup);
  const previewExpanded = usePreviewStore((s) => s.expanded);

  useEffect(() => {
    void loadSetup();
  }, [loadSetup]);

  if (loading && !doc) {
    return (
      <div className="grid h-full place-items-center bg-[var(--bg)] text-[var(--text-muted)]">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }

  if (doc && !doc.setup.completed) {
    return <SetupWizard doc={doc} onDone={() => setView("chat")} />;
  }

  if (doc && view === "settings") {
    return <SettingsView doc={doc} onBack={() => setView("chat")} />;
  }

  return (
    <div className="flex h-full bg-[var(--bg)]">
      {sidebarOpen && <Sidebar onOpenSettings={() => setView("settings")} />}
      <main className={cn("flex flex-1 flex-col min-w-0", previewExpanded && "hidden")}>
        <ChatView onToggleSidebar={() => setSidebarOpen((o) => !o)} />
      </main>
      <PreviewPanel />
    </div>
  );
}
