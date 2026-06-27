import { useState } from "react";
import { Sidebar } from "./Sidebar";
import { ChatView } from "./ChatView";

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-full">
      {sidebarOpen && <Sidebar />}
      <main className="flex flex-1 flex-col min-w-0">
        <ChatView onToggleSidebar={() => setSidebarOpen((o) => !o)} />
      </main>
    </div>
  );
}
