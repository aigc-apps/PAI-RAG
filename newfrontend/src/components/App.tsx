import { Sidebar } from "./Sidebar";
import { ChatView } from "./ChatView";

export function App() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <ChatView />
    </div>
  );
}
