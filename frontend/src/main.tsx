import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Toaster } from "sonner";
import { BrowserRouter } from "react-router-dom";
import { App } from "./components/App";
import "./index.css";

document.documentElement.removeAttribute("data-theme");
try {
  localStorage.removeItem("agent-chat:theme");
} catch {
  // Ignore storage access failures; the app defaults to the light workbench.
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
      <Toaster position="top-center" />
    </BrowserRouter>
  </StrictMode>
);
