import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

function Placeholder() {
  return <div>Agent Chat</div>;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Placeholder />
  </StrictMode>
);
