import React from "react"
import ReactDOM from "react-dom/client"

import App from "./App"
import { ErrorBoundary } from "./components/ErrorBoundary"
import "./index.css"
import { installGlobalFrontendLogging } from "./lib/frontendLogger"

installGlobalFrontendLogging()

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)
