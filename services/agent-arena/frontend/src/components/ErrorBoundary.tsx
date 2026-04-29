import React from "react"
import { AlertTriangle, RefreshCcw } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { reportFrontendLog } from "@/lib/frontendLogger"

type ErrorBoundaryState = {
  error: Error | null
}

export class ErrorBoundary extends React.Component<
  React.PropsWithChildren,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    reportFrontendLog({
      level: "error",
      source: "react.error_boundary",
      message: error.message,
      stack: error.stack,
      component_stack: info.componentStack || undefined,
    })
  }

  render() {
    if (!this.state.error) return this.props.children

    return (
      <main className="min-h-screen bg-muted/30 p-6">
        <div className="mx-auto max-w-3xl">
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>前端渲染失败</AlertTitle>
            <AlertDescription>
              <p className="mb-3">
                错误已经发送到后端日志：<code>logs/frontend.log</code>
              </p>
              <pre className="max-h-[320px] overflow-auto whitespace-pre-wrap rounded-md bg-background p-3 text-xs">
                {this.state.error.stack || this.state.error.message}
              </pre>
              <Button className="mt-4" variant="outline" onClick={() => window.location.reload()}>
                <RefreshCcw className="size-4" />
                刷新页面
              </Button>
            </AlertDescription>
          </Alert>
        </div>
      </main>
    )
  }
}
