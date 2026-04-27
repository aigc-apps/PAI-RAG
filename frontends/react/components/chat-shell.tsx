"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, LoaderCircle, Plus, Square, Trash2, User2 } from "lucide-react";
import { cancelSession, createSession, deleteSession, getSession, listSessions, streamChat } from "@/lib/api";
import type { AskUserPayload, ChatMessage, SessionSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";

const ASK_BLOCK_RE = /\[\[ASK_USER\]\]([\s\S]*?)\[\[\/ASK_USER\]\]/g;
const TURN_RE = /\n?── Turn \d+ ──\n/g;
const TOOL_RE = /\n🛠️\s+(\w+)/;
const SUMMARY_RE = /<summary>\s*([\s\S]*?)\s*<\/summary>/;
const THINKING_RE = /<thinking>[\s\S]*?<\/thinking>/g;
const INFO_END_RE = /\n?\[Info\]\s*模型未调用工具，任务结束。\s*/g;
const INTERNAL_RE = /\n?\[\[INTERNAL_TURN_START\]\]\n?/g;

type RenderSegment =
  | { type: "markdown"; content: string }
  | { type: "turn"; title: string; content: string; internal?: boolean }
  | { type: "ask"; payload: AskUserPayload };

function Markdown({ content }: { content: string }) {
  return (
    <div className="prose-agent">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function extractTurnTitle(content: string, turnNum: number) {
  const cleaned = content.replace(/```[\s\S]*?```/g, "").replace(THINKING_RE, "");
  const summary = SUMMARY_RE.exec(cleaned);
  if (summary?.[1]?.trim()) {
    const title = summary[1].trim().split("\n")[0];
    return title.length > 60 ? `${title.slice(0, 57)}...` : title;
  }
  const tool = TOOL_RE.exec(content);
  return tool ? `Turn ${turnNum}: ${tool[1]}` : `Turn ${turnNum}`;
}

function parsePlainTurns(text: string): RenderSegment[] {
  const rawParts = text.split(TURN_RE);
  const internalFlags = new Array(rawParts.length).fill(false);
  let seenInternal = false;
  for (let i = 1; i < rawParts.length; i += 1) {
    if (INTERNAL_RE.test(rawParts[i - 1] || "")) {
      seenInternal = true;
    }
    INTERNAL_RE.lastIndex = 0;
    internalFlags[i] = seenInternal;
  }
  const parts = rawParts.map((part) => part.replace(INTERNAL_RE, ""));

  if (parts.length <= 2) {
    const body = (parts.at(-1) ?? "").replace(INFO_END_RE, "").trim();
    if (!body) {
      return [];
    }
    if (parts.length === 2 && internalFlags[1]) {
      return [{ type: "turn", title: "Internal memory review", content: body, internal: true }];
    }
    if (TOOL_RE.test(body)) {
      return [{ type: "turn", title: extractTurnTitle(body, 1), content: body }];
    }
    const clean = body.replace(SUMMARY_RE, "").replace(THINKING_RE, "").trim();
    return clean ? [{ type: "markdown", content: clean }] : [];
  }

  const segments: RenderSegment[] = [];
  if (parts[0].trim()) {
    segments.push({ type: "markdown", content: parts[0].trim() });
  }

  const items = parts.map((content, index) => ({ content, index })).filter((item) => item.index > 0 && item.content.trim());
  items.forEach((item, displayIndex) => {
    const isLast = displayIndex === items.length - 1;
    const hasTools = TOOL_RE.test(item.content);
    if (internalFlags[item.index]) {
      segments.push({
        type: "turn",
        title: "Internal memory review",
        content: item.content.trim(),
        internal: true,
      });
    } else if (isLast && !hasTools) {
      const clean = item.content.replace(INFO_END_RE, "").replace(SUMMARY_RE, "").replace(THINKING_RE, "").trim();
      if (clean) {
        segments.push({ type: "markdown", content: clean });
      }
    } else {
      segments.push({
        type: "turn",
        title: extractTurnTitle(item.content, item.index),
        content: item.content.trim(),
      });
    }
  });

  return segments;
}

function parseContent(text: string): RenderSegment[] {
  const displayText = text.replace(INFO_END_RE, "");
  const segments: RenderSegment[] = [];
  let lastIndex = 0;
  for (const match of displayText.matchAll(ASK_BLOCK_RE)) {
    const before = displayText.slice(lastIndex, match.index);
    if (before.trim()) {
      segments.push(...parsePlainTurns(before));
    }
    try {
      segments.push({ type: "ask", payload: JSON.parse(match[1]) });
    } catch {
      segments.push({ type: "markdown", content: match[1] });
    }
    lastIndex = (match.index ?? 0) + match[0].length;
  }
  const rest = displayText.slice(lastIndex);
  if (rest.trim()) {
    segments.push(...parsePlainTurns(rest));
  }
  return segments;
}

function AskCard({ ask }: { ask: AskUserPayload }) {
  return (
    <Card className="border-blue-100 bg-blue-50/70 shadow-none">
      <CardContent className="space-y-4 p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-normal text-blue-700">Agent asks</span>
          <Badge variant="secondary">Input needed</Badge>
        </div>
        <Markdown content={ask.question || "Please provide input."} />
        {ask.candidates?.length ? (
          <div className="space-y-2">
            {ask.candidates.map((candidate, index) => (
              <div className="flex gap-3 rounded-md border border-blue-100 bg-white/80 p-3" key={`${candidate}-${index}`}>
                <span className="mt-0.5 text-sm font-semibold text-blue-700">{index + 1}</span>
                <Markdown content={candidate} />
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function MessageContent({ content, streaming = false }: { content: string; streaming?: boolean }) {
  const segments = useMemo(() => parseContent(content), [content]);
  if (!segments.length) {
    return streaming ? <span className="inline-flex h-4 w-1 animate-pulse rounded bg-blue-500" /> : null;
  }

  return (
    <div className="space-y-4">
      {segments.map((segment, index) => {
        if (segment.type === "ask") {
          return <AskCard key={index} ask={segment.payload} />;
        }
        if (segment.type === "turn") {
          return (
            <details
              className={cn(
                "overflow-hidden rounded-lg border bg-slate-50/80",
                segment.internal ? "border-amber-200 bg-amber-50/70" : "border-slate-200",
              )}
              key={index}
              open={streaming && index === segments.length - 1}
            >
              <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-slate-700">
                {segment.title}
              </summary>
              <div className="border-t border-slate-200 px-4 py-4">
                <Markdown content={segment.content} />
              </div>
            </details>
          );
        }
        return (
          <div key={index}>
            <Markdown content={segment.content} />
            {streaming && index === segments.length - 1 ? (
              <span className="mt-1 inline-flex h-4 w-1 animate-pulse rounded bg-blue-500" />
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function sessionTitle(session: SessionSummary) {
  return session.title?.trim() || "New Chat";
}

export function ChatShell() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const refreshSessions = useCallback(async () => {
    const data = await listSessions();
    setSessions(data);
    return data;
  }, []);

  const loadSession = useCallback(
    async (sessionId: string) => {
      const detail = await getSession(sessionId);
      setCurrentSessionId(detail.session_id);
      setMessages(detail.messages ?? []);
      await refreshSessions();
    },
    [refreshSessions],
  );

  useEffect(() => {
    let active = true;
    async function boot() {
      try {
        setError(null);
        const data = await refreshSessions();
        if (!active) {
          return;
        }
        if (data.length) {
          await loadSession(data[0].session_id);
        } else {
          const created = await createSession();
          setCurrentSessionId(created.session_id);
          setMessages(created.messages ?? []);
          await refreshSessions();
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }
    void boot();
    return () => {
      active = false;
    };
  }, [loadSession, refreshSessions]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, streaming]);

  const latestAssistant = [...messages].reverse().find((msg) => msg.role === "assistant");
  const isAnsweringAsk = Boolean(latestAssistant?.content.includes("[[ASK_USER]]"));

  async function handleNewSession() {
    const created = await createSession();
    setCurrentSessionId(created.session_id);
    setMessages(created.messages ?? []);
    await refreshSessions();
  }

  async function handleDeleteSession(sessionId: string) {
    await deleteSession(sessionId);
    const nextSessions = await refreshSessions();
    if (sessionId === currentSessionId) {
      if (nextSessions.length) {
        await loadSession(nextSessions[0].session_id);
      } else {
        await handleNewSession();
      }
    }
  }

  async function handleStop() {
    if (currentSessionId) {
      await cancelSession(currentSessionId);
    }
    abortRef.current?.abort();
    setStreaming(false);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || streaming) {
      return;
    }

    setInput("");
    setError(null);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);

    let activeSessionId = currentSessionId;
    let assistantContent = "";

    try {
      const returnedSessionId = await streamChat(
        currentSessionId,
        text,
        ({ sessionId, content }) => {
          activeSessionId = sessionId;
          assistantContent += content;
          setCurrentSessionId(sessionId);
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              next[next.length - 1] = { ...last, content: assistantContent };
            }
            return next;
          });
        },
        controller.signal,
      );

      activeSessionId = returnedSessionId || activeSessionId;
      if (activeSessionId) {
        setCurrentSessionId(activeSessionId);
        const detail = await getSession(activeSessionId);
        setMessages(detail.messages ?? []);
      }
      await refreshSessions();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant") {
            next[next.length - 1] = { ...last, content: `${assistantContent}\n\n**Error:** ${message}` };
          }
          return next;
        });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="grid h-screen overflow-hidden bg-transparent lg:grid-cols-[320px_1fr]">
      <aside className="hidden h-screen border-r border-slate-200/80 bg-white/80 backdrop-blur lg:flex lg:flex-col">
        <div className="flex items-center gap-3 px-6 py-5">
          <div className="grid h-11 w-11 place-items-center rounded-xl bg-blue-600 text-white shadow-sm">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-base font-semibold text-slate-900">PAI Assistant</h1>
            <p className="text-sm text-slate-500">Next.js client</p>
          </div>
        </div>
        <div className="px-4 pb-4">
          <Button className="w-full justify-start gap-2 rounded-lg" onClick={() => void handleNewSession()}>
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
        </div>
        <ScrollArea className="flex-1 px-3">
          <div className="space-y-2 pb-4">
            {sessions.map((session) => (
              <div className="flex items-center gap-2" key={session.session_id}>
                <button
                  className={cn(
                    "flex min-w-0 flex-1 items-center justify-between rounded-lg px-3 py-3 text-left text-sm transition-colors",
                    session.session_id === currentSessionId
                      ? "bg-blue-50 text-blue-700"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
                  )}
                  onClick={() => void loadSession(session.session_id)}
                >
                  <span className="truncate">{sessionTitle(session)}</span>
                  {session.running ? <Badge variant="outline">running</Badge> : null}
                </button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-slate-400 hover:bg-red-50 hover:text-red-600"
                  disabled={session.session_id === currentSessionId && sessions.length <= 1}
                  onClick={() => void handleDeleteSession(session.session_id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </ScrollArea>
        <div className="px-4 pb-5 pt-4">
          <Separator className="mb-4" />
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-[0_0_0_6px_rgba(34,197,94,0.12)]" />
            <span>Next proxy -&gt; FastAPI backend</span>
          </div>
        </div>
      </aside>

      <main className="grid h-screen min-h-0 grid-rows-[auto_auto_1fr_auto]">
        <header className="border-b border-slate-200/80 bg-white/75 px-5 py-4 backdrop-blur lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">{currentSessionId ? "Chat" : "Connecting"}</h2>
              <p className="text-sm text-slate-500">{loading ? "Loading sessions..." : `${messages.length} messages`}</p>
            </div>
            <div className="flex items-center gap-2">
              {streaming ? <Badge>Streaming</Badge> : <Badge variant="secondary">Idle</Badge>}
              <Button variant="outline" disabled={!streaming} onClick={() => void handleStop()}>
                <Square className="h-4 w-4" />
                Stop
              </Button>
            </div>
          </div>
        </header>

        {error ? (
          <div className="px-5 pt-4 lg:px-8">
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
          </div>
        ) : null}

        <ScrollArea className="min-h-0">
          <section className="flex w-full min-w-0 flex-col gap-6 px-4 py-6 lg:px-8">
            {!messages.length && !loading ? (
              <div className="mx-auto flex min-h-[50vh] max-w-2xl flex-col items-center justify-center text-center">
                <div className="mb-6 grid h-16 w-16 place-items-center rounded-2xl bg-blue-600 text-white shadow-panel">
                  <Bot className="h-8 w-8" />
                </div>
                <h3 className="text-3xl font-semibold text-slate-900">How can I help?</h3>
                <p className="mt-3 max-w-xl text-sm text-slate-500">
                  Ask a question, run a task, or continue an agent workflow with the current session state.
                </p>
              </div>
            ) : null}

            {messages.map((message, index) => (
              <article
                className={cn(
                  "flex w-full min-w-0 gap-4",
                  message.role === "user" ? "justify-end" : "justify-start",
                )}
                key={`${message.role}-${index}`}
              >
                {message.role !== "user" ? (
                  <div className="mt-1 hidden h-10 w-10 shrink-0 place-items-center rounded-xl bg-slate-900 text-white shadow-sm sm:grid">
                    <Bot className="h-4 w-4" />
                  </div>
                ) : null}

                <Card
                  className={cn(
                    "min-w-0 border shadow-sm",
                    message.role === "user"
                      ? "max-w-[78%] border-blue-200 bg-blue-600 text-white"
                      : "w-full border-slate-200 bg-white/90 shadow-panel",
                  )}
                >
                  <CardContent className="p-4 sm:p-5">
                    {message.role === "user" ? (
                      <div className="flex items-start gap-3">
                        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-white/15 text-white sm:hidden">
                          <User2 className="h-4 w-4" />
                        </div>
                        <div className="prose-agent max-w-none text-white [&_*]:text-inherit">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                        </div>
                      </div>
                    ) : (
                      <MessageContent
                        content={message.content}
                        streaming={streaming && index === messages.length - 1 && message.role === "assistant"}
                      />
                    )}
                  </CardContent>
                </Card>

                {message.role === "user" ? (
                  <div className="mt-1 hidden h-10 w-10 shrink-0 place-items-center rounded-xl bg-blue-600 text-white shadow-sm sm:grid">
                    <User2 className="h-4 w-4" />
                  </div>
                ) : null}
              </article>
            ))}

            {loading ? (
              <div className="flex items-center gap-3 text-sm text-slate-500">
                <LoaderCircle className="h-4 w-4 animate-spin" />
                Loading sessions
              </div>
            ) : null}
            <div ref={bottomRef} />
          </section>
        </ScrollArea>

        <div className="border-t border-slate-200/80 bg-white/80 px-4 py-4 backdrop-blur lg:px-8">
          <form className="flex w-full min-w-0 items-end gap-3" onSubmit={(event) => void handleSubmit(event)}>
            <Textarea
              className="max-h-40 min-h-[56px] resize-none rounded-xl bg-white shadow-sm"
              value={input}
              disabled={streaming}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSubmit(event);
                }
              }}
              placeholder={isAnsweringAsk ? "Answer the question..." : "Enter a task..."}
              rows={1}
            />
            <Button className="h-14 rounded-xl px-5" disabled={!input.trim() || streaming} type="submit">
              Send
            </Button>
          </form>
        </div>
      </main>
    </div>
  );
}
