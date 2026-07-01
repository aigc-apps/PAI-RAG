import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, Check } from "lucide-react";

const plainOutputLanguages = new Set(["text", "txt", "plain", "plaintext", "output", "console"]);

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const isPlainOutput = plainOutputLanguages.has(language.toLowerCase());
  const label = isPlainOutput ? "Output" : language;

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="group my-3 overflow-hidden rounded-[var(--radius-lg)] border border-[var(--code-border)] bg-[var(--code-bg)] shadow-[var(--shadow-sm)]">
      <div className="flex h-9 items-center justify-between border-b border-[var(--code-border)] bg-[var(--code-header)] px-3">
        <span className="font-mono text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
          {label}
        </span>
        <button
          type="button"
          aria-label="Copy code"
          onClick={copy}
          className="rounded-[var(--radius-sm)] p-1 text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>
      {isPlainOutput ? (
        <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words bg-[var(--code-bg)] px-4 py-3 font-mono text-[13px] leading-6 text-[var(--code-text)]">
          <code>{code}</code>
        </pre>
      ) : (
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          PreTag="div"
          customStyle={{
            margin: 0,
            background: "var(--code-bg)",
            color: "var(--code-text)",
            fontSize: "13px",
            padding: "14px 16px",
            lineHeight: 1.65,
          }}
        >
          {code}
        </SyntaxHighlighter>
      )}
    </div>
  );
}

export function Markdown({ content }: { content: string }) {
  return (
    <div className="max-w-none break-words text-[var(--text)] text-sm leading-7">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="mb-3 mt-6 text-xl font-bold text-[var(--text)]">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mb-2 mt-5 text-lg font-semibold text-[var(--text)]">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-2 mt-4 text-base font-semibold text-[var(--text)]">{children}</h3>
          ),
          h4: ({ children }) => (
            <h4 className="mb-1 mt-3 text-sm font-semibold text-[var(--text)]">{children}</h4>
          ),
          p: ({ children }) => <p className="mb-3 leading-7">{children}</p>,
          ul: ({ children }) => (
            <ul className="mb-3 list-disc space-y-1 pl-6 marker:text-[var(--text-faint)]">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-3 list-decimal space-y-1 pl-6 marker:text-[var(--text-faint)]">{children}</ol>
          ),
          li: ({ children }) => <li className="leading-7">{children}</li>,
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-[var(--accent)] underline underline-offset-2 hover:opacity-80"
              target="_blank"
              rel="noopener noreferrer"
            >
              {children}
            </a>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-[var(--text)]">{children}</strong>
          ),
          blockquote: ({ children }) => (
            <blockquote className="mb-3 border-l-2 border-[var(--border-strong)] pl-4 italic text-[var(--text-muted)]">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-4 border-0 border-t border-[var(--border)]" />,
          table: ({ children }) => (
            <div className="mb-3 overflow-x-auto">
              <table className="w-full border-collapse text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="border-b border-[var(--border-strong)]">{children}</thead>,
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-semibold text-[var(--text)] border-b border-[var(--border-strong)]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 border-b border-[var(--border)] text-[var(--text-muted)]">
              {children}
            </td>
          ),
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const isBlock = Boolean(match);
            if (!isBlock) {
              return (
                <code
                  className="font-mono text-[13px] rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-1.5 py-0.5 text-[var(--text)]"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <CodeBlock
                language={match![1]}
                code={String(children).replace(/\n$/, "")}
              />
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
