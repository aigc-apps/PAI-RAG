'use client';

import type { FC } from 'react';
import { MarkdownRenderer } from '@/components/customized/markdown/markdown';

/**
 * Smart renderer for tool arguments / results.
 *
 * - Objects / arrays → fenced ```json``` code block, rendered with the
 *   project's shared MarkdownRenderer so we get Prism syntax highlighting
 *   instead of a wall of monospace text.
 * - Strings that parse as JSON → same treatment. If the payload is the
 *   common `{data: "..."}` wrapper our agent tools emit, the inner string
 *   is unwrapped and rendered as Markdown so headings, bullet lists, and
 *   tables in extracted PDFs come through cleanly.
 * - Plain strings → rendered as Markdown (graceful no-op for text without
 *   markdown syntax).
 *
 * The outer wrapper neutralises Tailwind Typography's chunky vertical
 * rhythm so the content fits inside the compact tool-call card without
 * dominating the stream.
 */

function looksLikeJson(text: string): boolean {
  const t = text.trimStart();
  return t.startsWith('{') || t.startsWith('[');
}

function unwrapDataField(value: unknown): string | null {
  if (
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    'data' in value &&
    typeof (value as { data: unknown }).data === 'string' &&
    Object.keys(value as object).length === 1
  ) {
    return (value as { data: string }).data;
  }
  return null;
}

const toolContentClass =
  // Tight typography so the renderer fits inside a compact tool card.
  'text-[12px] leading-relaxed ' +
  '[&_.prose]:max-w-none ' +
  '[&_.prose_p]:my-1 ' +
  '[&_.prose_ul]:my-1 [&_.prose_ol]:my-1 [&_.prose_li]:my-0.5 ' +
  '[&_.prose_h1]:text-sm [&_.prose_h1]:my-1.5 [&_.prose_h1]:font-semibold ' +
  '[&_.prose_h2]:text-sm [&_.prose_h2]:my-1.5 [&_.prose_h2]:font-semibold ' +
  '[&_.prose_h3]:text-[13px] [&_.prose_h3]:my-1 [&_.prose_h3]:font-semibold ' +
  '[&_.prose_h4]:text-[12px] [&_.prose_h4]:my-1 [&_.prose_h4]:font-semibold ' +
  '[&_.prose_pre]:my-1.5 [&_.prose_pre]:text-[11px] ' +
  '[&_.prose_code]:text-[11px] ' +
  '[&_.prose_table]:my-1.5 [&_.prose_table]:text-[11px]';

export const ToolContent: FC<{ value: unknown }> = ({ value }) => {
  if (value === null || value === undefined) return null;

  // Non-string payloads → JSON fence (arrays, numbers, objects).
  if (typeof value !== 'string') {
    const text = '```json\n' + safeStringify(value) + '\n```';
    return (
      <div className={toolContentClass}>
        <MarkdownRenderer content={text} />
      </div>
    );
  }

  const trimmed = value.trim();
  if (!trimmed) return null;

  if (looksLikeJson(trimmed)) {
    try {
      const parsed = JSON.parse(trimmed);
      const unwrapped = unwrapDataField(parsed);
      if (unwrapped !== null) {
        return (
          <div className={toolContentClass}>
            <MarkdownRenderer content={unwrapped} />
          </div>
        );
      }
      return (
        <div className={toolContentClass}>
          <MarkdownRenderer
            content={'```json\n' + safeStringify(parsed) + '\n```'}
          />
        </div>
      );
    } catch {
      // Fall through — treat the raw text as markdown.
    }
  }

  return (
    <div className={toolContentClass}>
      <MarkdownRenderer content={trimmed} />
    </div>
  );
};

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
