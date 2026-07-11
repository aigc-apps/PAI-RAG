import { useCallback } from "react";
import { create } from "zustand";
import { en } from "./en";
import { zh } from "./zh";

export type Lang = "zh" | "en";
export type MessageKey = keyof typeof en;
type Params = Record<string, string | number>;

const KEY = "agent-chat:lang";
const dicts: Record<Lang, Record<string, string>> = { en, zh };

function initialLang(): Lang {
  try {
    const stored = localStorage.getItem(KEY);
    if (stored === "en" || stored === "zh") return stored;
  } catch {
    /* ignore */
  }
  return "zh";
}

function applyLang(lang: Lang): void {
  if (typeof document !== "undefined") document.documentElement.lang = lang;
}

/** Interpolate `{name}` placeholders in a template with `params`. */
function interpolate(template: string, params?: Params): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, k: string) =>
    k in params ? String(params[k]) : `{${k}}`,
  );
}

/** Resolve a key for a language, falling back to English then the key itself. */
export function translate(lang: Lang, key: MessageKey, params?: Params): string {
  const template = dicts[lang]?.[key] ?? en[key] ?? String(key);
  return interpolate(template, params);
}

interface I18nState {
  lang: Lang;
  setLang: (lang: Lang) => void;
}

export const useI18nStore = create<I18nState>((set) => ({
  lang: initialLang(),
  setLang: (lang) => {
    try {
      localStorage.setItem(KEY, lang);
    } catch {
      /* ignore */
    }
    applyLang(lang);
    set({ lang });
  },
}));

// Reflect the initial language on <html lang> at module load.
applyLang(useI18nStore.getState().lang);

export type TFunction = (key: MessageKey, params?: Params) => string;

/**
 * Translation hook. Components re-render when the language changes because they
 * subscribe to the shared language store. Mirrors the ergonomics of the theme
 * hook: `const { t, lang, setLang } = useI18n()`.
 */
export function useI18n(): { t: TFunction; lang: Lang; setLang: (l: Lang) => void } {
  const lang = useI18nStore((s) => s.lang);
  const setLang = useI18nStore((s) => s.setLang);
  const t = useCallback<TFunction>((key, params) => translate(lang, key, params), [lang]);
  return { t, lang, setLang };
}
