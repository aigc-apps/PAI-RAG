import { useState, useEffect } from "react";

export type Theme = "light" | "dark";
const KEY = "agent-chat:theme";

function systemTheme(): Theme {
  return typeof matchMedia !== "undefined" && matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark" : "light";
}
export function getTheme(): Theme {
  const stored = (typeof localStorage !== "undefined" && localStorage.getItem(KEY)) as Theme | null;
  return stored === "light" || stored === "dark" ? stored : systemTheme();
}
export function applyTheme(t: Theme): void {
  if (typeof document !== "undefined") document.documentElement.dataset.theme = t;
}
export function setTheme(t: Theme): Theme {
  try { localStorage.setItem(KEY, t); } catch { /* ignore */ }
  applyTheme(t);
  return t;
}
export function toggleTheme(): Theme {
  return setTheme(getTheme() === "dark" ? "light" : "dark");
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getTheme);
  useEffect(() => { applyTheme(theme); }, [theme]);
  return {
    theme,
    toggle: () => {
      const next: Theme = theme === "dark" ? "light" : "dark";
      setTheme(next);
      setThemeState(next);
    },
  };
}
