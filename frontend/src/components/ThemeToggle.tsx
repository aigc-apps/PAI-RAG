import { Sun, Moon } from "lucide-react";
import { useTheme } from "../lib/theme";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const label = theme === "dark" ? "Toggle theme: switch to light mode" : "Toggle theme: switch to dark mode";
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={toggle}
      className="focus-ring grid h-8 w-8 place-items-center rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-muted)] shadow-[var(--shadow-sm)] transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--surface)] hover:text-[var(--text)]"
    >
      {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
    </button>
  );
}
