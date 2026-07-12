import { Sun, Moon } from "lucide-react";
import { useTheme } from "../lib/theme";
import { ICON_BTN } from "../lib/ui";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button type="button" aria-label="Toggle theme" onClick={toggle} className={ICON_BTN}>
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}
