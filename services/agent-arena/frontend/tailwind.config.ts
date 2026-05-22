import type { Config } from "tailwindcss"
import defaultTheme from "tailwindcss/defaultTheme"

const config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "1rem",
      screens: {
        "2xl": "1440px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        arena: {
          accent: {
            DEFAULT: "#FF5A1F",
            hover: "#E84818",
            press: "#C73A0E",
            soft: "#FFF1EC",
            tint: "#FFE2D5",
          },
          success: {
            DEFAULT: "#00A86B",
            soft: "#E5F6EE",
          },
          warning: {
            DEFAULT: "#E89C1A",
            soft: "#FCF1DC",
          },
          danger: {
            DEFAULT: "#E11D48",
            soft: "#FDE6EA",
          },
          info: {
            DEFAULT: "#2A60E0",
            soft: "#E4ECFD",
          },
          neutral: {
            DEFAULT: "#6B7280",
            soft: "#EEF0F3",
          },
          border: {
            DEFAULT: "#E6E8EC",
            strong: "#D5D8DE",
            focus: "#FF5A1F",
          },
          bg: {
            page: "#F4F5F7",
            card: "#FFFFFF",
            subtle: "#FAFAFB",
            code: "#0E1116",
            topbar: "#0E1116",
            sidebar: "#FFFFFF",
            hover: "#F2F3F5",
          },
          text: {
            primary: "#0E1116",
            secondary: "#4A525E",
            tertiary: "#8A93A1",
            inverse: "#FFFFFF",
            "mute-dark": "#98A0AE",
          },
        },
      },
      fontFamily: {
        sans: [
          "Manrope",
          ...defaultTheme.fontFamily.sans,
        ],
        mono: [
          "JetBrains Mono",
          ...defaultTheme.fontFamily.mono,
        ],
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
        "arena-sm": "4px",
        arena: "6px",
        "arena-lg": "10px",
      },
      boxShadow: {
        "arena-sm": "0 1px 2px rgba(14,17,22,0.04)",
        "arena-md": "0 4px 16px rgba(14,17,22,0.06)",
        "arena-lg": "0 12px 40px rgba(14,17,22,0.08)",
      },
      spacing: {
        topbar: "52px",
        sidebar: "220px",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "arena-pulse": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(255,90,31,0.6)" },
          "50%": { boxShadow: "0 0 0 6px rgba(255,90,31,0)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "arena-pulse": "arena-pulse 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config

export default config
