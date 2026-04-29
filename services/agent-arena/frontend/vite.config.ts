import path from "node:path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  base: process.env.ARENA_PUBLIC_BASE || "/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: Number(process.env.ARENA_FRONTEND_PORT || 5173),
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${process.env.ARENA_BACKEND_PORT || 8787}`,
        changeOrigin: true,
      },
    },
  },
})
