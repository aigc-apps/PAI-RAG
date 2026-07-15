/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev-server proxy target. The app itself only ever calls relative `/v1` paths,
// so this is the single place the backend's address is known — override it when
// the backend is not on the default port:
//
//   BACKEND_URL=http://localhost:9000 npm run dev
//
// Build/preview don't proxy; there the `/v1` paths are served by whatever fronts
// the bundle, so this affects `npm run dev` only.
export default defineConfig(({ mode }) => {
  const backendUrl = loadEnv(mode, ".", "").BACKEND_URL || "http://localhost:8000";

  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        "/v1": {
          target: backendUrl,
          changeOrigin: true,
        },
      },
    },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["./vitest.setup.ts"],
    },
  };
});
