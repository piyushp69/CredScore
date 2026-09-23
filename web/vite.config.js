import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the FastAPI service so the app uses same-origin URLs.
// In production the built assets are served by that same service (see backend/app.py).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // /docs and /openapi.json back the sidebar's "API documentation" link.
    proxy: Object.fromEntries(
      ["/api", "/docs", "/openapi.json"].map((path) => [
        path,
        { target: process.env.CREDSCORE_API_URL || "http://127.0.0.1:8000", changeOrigin: true },
      ]),
    ),
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 900 },
});
