import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built UI is served by the Pipecat runner at /builder, so assets must be
// requested from there. In dev, /api and /client are proxied to that same
// server so the app behaves identically either way.
export default defineConfig({
  plugins: [react()],
  base: "/builder/",
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:7860",
      "/client": "http://localhost:7860",
    },
  },
});
