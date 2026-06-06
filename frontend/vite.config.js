import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on 5173. The Pipecat SDK connects directly to the agent on :7860,
// so no proxy is required, but we set a stable port and open the browser.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: true },
});
