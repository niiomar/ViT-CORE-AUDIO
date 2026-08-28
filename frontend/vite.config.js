import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // 1. DEVELOPMENT SETTINGS (Routing localhost:5173 to FastAPI)
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8003',
        changeOrigin: true,
        secure: false,
      }
    }
  },

  // 2. PRODUCTION SETTINGS (Building directly into the backend)
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
      },
    },
  },

  // 3. TEST SETTINGS (read by `vitest run` — ignored by plain `vite build`)
  test: {
    environment: 'jsdom',
    globals: false,
  },
});
