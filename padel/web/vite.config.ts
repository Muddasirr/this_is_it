import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the API runs on :8010 (uvicorn padel_analyzer.api.main:app --port 8010)
export default defineConfig({
  plugins: [react()],
  server: { port: 5175, proxy: { '/api': 'http://localhost:8010' } },
})
