import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  preview: {
    host: '0.0.0.0',
    port: parseInt(process.env.PORT || '3000'),
    allowedHosts: [
      'localhost',
      '*.railway.app',  // Allow all Railway domains
      'frontend-production-683e.up.railway.app' // Your specific domain
    ]
  },
  server: {
    host: '0.0.0.0',
    port: parseInt(process.env.PORT || '3000'),
    allowedHosts: [
      'localhost',
      '*.railway.app',
      'frontend-production-683e.up.railway.app'
    ]
  }
})
