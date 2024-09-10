import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../static',
    // emptyOutDir: true,  // Comment out or remove this line
    sourcemap: true
  },
  server: {
    proxy: {
      '/ask': 'http://localhost:5000',
      '/chat': 'http://localhost:5000'
    }
  }
})