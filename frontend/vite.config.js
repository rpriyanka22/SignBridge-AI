import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: [
      '@mediapipe/holistic',
      '@mediapipe/camera_utils',
      '@mediapipe/drawing_utils'
    ]
  }
});