/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#060B1A',
          900: '#0B132B',
          850: '#111C3D',
          800: '#1C2541',
          700: '#27355B',
          600: '#3A4C7A',
        },
        electric: {
          green: '#10B981',
          cyan: '#06B6D4',
          accent: '#00F5A0',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)',
        'card-hover': '0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04)',
        'glow-green': '0 0 15px -3px rgba(16, 185, 129, 0.35)',
        'glow-cyan': '0 0 15px -3px rgba(6, 182, 212, 0.35)',
      }
    },
  },
  plugins: [],
}
