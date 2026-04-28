/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          green:  '#16a34a',
          green2: '#22c55e',
          dark:   '#0a0a0a',
          card:   '#111111',
          border: '#222222',
          muted:  '#888888',
        }
      }
    },
  },
  plugins: [],
}
