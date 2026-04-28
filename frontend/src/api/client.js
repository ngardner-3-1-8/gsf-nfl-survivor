const raw = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Ensure the URL always starts with http
const API_URL = raw.startsWith('http') ? raw : `https://${raw}`

export async function fetchWeeks() {
  const res = await fetch(`${API_URL}/api/weeks`)
  if (!res.ok) throw new Error('Failed to fetch weeks')
  return res.json()
}

export async function runOptimizer(constraints) {
  const res = await fetch(`${API_URL}/api/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(constraints),
  })
  if (!res.ok) throw new Error('Optimizer request failed')
  return res.json()
}
