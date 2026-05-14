const raw = import.meta.env.VITE_API_URL || 'http://localhost:8000'
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

export async function fetchPickPercentages() {
  const res = await fetch(`${API_URL}/api/pick-percentages`)
  if (!res.ok) throw new Error('Failed to fetch pick percentages')
  return res.json()
}

export async function fetchLastUpdated() {
  const res = await fetch(`${API_URL}/api/last-updated`)
  if (!res.ok) throw new Error('Failed to fetch last updated')
  return res.json()
}

export async function fetchSchedule(week = null) {
  const url = week
    ? `${API_URL}/api/schedule?week=${week}`
    : `${API_URL}/api/schedule`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch schedule')
  return res.json()
}

export async function fetchRankings() {
  const res = await fetch(`${API_URL}/api/rankings`)
  if (!res.ok) throw new Error('Failed to fetch rankings')
  return res.json()
}

export async function fetchRecommendedBets() {
  const res = await fetch(`${API_URL}/api/recommended-bets`)
  if (!res.ok) throw new Error('Failed to fetch recommended bets')
  return res.json()
}

export async function getBets(username) {
  const res = await fetch(`${API_URL}/api/bets/${encodeURIComponent(username)}`)
  if (!res.ok) throw new Error('Failed to fetch bets')
  return res.json()
}

export async function addBet(username, bet) {
  const res = await fetch(`${API_URL}/api/bets/${encodeURIComponent(username)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(bet),
  })
  if (!res.ok) throw new Error('Failed to add bet')
  return res.json()
}

export async function updateBet(username, betId, updates) {
  const res = await fetch(`${API_URL}/api/bets/${encodeURIComponent(username)}/${betId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  })
  if (!res.ok) throw new Error('Failed to update bet')
  return res.json()
}

export async function deleteBet(username, betId) {
  const res = await fetch(`${API_URL}/api/bets/${encodeURIComponent(username)}/${betId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete bet')
  return res.json()
}
