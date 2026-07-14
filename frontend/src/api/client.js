const raw = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_URL = raw.startsWith('http') ? raw : `https://${raw}`

export async function fetchWeeks(year = null) {
  const url = year != null
    ? `${API_URL}/api/weeks?year=${year}`
    : `${API_URL}/api/weeks`
  const res = await fetch(url)
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

export async function fetchSchedule(week = null, year = null) {
  const params = new URLSearchParams()
  if (week != null) params.set('week', week)
  if (year != null) params.set('year', year)
  const query = params.toString()
  const url = `${API_URL}/api/schedule${query ? '?' + query : ''}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch schedule')
  return res.json()
}

export async function fetchRankings(year = null) {
  const url = year != null
    ? `${API_URL}/api/rankings?year=${year}`
    : `${API_URL}/api/rankings`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch rankings')
  return res.json()
}

export async function fetchRecommendedBets(year = null) {
  const url = year != null
    ? `${API_URL}/api/recommended-bets?year=${year}`
    : `${API_URL}/api/recommended-bets`
  const res = await fetch(url)
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

export async function fetchScheduleForWeek(week) {
  const res = await fetch(`${API_URL}/api/schedule?week=${week}`)
  if (!res.ok) throw new Error('Failed to fetch schedule')
  return res.json()
}

export async function fetchContestYears() {
  const res = await fetch(`${API_URL}/api/contest/years/available`)
  if (!res.ok) throw new Error('Failed to fetch contest years')
  return res.json()
}

export async function fetchAvailableYears() {
  const res = await fetch(`${API_URL}/api/available-years`)
  if (!res.ok) throw new Error('Failed to fetch available years')
  return res.json()
}

export async function fetchContestData(year, asOfWeek = null) {
  const url = asOfWeek != null
    ? `${API_URL}/api/contest/${year}?as_of_week=${asOfWeek}`
    : `${API_URL}/api/contest/${year}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch contest data for ${year}`)
  return res.json()
}

export async function fetchBettingHistory(year = null) {
  const url = year != null
    ? `${API_URL}/api/betting-history?year=${year}`
    : `${API_URL}/api/betting-history`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch betting history')
  return res.json()
}

export async function fetchTransactionYears() {
  const res = await fetch(`${API_URL}/api/transactions/years/available`)
  if (!res.ok) throw new Error('Failed to fetch transaction years')
  return res.json()
}
 
export async function fetchTransactions(year) {
  const res = await fetch(`${API_URL}/api/transactions/${year}`)
  if (!res.ok) throw new Error(`Failed to fetch transactions for ${year}`)
  return res.json()
}

export async function fetchEntryAnalytics(year = null) {
  const url = year != null
    ? `${API_URL}/api/entry-analytics?year=${year}`
    : `${API_URL}/api/entry-analytics`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch entry analytics')
  return res.json()
}
