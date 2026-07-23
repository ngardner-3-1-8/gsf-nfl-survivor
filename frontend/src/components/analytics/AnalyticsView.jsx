import { useState, useEffect, useMemo } from 'react'
import { fetchEntryAnalytics, fetchEntryAnalyticsAvailable } from '../../api/client'

function fmtMoney(v) {
  if (v == null) return '—'
  return `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}
function fmtPct(v, digits = 2) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(digits)}%`
}
function survivalColor(p) {
  if (p == null) return 'text-gray-500'
  if (p >= 0.02) return 'text-green-400'
  if (p >= 0.005) return 'text-yellow-400'
  return 'text-red-400'
}

const PAGE_SIZE = 50

export default function AnalyticsView() {
  const [available, setAvailable] = useState({})   // { "2025": [2,3,4...] }
  const [year, setYear] = useState(null)
  const [week, setWeek] = useState(null)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState('rank')
  const [sortDir, setSortDir] = useState('asc')
  const [page, setPage] = useState(0)

  // 1. Load what's available, default to the newest year+week that has data
  useEffect(() => {
    fetchEntryAnalyticsAvailable()
      .then(d => {
        const avail = d.available || {}
        setAvailable(avail)
        const years = Object.keys(avail).map(Number).sort((a, b) => b - a)
        if (years.length === 0) {
          setError('No entry data available yet')
          setLoading(false)
          return
        }
        const y = years[0]
        const weeks = avail[String(y)] || []
        setYear(y)
        setWeek(weeks.length ? weeks[weeks.length - 1] : null)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  // 2. Load rankings whenever year/week changes
  useEffect(() => {
    if (year == null || week == null) return
    setLoading(true)
    setGenerating(true)
    setError(null)
    fetchEntryAnalytics(year, week)
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => { setLoading(false); setGenerating(false) })
  }, [year, week])

  const rankings = data?.rankings || []

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    let rows = q
      ? rankings.filter(r =>
          String(r.entry || '').toLowerCase().includes(q) ||
          String(r.contestant || '').toLowerCase().includes(q))
      : [...rankings]
    rows.sort((a, b) => {
      const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0
      const an = parseFloat(av), bn = parseFloat(bv)
      if (!isNaN(an) && !isNaN(bn)) return sortDir === 'asc' ? an - bn : bn - an
      return sortDir === 'asc'
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av))
    })
    return rows
  }, [rankings, search, sortKey, sortDir])

  useEffect(() => { setPage(0) }, [search, sortKey, sortDir, year, week])

  const pageRows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir(key === 'rank' ? 'asc' : 'desc') }
  }

  const SortTh = ({ col, label, right = false }) => (
    <th onClick={() => handleSort(col)}
      className={`px-4 py-2.5 text-xs font-medium text-gray-500 cursor-pointer hover:text-white select-none whitespace-nowrap ${right ? 'text-right' : 'text-left'}`}>
      {label}
      {sortKey === col && <span className="ml-1 text-green-400">{sortDir === 'asc' ? '↑' : '↓'}</span>}
    </th>
  )

  const years = Object.keys(available).map(Number).sort((a, b) => b - a)
  const weeksForYear = available[String(year)] || []

  const selectorBar = (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-3 flex-wrap">
      <span className="text-xs text-gray-500 uppercase tracking-wide">Season</span>
      <div className="flex gap-1.5 flex-wrap">
        {years.map(y => (
          <button key={y}
            onClick={() => {
              setYear(y)
              const w = available[String(y)] || []
              setWeek(w.length ? w[w.length - 1] : null)
            }}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
              year === y ? 'bg-green-600 text-white border-green-600'
                : 'border-gray-700 text-gray-400 hover:text-white'}`}>
            {y}
          </button>
        ))}
      </div>

      <span className="text-xs text-gray-500 uppercase tracking-wide ml-2">Week</span>
      <div className="flex gap-1 flex-wrap">
        {weeksForYear.map(w => (
          <button key={w} onClick={() => setWeek(w)}
            className={`text-xs px-2.5 py-1.5 rounded-lg border transition-colors font-mono ${
              week === w ? 'bg-gray-700 text-white border-gray-600'
                : 'border-gray-700 text-gray-500 hover:text-white'}`}>
            {w}
          </button>
        ))}
      </div>

      <div className="ml-auto">
        <input type="text" value={search} onChange={e => setSearch(e.target.value)}
          placeholder="Search entry or contestant..."
          className="bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-2 w-56 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600" />
      </div>
    </div>
  )

  if (loading) return (
    <div className="flex flex-col gap-4">
      {years.length > 0 && selectorBar}
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
        <span className="text-gray-400 text-sm">
          {generating ? `Building rankings for ${year} week ${week}...` : 'Loading...'}
        </span>
        {generating && (
          <span className="text-gray-600 text-xs">
            First load for a week runs the simulation and can take a minute
          </span>
        )}
      </div>
    </div>
  )

  if (error) return (
    <div className="flex flex-col gap-4">
      {years.length > 0 && selectorBar}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl flex flex-col items-center justify-center h-40 gap-2">
        <p className="text-gray-400 text-sm">Rankings unavailable for this selection</p>
        <p className="text-gray-600 text-xs">{error}</p>
      </div>
    </div>
  )

  const predictedPicks = data?.predicted_picks || []

  return (
    <div className="flex flex-col gap-4">
      {selectorBar}

      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-white font-semibold text-sm">
          {data?.year} {data?.mode === 'preseason' ? 'Preseason Projection' : `Week ${data?.week}`}
        </p>
        <p className="text-xs text-gray-500">
          {data?.entry_count?.toLocaleString()} alive entries · ranked by fair value
        </p>
      </div>

      {predictedPicks.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
          <p className="text-xs font-medium text-gray-400 mb-2">
            Predicted field picks this week
          </p>
          <div className="flex gap-2 flex-wrap">
            {predictedPicks.slice(0, 10).map(p => (
              <div key={p.team} className="bg-gray-800 rounded-lg px-3 py-1.5 flex items-center gap-2">
                <span className="text-white text-xs font-semibold">{p.team}</span>
                <span className="text-green-400 text-xs font-mono">
                  {(p.predicted_pick_pct * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <SortTh col="rank" label="Rank" />
                <SortTh col="entry" label="Entry" />
                <SortTh col="wins" label="Wins" right />
                <SortTh col="teams_remaining" label="Teams Left" right />
                <SortTh col="optimal_win_path_prob" label="Best Path Win%" right />
                <SortTh col="survival_prob" label="Survival Prob" right />
                <SortTh col="fair_value" label="Fair Value" right />
                <th className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">
                  Predicted Next Picks
                </th>
              </tr>
            </thead>
            <tbody>
              {pageRows.map(r => (
                <tr key={r.entry} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                  <td className="px-4 py-2.5 font-mono text-xs text-gray-400">#{r.rank}</td>
                  <td className="px-4 py-2.5">
                    <div className="flex flex-col">
                      <span className="text-white text-xs font-medium">{r.entry}</span>
                      {r.contestant !== r.entry && (
                        <span className="text-gray-600 text-xs">{r.contestant}</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-300">{r.wins}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-300">{r.teams_remaining}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-400">
                    {fmtPct(r.optimal_win_path_prob)}
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <span className={`font-mono text-xs ${survivalColor(r.survival_prob)}`}>
                      {fmtPct(r.survival_prob)}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <span className="font-mono text-xs font-semibold text-green-400">
                      {fmtMoney(r.fair_value)}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-xs text-gray-400">
                    {r.predicted_next_picks || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="px-4 py-2.5 border-t border-gray-800 flex items-center gap-3 text-xs">
          <span className="text-gray-600">
            {filtered.length.toLocaleString()} entries{search && ` matching "${search}"`}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
              className="px-2 py-1 rounded border border-gray-700 text-gray-400 hover:text-white disabled:opacity-30">
              ← Prev
            </button>
            <span className="text-gray-500">Page {page + 1} of {totalPages}</span>
            <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 rounded border border-gray-700 text-gray-400 hover:text-white disabled:opacity-30">
              Next →
            </button>
          </div>
        </div>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-xs text-gray-500">
        <p className="font-medium text-gray-400 mb-1">How these numbers are calculated</p>
        <p>
          Each entry is behaviorally profiled from its pick history (chalk vs contrarian,
          home/away lean, favorite tendency, EV alignment). Survival probability and fair
          value come from a Monte Carlo season simulation with shared game outcomes, so
          correlated eliminations are priced in. Switch weeks above to see how rankings
          shifted as the season progressed.
        </p>
      </div>
    </div>
  )
}
