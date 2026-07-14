import { useState, useEffect, useMemo } from 'react'
import { fetchEntryAnalytics } from '../../api/client'

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
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState('rank')
  const [sortDir, setSortDir] = useState('asc')
  const [page, setPage] = useState(0)

  useEffect(() => {
    fetchEntryAnalytics()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const rankings = data?.rankings || []

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    let rows = q
      ? rankings.filter(r =>
          String(r.entry || '').toLowerCase().includes(q) ||
          String(r.contestant || '').toLowerCase().includes(q))
      : [...rankings]
    rows.sort((a, b) => {
      const av = a[sortKey] ?? 0
      const bv = b[sortKey] ?? 0
      const an = parseFloat(av), bn = parseFloat(bv)
      if (!isNaN(an) && !isNaN(bn)) return sortDir === 'asc' ? an - bn : bn - an
      return sortDir === 'asc'
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av))
    })
    return rows
  }, [rankings, search, sortKey, sortDir])

  // Reset page when the filter changes
  useEffect(() => { setPage(0) }, [search, sortKey, sortDir])

  const pageRows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir(key === 'rank' ? 'asc' : 'desc') }
  }

  const SortTh = ({ col, label, right = false }) => (
    <th
      onClick={() => handleSort(col)}
      className={`px-4 py-2.5 text-xs font-medium text-gray-500 cursor-pointer hover:text-white select-none whitespace-nowrap ${right ? 'text-right' : 'text-left'}`}
    >
      {label}
      {sortKey === col && (
        <span className="ml-1 text-green-400">{sortDir === 'asc' ? '↑' : '↓'}</span>
      )}
    </th>
  )

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading entry analytics...</span>
    </div>
  )

  if (error) return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl flex flex-col items-center justify-center h-48 gap-2">
      <p className="text-gray-400 text-sm">Entry analytics not available yet</p>
      <p className="text-gray-600 text-xs">Run the daily_2 entry analytics job to generate rankings</p>
    </div>
  )

  const predictedPicks = data?.predicted_picks || []

  return (
    <div className="flex flex-col gap-4">

      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div>
          <p className="text-white font-semibold text-sm">
            Entry Analytics — {data?.year}
            {data?.mode === 'preseason'
              ? ' (Preseason Projection)'
              : ` Week ${data?.week}`}
          </p>
          <p className="text-gray-500 text-xs mt-0.5">
            {data?.entry_count?.toLocaleString()} alive entries · fair value from
            season simulation · search for your entry below
          </p>
        </div>
        <div className="ml-auto">
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search entry or contestant name..."
            className="bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-2 w-64 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600"
          />
        </div>
      </div>

      {/* Predicted field pick% strip */}
      {predictedPicks.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
          <p className="text-xs font-medium text-gray-400 mb-2">
            Predicted field picks this week (behavioral model)
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

      {/* Entry table */}
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
                  <td className="px-4 py-2.5 font-mono text-xs text-gray-400">
                    #{r.rank}
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex flex-col">
                      <span className="text-white text-xs font-medium">{r.entry}</span>
                      {r.contestant !== r.entry && (
                        <span className="text-gray-600 text-xs">{r.contestant}</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-300">
                    {r.wins}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-300">
                    {r.teams_remaining}
                  </td>
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

        {/* Pagination */}
        <div className="px-4 py-2.5 border-t border-gray-800 flex items-center gap-3 text-xs">
          <span className="text-gray-600">
            {filtered.length.toLocaleString()} entries
            {search && ` matching "${search}"`}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-2 py-1 rounded border border-gray-700 text-gray-400 hover:text-white disabled:opacity-30"
            >
              ← Prev
            </button>
            <span className="text-gray-500">
              Page {page + 1} of {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 rounded border border-gray-700 text-gray-400 hover:text-white disabled:opacity-30"
            >
              Next →
            </button>
          </div>
        </div>
      </div>

      {/* Methodology */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-xs text-gray-500">
        <p className="font-medium text-gray-400 mb-1">How these numbers are calculated</p>
        <p>
          Each entry is behaviorally profiled from its pick history (chalk vs contrarian,
          home/away lean, favorite tendency, EV alignment). Pick predictions are fractional
          probabilities per team. Survival probability and fair value come from a Monte
          Carlo season simulation with shared game outcomes — so correlated eliminations
          (chalk entries dying together) are priced in. Fair value = expected pot share,
          useful as a reference price if buying or selling an entry.
        </p>
      </div>
    </div>
  )
}
