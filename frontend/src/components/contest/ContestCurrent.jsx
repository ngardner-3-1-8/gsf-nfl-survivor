import { useState, useEffect } from 'react'
import { fetchContestData, fetchSchedule } from '../../api/client'

function estimatePrize(totalEntries) {
  return totalEntries * 1500 * 0.9
}

function calcSurvivalOdds(remainingWeeks, avgEliminationRate) {
  let prob = 1.0
  for (let w = 0; w < remainingWeeks; w++) {
    prob *= (1 - avgEliminationRate)
  }
  return prob
}

// Format currency with commas
function formatCurrency(val) {
  if (!val && val !== 0) return '—'
  return '$' + Math.round(val).toLocaleString()
}

export default function ContestCurrent({ years }) {
  const [selectedYear, setSelectedYear] = useState(years[0] || null)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [viewMode, setViewMode] = useState('contestants') // contestants | entries | all-entries
  const [asOfWeek, setAsOfWeek] = useState(null) // null = latest
  const [searchTerm, setSearchTerm] = useState('')
  const [upcomingWeek, setUpcomingWeek] = useState(null)
  const [entrySearch, setEntrySearch] = useState('')

  useEffect(() => {
    if (years.length > 0 && !selectedYear) setSelectedYear(years[0])
  }, [years])

  useEffect(() => {
    if (!selectedYear) return
    setLoading(true)
    Promise.all([
      fetchContestData(selectedYear),
      fetchSchedule(),
    ])
      .then(([contestData, schedData]) => {
        setData(contestData)
        const uw = schedData.upcoming_week || contestData.summary?.num_weeks
        setUpcomingWeek(uw)
        setAsOfWeek(uw) // default to current week
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [selectedYear])

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading...</span>
    </div>
  )

  if (!data) return (
    <div className="flex items-center justify-center h-40">
      <p className="text-gray-500 text-sm">No data available</p>
    </div>
  )

  const { summary, survival_curve, contestant_stats, entry_stats, all_entries } = data
  const effectiveWeek = asOfWeek || summary.num_weeks

  // Stats as-of selected week
  const currentCurve = survival_curve.find(c => c.week === effectiveWeek)
    || survival_curve[survival_curve.length - 1]
  const survivors = currentCurve?.surviving || 0
  
  // Contestants alive as-of selected week
  const contestantsAlive = new Set(
    (all_entries || [])
      .filter(e => e.total_wins >= effectiveWeek)
      .map(e => e.contestant)
  ).size

  const prizePool = estimatePrize(summary.total_entries)
  const settledWeeks = survival_curve.filter(c => c.week > 1 && c.week <= effectiveWeek)
  const avgElimRate = settledWeeks.length > 0
    ? settledWeeks.reduce((s, c) => s + c.pct_eliminated / 100, 0) / settledWeeks.length
    : 0.15
  const remainingWeeks = summary.num_weeks - effectiveWeek + 1
  const survivalOdds = calcSurvivalOdds(remainingWeeks, avgElimRate)
  const expectedValuePerEntry = survivors > 0
    ? (prizePool / survivors) * survivalOdds
    : 0

  // Filter entries/contestants as-of week
  const survivingEntries = (entry_stats || []).filter(e => e.total_wins >= effectiveWeek)
  const survivingContestants = (contestant_stats || []).filter(c => c.surviving > 0)

  const filteredContestants = survivingContestants.filter(c =>
    !searchTerm || c.contestant.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredAllEntries = (all_entries || []).filter(e => {
    const alive = e.total_wins >= effectiveWeek
    const matchSearch = !entrySearch ||
      e.entry.toLowerCase().includes(entrySearch.toLowerCase()) ||
      e.contestant.toLowerCase().includes(entrySearch.toLowerCase())
    return alive && matchSearch
  })

  const weekOptions = Array.from({ length: summary.num_weeks }, (_, i) => i + 1)

  return (
    <div className="flex flex-col gap-6">

      {/* Controls bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400 uppercase tracking-wide">Season</label>
          <select
            value={selectedYear || ''}
            onChange={e => setSelectedYear(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          >
            {years.map(y => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400 uppercase tracking-wide">As of Week</label>
          <select
            value={asOfWeek || ''}
            onChange={e => setAsOfWeek(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          >
            {weekOptions.map(w => (
              <option key={w} value={w}>
                Week {w}{w === upcomingWeek ? ' (current)' : ''}
              </option>
            ))}
          </select>
        </div>

        {/* View tabs */}
        <div className="flex gap-1 ml-auto">
          {[
            { id: 'contestants', label: 'Contestants' },
            { id: 'entries', label: 'Entry Paths' },
            { id: 'all-entries', label: 'All Entries' },
          ].map(v => (
            <button
              key={v.id}
              onClick={() => setViewMode(v.id)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                viewMode === v.id
                  ? 'bg-gray-700 text-white border-gray-600 font-medium'
                  : 'border-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>

      {/* Key stats */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {[
          {
            label: `Wk ${effectiveWeek} Entries`,
            value: survivors.toLocaleString(),
            sub: `of ${summary.total_entries.toLocaleString()} total`,
            color: 'text-green-400',
          },
          {
            label: `Wk ${effectiveWeek} Contestants`,
            value: contestantsAlive.toLocaleString(),
            sub: `of ${summary.total_contestants.toLocaleString()} total`,
            color: 'text-blue-400',
          },
          {
            label: 'Est. Prize Pool',
            value: formatCurrency(prizePool),
            sub: `${summary.total_entries.toLocaleString()} entries`,
            color: 'text-yellow-400',
          },
          {
            label: 'Avg Survival Odds',
            value: `${(survivalOdds * 100).toFixed(2)}%`,
            sub: `${remainingWeeks} weeks left`,
            color: 'text-purple-400',
          },
          {
            label: 'Exp. Value / Entry',
            value: formatCurrency(expectedValuePerEntry),
            sub: 'if currently surviving',
            color: 'text-orange-400',
          },
        ].map(s => (
          <div key={s.label} className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
            <p className="text-xs text-gray-500 mb-1">{s.label}</p>
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
            <p className="text-xs text-gray-600 mt-0.5">{s.sub}</p>
          </div>
        ))}
      </div>

      {/* Survival curve table */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <p className="text-white font-semibold text-sm">
            {selectedYear} Survival Curve
            <span className="ml-2 text-xs text-gray-500 font-normal">
              → Week {effectiveWeek} highlighted
            </span>
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                {['Week', 'Entries', 'Contestants', '% Entries Left', 'Eliminated', '% Eliminated', 'Cumul. Elim%'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {survival_curve.map(row => {
                const isCurrent = row.week === effectiveWeek
                const cumElim = ((summary.total_entries - row.surviving) / summary.total_entries * 100)
                // Contestants alive this week
                const contestantsThisWeek = new Set(
                  (all_entries || [])
                    .filter(e => e.total_wins >= row.week)
                    .map(e => e.contestant)
                ).size
                return (
                  <tr
                    key={row.week}
                    className={`border-b border-gray-800/50 cursor-pointer ${
                      isCurrent ? 'bg-green-950/30' : 'hover:bg-gray-800/20'
                    }`}
                    onClick={() => setAsOfWeek(row.week)}
                    title="Click to view stats as of this week"
                  >
                    <td className={`px-4 py-2.5 font-medium text-xs ${isCurrent ? 'text-green-400' : 'text-gray-400'}`}>
                      {isCurrent ? '→ ' : ''}Wk {row.week}
                    </td>
                    <td className="px-4 py-2.5 text-white font-mono text-xs">{row.surviving.toLocaleString()}</td>
                    <td className="px-4 py-2.5 text-blue-300 font-mono text-xs">{contestantsThisWeek.toLocaleString()}</td>
                    <td className="px-4 py-2.5 font-mono text-xs">
                      <span className={row.pct_remaining >= 50 ? 'text-green-400' : row.pct_remaining >= 10 ? 'text-yellow-400' : 'text-red-400'}>
                        {row.pct_remaining.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-gray-400 font-mono text-xs">{row.eliminated.toLocaleString()}</td>
                    <td className="px-4 py-2.5 font-mono text-xs">
                      <span className={row.pct_eliminated >= 20 ? 'text-red-400' : row.pct_eliminated >= 5 ? 'text-yellow-400' : 'text-gray-400'}>
                        {row.pct_eliminated.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-gray-500 font-mono text-xs">{cumElim.toFixed(1)}%</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-2.5 border-t border-gray-800 text-xs text-gray-600">
          Click any row to view stats as of that week
        </div>
      </div>

      {/* ── View: Contestants ── */}
      {viewMode === 'contestants' && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800 flex items-center gap-3 flex-wrap">
            <p className="text-white font-semibold text-sm">
              Contestant Leaderboard — Week {effectiveWeek}
            </p>
            <span className="text-gray-500 text-xs">
              {filteredContestants.length} with surviving entries
            </span>
            <input
              type="text"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              placeholder="Search contestant..."
              className="ml-auto bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-1.5 w-48 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600"
            />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  {['#', 'Contestant', 'Entries', 'Surviving', 'Max Wins', 'Avg Wins',
                    'Best EV Path', 'Best Win% Path', 'Avg Teams Left', 'Pool Strength', 'Exp. Value'].map(h => (
                    <th key={h} className="text-left px-3 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredContestants.slice(0, 100).map((c, i) => {
                  const expValue = c.surviving * expectedValuePerEntry
                  return (
                    <tr key={c.contestant} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                      <td className="px-3 py-2.5 text-gray-500 text-xs font-mono">{i + 1}</td>
                      <td className="px-3 py-2.5 text-white font-medium text-xs">{c.contestant}</td>
                      <td className="px-3 py-2.5 text-gray-400 text-xs font-mono">{c.entries}</td>
                      <td className="px-3 py-2.5 text-xs font-mono">
                        <span className={c.surviving > 0 ? 'text-green-400 font-semibold' : 'text-gray-600'}>
                          {c.surviving}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-gray-300 text-xs font-mono">{c.max_wins}</td>
                      <td className="px-3 py-2.5 text-gray-400 text-xs font-mono">{c.avg_wins}</td>
                      <td className="px-3 py-2.5 text-green-400 text-xs font-mono">{c.best_ev_path?.toFixed(4)}</td>
                      <td className="px-3 py-2.5 text-blue-400 text-xs font-mono">{c.best_win_path?.toFixed(1)}%</td>
                      <td className="px-3 py-2.5 text-gray-400 text-xs font-mono">{c.avg_remaining_teams}</td>
                      <td className="px-3 py-2.5 text-xs font-mono">
                        <span className={c.avg_pool_strength >= 60 ? 'text-green-400' : c.avg_pool_strength >= 55 ? 'text-yellow-400' : 'text-gray-400'}>
                          {c.avg_pool_strength?.toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-purple-400 text-xs font-mono">
                        {expValue > 0 ? formatCurrency(expValue) : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── View: Entry Paths ── */}
      {viewMode === 'entries' && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-white font-semibold text-sm">
              Best Remaining Paths — Entry Level · Week {effectiveWeek}
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  {['#', 'Entry', 'Contestant', 'Wins', 'Teams Left',
                    'Pool Strength', 'Best EV Path', 'Best Win% Path', 'Remaining Teams'].map(h => (
                    <th key={h} className="text-left px-3 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {survivingEntries.slice(0, 200).map((e, i) => (
                  <tr key={e.entry} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                    <td className="px-3 py-2.5 text-gray-500 text-xs font-mono">{i + 1}</td>
                    <td className="px-3 py-2.5 text-white font-medium text-xs">{e.entry}</td>
                    <td className="px-3 py-2.5 text-gray-400 text-xs">{e.contestant}</td>
                    <td className="px-3 py-2.5 text-gray-300 text-xs font-mono">{e.total_wins}</td>
                    <td className="px-3 py-2.5 text-xs font-mono">
                      <span className={e.remaining_count >= 20 ? 'text-green-400' : e.remaining_count >= 10 ? 'text-yellow-400' : 'text-red-400'}>
                        {e.remaining_count}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-xs font-mono">
                      <span className={e.pool_avg_win_pct >= 60 ? 'text-green-400' : e.pool_avg_win_pct >= 55 ? 'text-yellow-400' : 'text-gray-400'}>
                        {e.pool_avg_win_pct?.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-green-400 text-xs font-mono font-semibold">{e.optimal_ev_path?.toFixed(4)}</td>
                    <td className="px-3 py-2.5 text-blue-400 text-xs font-mono">{e.optimal_win_path?.toFixed(1)}%</td>
                    <td className="px-3 py-2.5 text-xs text-gray-500 max-w-xs truncate">
                      {(e.remaining_teams || []).join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── View: All Entries ── */}
      {viewMode === 'all-entries' && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800 flex items-center gap-3 flex-wrap">
            <p className="text-white font-semibold text-sm">
              All Entries — Week {effectiveWeek}
            </p>
            <span className="text-gray-500 text-xs">{filteredAllEntries.length} surviving entries</span>
            <input
              type="text"
              value={entrySearch}
              onChange={e => setEntrySearch(e.target.value)}
              placeholder="Search entry or contestant..."
              className="ml-auto bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-1.5 w-56 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600"
            />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left px-3 py-2.5 text-xs font-medium text-gray-500">Entry</th>
                  <th className="text-left px-3 py-2.5 text-xs font-medium text-gray-500">Contestant</th>
                  <th className="text-left px-3 py-2.5 text-xs font-medium text-gray-500">Wins</th>
                  {/* Week columns */}
                  {Array.from({ length: effectiveWeek }, (_, i) => i + 1).map(w => (
                    <th key={w} className="text-center px-2 py-2.5 text-xs font-medium text-gray-500 min-w-[40px]">
                      W{w}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredAllEntries.slice(0, 200).map(e => (
                  <tr key={e.entry} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                    <td className="px-3 py-2 text-white font-medium text-xs">{e.entry}</td>
                    <td className="px-3 py-2 text-gray-400 text-xs">{e.contestant}</td>
                    <td className="px-3 py-2 text-gray-300 text-xs font-mono">{e.total_wins}</td>
                    {Array.from({ length: effectiveWeek }, (_, i) => i + 1).map(w => (
                      <td key={w} className="px-2 py-2 text-center">
                        <span className="text-xs font-mono text-gray-300">
                          {e.picks?.[w] || '—'}
                        </span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filteredAllEntries.length > 200 && (
            <div className="px-4 py-2.5 border-t border-gray-800 text-xs text-gray-500">
              Showing 200 of {filteredAllEntries.length} entries — use search to narrow results
            </div>
          )}
        </div>
      )}

    </div>
  )
}
