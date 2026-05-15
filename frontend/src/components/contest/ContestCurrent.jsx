import { useState, useEffect } from 'react'
import { fetchContestData, fetchSchedule } from '../../api/client'

// Circa prize structure estimate
// $1,000 entry × N entries — top survivors split
function estimatePrize(totalEntries) {
  return totalEntries * 1000 * 1 // 100% of entry fees to prize pool
}

function calcSurvivalOdds(currentSurvivors, remainingWeeks, avgEliminationRate) {
  // Geometric model: probability of surviving each future week
  let prob = 1.0
  for (let w = 0; w < remainingWeeks; w++) {
    prob *= (1 - avgEliminationRate)
  }
  return prob
}

export default function ContestCurrent({ years }) {
  const currentYear = years[0] || new Date().getFullYear()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [viewMode, setViewMode] = useState('contestant') // contestant | entry
  const [searchTerm, setSearchTerm] = useState('')
  const [upcomingWeek, setUpcomingWeek] = useState(null)

  useEffect(() => {
    Promise.all([
      fetchContestData(currentYear),
      fetchSchedule(),
    ])
      .then(([contestData, schedData]) => {
        setData(contestData)
        setUpcomingWeek(schedData.upcoming_week)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [currentYear])

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading current season data...</span>
    </div>
  )

  if (!data) return (
    <div className="flex items-center justify-center h-40">
      <p className="text-gray-500 text-sm">No data available for {currentYear}</p>
    </div>
  )

  const { summary, survival_curve, contestant_stats } = data
  const currentWeek = upcomingWeek || summary.num_weeks
  const currentCurve = survival_curve.find(c => c.week === currentWeek) || survival_curve[survival_curve.length - 1]
  const survivors = currentCurve?.surviving || 0
  const prizePool = estimatePrize(summary.total_entries)

  // Average weekly elimination rate from history
  const settledWeeks = survival_curve.filter(c => c.week > 1 && c.week <= currentWeek)
  const avgElimRate = settledWeeks.length > 0
    ? settledWeeks.reduce((s, c) => s + c.pct_eliminated / 100, 0) / settledWeeks.length
    : 0.15
  const remainingWeeks = summary.num_weeks - currentWeek + 1

  // Per-entry survival odds and expected value
  const survivalOdds = calcSurvivalOdds(survivors, remainingWeeks, avgElimRate)
  const expectedValuePerEntry = survivors > 0
    ? (prizePool / survivors) * survivalOdds
    : 0

  // Filter contestants
  const filtered = contestant_stats.filter(c =>
    !searchTerm || c.contestant.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="flex flex-col gap-6">

      {/* Key stats bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: `Week ${currentWeek} Survivors`, value: survivors.toLocaleString(), sub: `${currentCurve?.pct_remaining?.toFixed(1)}% remaining`, color: 'text-green-400' },
          { label: 'Est. Prize Pool', value: `$${(prizePool / 1e6).toFixed(2)}M`, sub: `${summary.total_entries.toLocaleString()} entries`, color: 'text-yellow-400' },
          { label: 'Avg Survival Odds', value: `${(survivalOdds * 100).toFixed(2)}%`, sub: `${remainingWeeks} weeks remaining`, color: 'text-blue-400' },
          { label: 'Exp. Value / Entry', value: `$${expectedValuePerEntry.toFixed(0)}`, sub: 'if currently surviving', color: 'text-purple-400' },
        ].map(s => (
          <div key={s.label} className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
            <p className="text-xs text-gray-500 mb-1">{s.label}</p>
            <p className={`text-xl font-bold ${s.color}`}>{s.value}</p>
            <p className="text-xs text-gray-600 mt-0.5">{s.sub}</p>
          </div>
        ))}
      </div>

      {/* Current survival curve */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
        <p className="text-white font-semibold text-sm mb-4">
          {currentYear} Survival Curve
          <span className="ml-2 text-xs text-gray-500 font-normal">
            Week {currentWeek} highlighted
          </span>
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800">
                {['Week', 'Survivors', '% Remaining', 'Eliminated', '% Eliminated', 'Cumulative Elim%'].map(h => (
                  <th key={h} className="text-left px-3 py-2 text-gray-500 font-medium whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {survival_curve.map(row => {
                const isCurrent = row.week === currentWeek
                const cumElim = ((summary.total_entries - row.surviving) / summary.total_entries * 100)
                return (
                  <tr
                    key={row.week}
                    className={`border-b border-gray-800/50 ${isCurrent ? 'bg-green-950/30' : 'hover:bg-gray-800/20'}`}
                  >
                    <td className={`px-3 py-2 font-medium ${isCurrent ? 'text-green-400' : 'text-gray-400'}`}>
                      {isCurrent ? '→ ' : ''}Wk {row.week}
                    </td>
                    <td className="px-3 py-2 text-white font-mono">{row.surviving.toLocaleString()}</td>
                    <td className="px-3 py-2 font-mono">
                      <span className={row.pct_remaining >= 50 ? 'text-green-400' : row.pct_remaining >= 10 ? 'text-yellow-400' : 'text-red-400'}>
                        {row.pct_remaining.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-400 font-mono">{row.eliminated.toLocaleString()}</td>
                    <td className="px-3 py-2 font-mono">
                      <span className={row.pct_eliminated >= 20 ? 'text-red-400' : row.pct_eliminated >= 5 ? 'text-yellow-400' : 'text-gray-400'}>
                        {row.pct_eliminated.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-500 font-mono">{cumElim.toFixed(1)}%</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Entry leaderboard — best remaining paths */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center gap-3 flex-wrap">
          <p className="text-white font-semibold text-sm">Best Remaining Paths — Entry Level</p>
          <p className="text-gray-500 text-xs">Ranked by optimal EV path through remaining available teams</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                {['#', 'Entry', 'Contestant', 'Wins', 'Teams Left', 'Pool Strength', 'Best EV Path', 'Best Win% Path', 'Remaining Teams'].map(h => (
                  <th key={h} className="text-left px-3 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data.entry_stats || []).slice(0, 100).map((e, i) => (
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
                  <td className="px-3 py-2.5 text-green-400 text-xs font-mono font-semibold">
                    {e.optimal_ev_path?.toFixed(4)}
                  </td>
                  <td className="px-3 py-2.5 text-blue-400 text-xs font-mono">
                    {e.optimal_win_path?.toFixed(1)}%
                  </td>
                  <td className="px-3 py-2.5 text-xs text-gray-500 max-w-xs truncate">
                    {(e.remaining_teams || []).join(', ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Contestant leaderboard */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center gap-3 flex-wrap">
          <p className="text-white font-semibold text-sm">Contestant Leaderboard</p>
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
                {['#', 'Contestant', 'Entries', 'Surviving', 'Max Wins', 'Avg Wins', 'Best EV Path', 'Best Win% Path', 'Avg Teams Left', 'Avg Pool Strength', 'Exp. Value'].map(h => (
                  <th key={h} className="text-left px-3 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 100).map((c, i) => {
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
                      {expValue > 0 ? `$${expValue.toFixed(0)}` : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-2.5 border-t border-gray-800 text-xs text-gray-500">
          Showing {Math.min(100, filtered.length)} of {filtered.length} contestants
        </div>
      </div>

    </div>
  )
}
