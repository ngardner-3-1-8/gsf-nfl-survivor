import { useState } from 'react'

function weatherEmoji(temp, precip, wind, dome) {
  if (dome) return '🏟️'
  if (temp === null || temp === undefined) return ''
  const t = parseFloat(temp)
  const p = parseFloat(precip) || 0
  const w = parseFloat(wind) || 0

  if (p > 0.3 && t < 32) return '❄️🌨️'
  if (p > 0.3 && t < 40) return '🥶🌧️'
  if (p > 0.3)            return '🌧️'
  if (p > 0.1 && t < 32) return '🌨️'
  if (p > 0.1 && t < 40) return '🥶🌦️'
  if (p > 0.1)            return '🌦️'
  if (t <= 20)            return '🥶🧊⛄'
  if (t <= 30 && w > 15) return '🥶💨'
  if (t <= 32)            return '🥶⛅'
  if (t <= 50 && w > 15) return '🧥💨'
  if (t <= 50)            return '🧥'
  if (t <= 70 && w > 20) return '⛅💨'
  if (t >= 70 && w > 15) return '☀️💨'
  if (t >= 90)            return '🥵'
  if (t >= 65)            return '☀️'
  if (w > 15)             return '💨'
  return '⛅'
}

function holidayEmoji(pick) {
  const turkey = pick.is_thanksgiving ? '🦃' : ''
  const tree = pick.is_christmas ? '🎄' : ''
  return turkey + tree
}

function pickPctColor(pickPct, weekPcts) {
  if (!weekPcts || weekPcts.length === 0) return 'text-gray-400'
  const rank = weekPcts.findIndex(p => p <= pickPct) + 1
  if (rank <= 2) return 'text-red-400'
  if (rank <= 5) return 'text-yellow-400'
  return 'text-green-400'
}

export default function PickCard({ solution, index, label, allWeeklyPickPcts }) {
  const [isOpen, setIsOpen] = useState(index === 0) // first card open by default

  const totalEV = solution.reduce((sum, p) => sum + p.ev, 0)
  const avgWin = solution.reduce((sum, p) => sum + p.win_pct, 0) / solution.length

  // Summary of top 3 picks for the collapsed preview
  const previewPicks = solution.slice(0, 3)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">

      {/* Header — always visible, click to toggle */}
      <button
        onClick={() => setIsOpen(prev => !prev)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-800/50 transition-colors text-left"
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {index === 0 && <span className="text-yellow-400 text-xs flex-shrink-0">★</span>}
          <span className="text-sm font-semibold text-white flex-shrink-0">
            {label} — Solution {index + 1}
          </span>

          {/* Collapsed preview — show first 3 teams */}
          {!isOpen && (
            <div className="flex items-center gap-1.5 ml-3 overflow-hidden">
              {previewPicks.map((pick, i) => (
                <span key={i} className="flex items-center gap-1">
                  <span className="text-xs text-gray-400 font-mono flex-shrink-0">
                    {pick.circa_week || `Wk${pick.week}`}
                  </span>
                  <span className="text-xs text-white font-medium truncate flex-shrink-0">
                    {pick.team.split(' ').pop()}
                  </span>
                  {i < previewPicks.length - 1 && (
                    <span className="text-gray-700 text-xs">·</span>
                  )}
                </span>
              ))}
              {solution.length > 3 && (
                <span className="text-xs text-gray-600 flex-shrink-0">
                  +{solution.length - 3} more
                </span>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-4 flex-shrink-0 ml-4">
          <div className="flex items-center gap-4 text-xs text-gray-400">
            <span>
              EV: <span className="text-green-400 font-medium">{totalEV.toFixed(3)}</span>
            </span>
            <span>
              Win%: <span className="text-blue-400 font-medium">{(avgWin * 100).toFixed(1)}%</span>
            </span>
          </div>
          {/* Chevron */}
          <span className={`text-gray-500 transition-transform duration-200 text-xs ${isOpen ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </div>
      </button>

      {/* Accordion body — full table */}
      {isOpen && (
        <div className="border-t border-gray-800 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-800">
                <th className="text-left px-4 py-2 font-medium">Wk</th>
                <th className="text-left px-4 py-2 font-medium">🗓</th>
                <th className="text-left px-4 py-2 font-medium">Day</th>
                <th className="text-left px-4 py-2 font-medium">Team</th>
                <th className="text-left px-4 py-2 font-medium">QB</th>
                <th className="text-left px-4 py-2 font-medium">vs</th>
                <th className="text-left px-4 py-2 font-medium">H/A</th>
                <th className="text-left px-4 py-2 font-medium">Stadium</th>
                <th className="text-left px-4 py-2 font-medium">🌤</th>
                <th className="text-right px-4 py-2 font-medium">Rest</th>
                <th className="text-right px-4 py-2 font-medium">Rest+/-</th>
                <th className="text-right px-4 py-2 font-medium">Cum Rest</th>
                <th className="text-right px-4 py-2 font-medium">Win%</th>
                <th className="text-right px-4 py-2 font-medium">EV</th>
                <th className="text-right px-4 py-2 font-medium">Pick%</th>
              </tr>
            </thead>
            <tbody>
              {solution.map((pick, i) => {
                const isClose = pick.win_pct < 0.65
                const weekPcts = allWeeklyPickPcts?.[pick.week] || []
                const rank = weekPcts.findIndex(p => p <= pick.pick_pct) + 1
                const weather = weatherEmoji(pick.temperature, pick.precipitation, pick.wind, pick.dome)
                const holiday = holidayEmoji(pick)

                return (
                  <tr
                    key={i}
                    className={`
                      border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors
                      ${isClose ? 'bg-yellow-950/10' : ''}
                    `}
                  >
                    <td className="px-4 py-2.5 text-gray-400 font-mono text-xs">
                      {pick.circa_week || pick.week}
                    </td>
                    <td className="px-4 py-2.5 text-base">{holiday}</td>
                    <td className="px-4 py-2.5 text-gray-400 text-xs">{pick.day || '—'}</td>
                    <td className="px-4 py-2.5 font-semibold text-white">{pick.team}</td>
                    <td className="px-4 py-2.5 text-gray-300 text-xs">{pick.starting_qb || '—'}</td>
                    <td className="px-4 py-2.5 text-gray-400 text-xs">{pick.opponent}</td>
                    <td className="px-4 py-2.5">
                      <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                        pick.home_or_away === 'Home'
                          ? 'bg-blue-900/50 text-blue-300'
                          : 'bg-gray-800 text-gray-400'
                      }`}>
                        {pick.home_or_away}
                      </span>
                    </td>
                    <td className={`px-4 py-2.5 text-xs ${
                      pick.is_international ? 'text-red-400 font-medium' : 'text-white'
                    }`}>
                      {pick.stadium || '—'}
                    </td>
                    <td className="px-4 py-2.5 text-base">{weather}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-300">
                      {pick.weekly_rest != null ? pick.weekly_rest : '—'}
                    </td>
                    <td className={`px-4 py-2.5 text-right font-mono text-xs ${
                      pick.rest_advantage > 0 ? 'text-green-400' :
                      pick.rest_advantage < 0 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {pick.rest_advantage != null
                        ? (pick.rest_advantage > 0 ? '+' : '') + pick.rest_advantage
                        : '—'}
                    </td>
                    <td className={`px-4 py-2.5 text-right font-mono text-xs ${
                      pick.cumulative_rest_advantage > 0 ? 'text-green-400' :
                      pick.cumulative_rest_advantage < 0 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {pick.cumulative_rest_advantage != null
                        ? (pick.cumulative_rest_advantage > 0 ? '+' : '') + pick.cumulative_rest_advantage
                        : '—'}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-xs">
                      <span className={`${
                        pick.win_pct >= 0.75 ? 'text-green-400' :
                        pick.win_pct >= 0.65 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {(pick.win_pct * 100).toFixed(1)}%
                        {isClose && <span className="ml-1 text-yellow-500" title="Close game">⚠️</span>}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-xs text-green-400">
                      {pick.ev.toFixed(4)}
                    </td>
                    <td className={`px-4 py-2.5 text-right font-mono text-xs ${pickPctColor(pick.pick_pct, weekPcts)}`}>
                      {(pick.pick_pct * 100).toFixed(1)}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
