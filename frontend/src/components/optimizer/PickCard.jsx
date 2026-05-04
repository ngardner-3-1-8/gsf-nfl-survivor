// Helper: weather emoji from temperature, precipitation, wind
function weatherEmoji(temp, precip, wind) {
  if (dome) return '🏟️'
  if (temp === null || temp === undefined) return ''
  const t = parseFloat(temp)
  const p = parseFloat(precip) || 0
  const w = parseFloat(wind) || 0


  // Precipitation first — check cold vs warm variants
  if (p > 0.3 && t < 32) return '❄️🌨️'   // Heavy snow
  if (p > 0.3 && t < 40) return '🥶🌧️'   // Heavy rain, very cold
  if (p > 0.3)            return '🌧️'      // Heavy rain
  if (p > 0.1 && t < 32) return '🌨️'      // Light snow
  if (p > 0.1 && t < 40) return '🥶🌦️'   // Light rain, very cold
  if (p > 0.1)            return '🌦️'      // Light rain

  // No precipitation — check temperature + wind
  // Temperature checks must go from coldest to hottest
  if (t <= 20)            return '🥶🧊⛄'  // Extreme cold
  if (t <= 30 && w > 15) return '🥶💨'    // Freezing + windy
  if (t <= 32)            return '🥶⛅'    // Freezing
  if (t <= 50 && w > 15) return '🧥💨'    // Cold + windy
  if (t <= 50)            return '🧥'      // Cold
  if (t <= 70 && w > 15) return '⛅💨'    // Mild + very windy
  if (t >= 70 && w > 15) return '☀️💨'    // Warm + breezy
  if (t >= 90)            return '🥵'      // Very hot — must come before ≥ 65
  if (t >= 65)            return '☀️'      // Warm/sunny
  if (w > 15)             return '💨'      // Mild but windy
  return '⛅'                              // Default: partly cloudy
}

function holidayEmoji(pick) {
  const turkey = pick.is_thanksgiving ? '🦃' : ''
  const tree = pick.is_christmas ? '🎄' : ''
  return turkey + tree  // shows one, both, or neither
}

// Helper: pick% color — top 2 red, next 3 yellow, rest green
// allPickPcts is a sorted array of all pick percentages for that week
function pickPctColor(pct, rank) {
  if (rank <= 2) return 'text-red-400'
  if (rank <= 5) return 'text-yellow-400'
  return 'text-green-400'
}

export default function PickCard({ solution, index, label, allWeeklyPickPcts }) {
  const totalEV = solution.reduce((sum, p) => sum + p.ev, 0)
  const avgWin = solution.reduce((sum, p) => sum + p.win_pct, 0) / solution.length

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      {/* Card header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="flex items-center gap-2">
          {index === 0 && <span className="text-yellow-400 text-xs">★</span>}
          <span className="text-sm font-semibold text-white">
            {label} — Solution {index + 1}
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs text-gray-400">
          <span>Total EV: <span className="text-green-400 font-medium">{totalEV.toFixed(3)}</span></span>
          <span>Avg Win%: <span className="text-blue-400 font-medium">{(avgWin * 100).toFixed(1)}%</span></span>
        </div>
      </div>

      {/* Picks table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-800">
              <th className="text-left px-4 py-2 font-medium">Wk</th>
              <th className="text-left px-4 py-2 font-medium">🗓</th>
              <th className="text-left px-4 py-2 font-medium">Day</th>
              <th className="text-left px-4 py-2 font-medium">Team</th>
              <th className="text-left px-4 py-2 font-medium">Starting QB</th>
              <th className="text-left px-4 py-2 font-medium">vs</th>
              <th className="text-left px-4 py-2 font-medium">H/A</th>
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
              // Rank among all teams that week (1 = highest pick%)
              const rank = weekPcts.filter(p => p > pick.pick_pct).length + 1
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
                  <td className="px-4 py-2.5 text-gray-400 font-mono text-xs">{pick.circa_week || pick.week}</td>
                  <td className="px-4 py-2.5 text-base">{holidayEmoji(pick)}</td>
                  <td className="px-4 py-2.5 text-gray-400 text-xs">{pick.day_of_week || '—'}</td>
                  <td className="px-4 py-2.5 font-semibold text-white">{pick.team}</td>
                  <td className="px-4 py-2.5 text-gray-300 text-xs">
                    {pick.starting_qb || '—'}
                  </td>
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
                  <td className="px-4 py-2.5 text-base">{weather}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-300">
                    {pick.days_of_rest ?? '—'}
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
                    pick.cumulative_rest > 0 ? 'text-green-400' :
                    pick.cumulative_rest < 0 ? 'text-red-400' : 'text-gray-400'
                  }`}>
                    {pick.cumulative_rest != null
                      ? (pick.cumulative_rest > 0 ? '+' : '') + pick.cumulative_rest
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
                  <td className={`px-4 py-2.5 text-right font-mono text-xs ${pickPctColor(pick.pick_pct, rank)}`}>
                    {(pick.pick_pct * 100).toFixed(1)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
