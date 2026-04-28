export default function PickCard({ solution, index, label }) {
  const totalEV = solution.reduce((sum, p) => sum + p.ev, 0)
  const avgWin = solution.reduce((sum, p) => sum + p.win_pct, 0) / solution.length

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      {/* Card header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="flex items-center gap-2">
          {index === 0 && (
            <span className="text-yellow-400 text-xs">★</span>
          )}
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
              <th className="text-left px-4 py-2 font-medium">Team</th>
              <th className="text-left px-4 py-2 font-medium">vs</th>
              <th className="text-left px-4 py-2 font-medium">H/A</th>
              <th className="text-right px-4 py-2 font-medium">Spread</th>
              <th className="text-right px-4 py-2 font-medium">Win%</th>
              <th className="text-right px-4 py-2 font-medium">EV</th>
              <th className="text-right px-4 py-2 font-medium">Pick%</th>
            </tr>
          </thead>
          <tbody>
            {solution.map((pick, i) => (
              <tr
                key={i}
                className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
              >
                <td className="px-4 py-2.5 text-gray-400 font-mono text-xs">
                  {pick.week}
                </td>
                <td className="px-4 py-2.5 font-semibold text-white">
                  {pick.team}
                </td>
                <td className="px-4 py-2.5 text-gray-400 text-xs">
                  {pick.opponent}
                </td>
                <td className="px-4 py-2.5">
                  <span className={`
                    text-xs font-medium px-1.5 py-0.5 rounded
                    ${pick.home_or_away === 'Home'
                      ? 'bg-blue-900/50 text-blue-300'
                      : 'bg-gray-800 text-gray-400'}
                  `}>
                    {pick.home_or_away}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-300">
                  {pick.spread > 0 ? '+' : ''}{pick.spread?.toFixed(1) ?? '—'}
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs">
                  <span className={
                    pick.win_pct >= 0.7 ? 'text-green-400' :
                    pick.win_pct >= 0.6 ? 'text-yellow-400' : 'text-gray-300'
                  }>
                    {(pick.win_pct * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-green-400">
                  {pick.ev.toFixed(4)}
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-400">
                  {(pick.pick_pct * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
