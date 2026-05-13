import { useState } from 'react'

// ── Column definitions per view ──────────────────────────────
const COLUMNS = {
  'Overview': [
    { key: 'Week_x', label: 'Wk', render: (v, r) => r['Circa Week'] || v },
    { key: 'Date_x', label: 'Date', render: v => v ? new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '—' },
    { key: '_day', label: 'Day', render: (_, r) => dayLabel(r) },
    { key: 'Away Team', label: 'Away Team' },
    { key: 'Away QB', label: 'Away QB', render: v => v || '—' },
    { key: 'Home Team', label: 'Home Team' },
    { key: 'Home QB', label: 'Home QB', render: v => v || '—' },
    { key: 'Actual Stadium', label: 'Stadium', render: (v, r) => v || '—', className: (_, r) => r['International Game'] ? 'text-red-400 font-medium' : 'text-gray-300' },
    { key: '_weather', label: '🌤', render: (_, r) => weatherEmoji(r) },
    { key: '_holiday', label: '🗓', render: (_, r) => holidayEmoji(r) },
  ],
'Odds & Win%': [
  { key: 'Week_x', label: 'Wk', render: (v, r) => r['Circa Week'] || v },
  { key: 'Away Team', label: 'Away Team' },
  { key: 'Home Team', label: 'Home Team' },
  { key: 'Consensus Away Win Pct', label: 'Away Consensus', render: v => winPctCell(v) },
  { key: 'Consensus Home Win Pct', label: 'Home Consensus', render: v => winPctCell(v) },
  { key: 'Sim_Away_Win_Pct', label: 'Away MC', render: v => winPctCell(v) },
  { key: 'Sim_Home_Win_Pct', label: 'Home MC', render: v => winPctCell(v) },
  { key: 'Away Team Sportsbook Fair Odds', label: 'Away SB Odds', render: v => winPctCell(v) },
  { key: 'Home Team Sportsbook Fair Odds', label: 'Home SB Odds', render: v => winPctCell(v) },
  { key: 'Away Team Massey-Peabody Fair Odds', label: 'Away MP Odds', render: v => winPctCell(v) },
  { key: 'Home Team Massey-Peabody Fair Odds', label: 'Home MP Odds', render: v => winPctCell(v) },
  { key: 'Away Team Sportsbook Spread', label: 'Away SB Spread', render: v => spreadCell(v) },
  { key: 'Home Team Sportsbook Spread', label: 'Home SB Spread', render: v => spreadCell(v) },
  { key: 'Generic Sports Fan Away Team Spread', label: 'Away GSF Spread', render: v => spreadCell(v) },
  { key: 'Generic Sports Fan Home Team Spread', label: 'Home GSF Spread', render: v => spreadCell(v) },
  { key: 'Massey-Peabody Away Team Spread', label: 'Away MP Spread', render: v => spreadCell(v) },
  { key: 'Massey-Peabody Home Team Spread', label: 'Home MP Spread', render: v => spreadCell(v) },
  { key: 'Sim_Spread_Mean', label: 'Home Sim Spread Mean', render: v => spreadCell(v) },
  { key: 'Sim_Spread_Median', label: 'Home Sim Spread Median', render: v => spreadCell(v) },
  { key: 'Sim_Spread_Mean', label: 'Away Sim Spread Mean', render: v => spreadCell(v) },
  { key: 'Sim_Spread_Median', label: 'Away Sim Spread Median', render: v => spreadCell(v) },
  { key: 'Sim_Spread_Variance_Label', label: 'Sim Spread Variance', render: v => spreadCell(v) },
  { key: 'Sim_Total_Mean', label: 'Sim Total Mean', render: v => spreadCell(v) },
  { key: 'Sim_Total_Median', label: 'Sim Total Median', render: v => spreadCell(v) },
],
  'Situational': [
    { key: 'Week_x', label: 'Wk', render: (v, r) => r['Circa Week'] || v },
    { key: 'Away Team', label: 'Away Team' },
    { key: 'Home Team', label: 'Home Team' },
    { key: 'Away Team Weekly Rest', label: 'Away Rest', render: v => v ?? '—' },
    { key: 'Home Team Weekly Rest', label: 'Home Rest', render: v => v ?? '—' },
    { key: 'Away Team Current Week Cumulative Rest Advantage', label: 'Away Cum Rest', render: v => restCell(v) },
    { key: 'Home Team Current Week Cumulative Rest Advantage', label: 'Home Cum Rest', render: v => restCell(v) },
    { key: 'Divisional Matchup?', label: 'Div?', render: v => v === 1 || v === 'Divisional' || v === true ? '✓' : '—' },
    { key: 'Thursday Night Game', label: 'TNF', render: v => String(v).toLowerCase() === 'true' ? 'TNF' : '—' },
    { key: 'Back to Back Away Games', label: 'B2B Away', render: v => String(v).toLowerCase() === 'true' ? '⚠️' : '—' },
    { key: 'Away Team 3 games in 10 days', label: '3in10', render: v => String(v).toLowerCase() === 'yes' ? '⚠️' : '—' },
    { key: 'Away Team 4 games in 17 days', label: '4in17', render: v => String(v).toLowerCase() === 'yes' ? '⚠️' : '—' },
    { key: 'International Game', label: 'Intl?', render: v => v ? '🌍' : '—' },
  ],
  'Contest': [
    { key: 'Week_x', label: 'Wk', render: (v, r) => r['Circa Week'] || v },
    { key: 'Away Team', label: 'Away Team' },
    { key: 'Home Team', label: 'Home Team' },
    { key: 'Away Pick %', label: 'Away Pick%', render: (v, r, allRows) => pickPctCell(v, 'Away Team', r, allRows) },
    { key: 'Home Pick %', label: 'Home Pick%', render: (v, r, allRows) => pickPctCell(v, 'Home Team', r, allRows) },
    { key: 'Away Team EV', label: 'Away EV', render: v => v != null ? Number(v).toFixed(4) : '—' },
    { key: 'Home Team EV', label: 'Home EV', render: v => v != null ? Number(v).toFixed(4) : '—' },
    { key: 'Away Team Star Rating', label: 'Away Future Val', render: v => v ?? '—' },
    { key: 'Home Team Star Rating', label: 'Home Future Val', render: v => v ?? '—' },
    { key: 'Total Remaining Entries at Start of Week', label: 'Entries', render: v => v ? Number(v).toLocaleString() : '—' },
    { key: 'Home Team Expected Availability', label: 'Home Avail', render: v => v != null ? (Number(v) * 100).toFixed(1) + '%' : '—' },
    { key: 'Away Team Expected Availability', label: 'Away Avail', render: v => v != null ? (Number(v) * 100).toFixed(1) + '%' : '—' },
  ],
  'Betting': [
    { key: 'Week_x', label: 'Wk', render: (v, r) => r['Circa Week'] || v },
    { key: 'Away Team', label: 'Away Team' },
    { key: 'Home Team', label: 'Home Team' },
    // Spread bets
    { key: 'GSF Spread Bet', label: 'GSF Spread', render: v => v || '—' },
    { key: 'GSF Spread Edge', label: 'GSF Spread Edge', render: v => edgeCell(v) },
    { key: 'Massey-Peabody Spread Bet', label: 'MP Spread', render: v => v || '—' },
    { key: 'Massey-Peabody Spread Edge', label: 'MP Spread Edge', render: v => edgeCell(v) },
    { key: 'Monte Carlo Spread Bet', label: 'MC Spread', render: v => v || '—' },
    { key: 'Monte Carlo Spread Edge', label: 'MC Spread Edge', render: v => edgeCell(v) },
    { key: 'Consensus Spread Bet', label: 'Con Spread', render: v => v || '—' },
    { key: 'Consensus Spread Edge', label: 'Con Spread Edge', render: v => edgeCell(v) },
    // Moneyline bets — edges are percentages
    { key: 'GSF Moneyline Bet', label: 'GSF ML', render: v => v || '—' },
    { key: 'GSF Moneyline Edge', label: 'GSF ML Edge', render: v => edgePctCell(v) },
    { key: 'Massey-Peabody Moneyline Bet', label: 'MP ML', render: v => v || '—' },
    { key: 'Massey-Peabody Moneyline Edge', label: 'MP ML Edge', render: v => edgePctCell(v) },
    { key: 'Monte Carlo Moneyline Bet', label: 'MC ML', render: v => v || '—' },
    { key: 'Monte Carlo Moneyline Edge', label: 'MC ML Edge', render: v => edgePctCell(v) },
    { key: 'Consensus Moneyline Bet', label: 'Con ML', render: v => v || '—' },
    { key: 'Consensus Moneyline Edge', label: 'Con ML Edge', render: v => edgePctCell(v) },
    // Total
    { key: 'Monte Carlo Total Bet', label: 'MC Total', render: v => v || '—' },
    { key: 'Monte Carlo Total Edge', label: 'MC Total Edge', render: v => edgeCell(v) },
    { key: 'MC Bet Direction', label: 'MC Direction', render: v => v || '—' },
  ],
  'Bayesian': [
    { key: 'Week_x', label: 'Wk', render: (v, r) => r['Circa Week'] || v },
    { key: 'Away Team', label: 'Away Team' },
    { key: 'Home Team', label: 'Home Team' },
    { key: 'Massey-Peabody Bayesian Same Winner Across All Metrics', label: 'MP All Metrics', render: v => bayesianCell(v) },
    { key: 'Generic Sports Fan Bayesian Same Adjusted Winner Across All Metrics', label: 'GSF All Metrics', render: v => bayesianCell(v) },
    { key: 'Massey-Peabody Bayesian Same Current and Preseason Adjusted Winner', label: 'MP Pre + Current', render: v => bayesianCell(v) },
    { key: 'Generic Sports Fan Bayesian Current and Preseason Adjusted Winner', label: 'GSF Pre + Current', render: v => bayesianCell(v) },
    { key: 'Massey-Peabody Bayesian Same Current and Adjusted Current Winner', label: 'MP Current + Adj', render: v => bayesianCell(v) },
    { key: 'Generic Sports Fan Bayesian Same Current and Adjusted Current Winner', label: 'GSF Current + Adj', render: v => bayesianCell(v) },
    { key: 'Sportsbook Bayesian Same Current and Preseason Adjusted Winner', label: 'SB Pre + Current', render: v => bayesianCell(v) },
    { key: 'Sim Bayesian Same Current and Preseason Adjusted Winner', label: 'Sim Pre + Current', render: v => bayesianCell(v) },
    { key: 'Consensus Bayesian Same Current and Preseason Adjusted Winner', label: 'Con Pre + Current', render: v => bayesianCell(v) },
  ],
}

// ── Helper renderers ──────────────────────────────────────────

function dayLabel(row) {
  try {
    const date = new Date(row['Date_x'])
    const day = date.toLocaleDateString('en-US', { weekday: 'long' })
    const time = String(row['Time'] || '')
    const isTNF = String(row['Thursday Night Game'] || '').toLowerCase() === 'true'
    const isMNF = day === 'Monday' && time >= '19:00'
    const isSNF = day === 'Sunday' && time >= '19:00'
    if (isTNF) return 'Thu 🌙'
    if (isMNF) return 'Mon 🌙'
    if (isSNF) return 'Sun 🌙'
    return day.slice(0, 3)
  } catch { return '—' }
}

function weatherEmoji(row) {
  if (row['Dome']) return '🏟️'
  const t = parseFloat(row['Temperature'])
  const p = parseFloat(row['Precipitation']) || 0
  const w = parseFloat(row['Wind']) || 0
  if (isNaN(t)) return ''
  if (p > 0.3 && t < 32) return '❄️🌨️'
  if (p > 0.3 && t < 40) return '🥶🌧️'
  if (p > 0.3) return '🌧️'
  if (p > 0.1 && t < 32) return '🌨️'
  if (p > 0.1 && t < 40) return '🥶🌦️'
  if (p > 0.1) return '🌦️'
  if (t <= 20) return '🥶🧊⛄'
  if (t <= 30 && w > 15) return '🥶💨'
  if (t <= 32) return '🥶⛅'
  if (t <= 50 && w > 15) return '🧥💨'
  if (t <= 50) return '🧥'
  if (t <= 70 && w > 20) return '⛅💨'
  if (t >= 70 && w > 15) return '☀️💨'
  if (t >= 90) return '🥵'
  if (t >= 65) return '☀️'
  if (w > 15) return '💨'
  return '⛅'
}

function holidayEmoji(row) {
  const isThanksgiving =
    row['Away Team Thanksgiving Favorite'] || row['Home Team Thanksgiving Favorite'] ||
    row['Away Team Thanksgiving Underdog'] || row['Home Team Thanksgiving Underdog']
  const isChristmas =
    row['Away Team Christmas Favorite'] || row['Home Team Christmas Favorite'] ||
    row['Away Team Christmas Underdog'] || row['Home Team Christmas Underdog']
  return (isThanksgiving ? '🦃' : '') + (isChristmas ? '🎄' : '')
}

function winPctCell(val) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  const pct = (v * 100).toFixed(1) + '%'
  if (v >= 0.70) return <span className="text-green-400 font-mono text-xs">{pct}</span>
  if (v >= 0.55) return <span className="text-yellow-400 font-mono text-xs">{pct}</span>
  return <span className="text-red-400 font-mono text-xs">{pct}</span>
}

function spreadCell(val) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  const formatted = (v > 0 ? '+' : '') + v.toFixed(1)
  if (v <= -7) return <span className="text-green-400 font-mono text-xs">{formatted}</span>
  if (v <= -3) return <span className="text-yellow-400 font-mono text-xs">{formatted}</span>
  return <span className="text-gray-400 font-mono text-xs">{formatted}</span>
}

function restCell(val) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  const formatted = (v > 0 ? '+' : '') + v.toFixed(0)
  if (v > 0) return <span className="text-green-400 font-mono text-xs">{formatted}</span>
  if (v < 0) return <span className="text-red-400 font-mono text-xs">{formatted}</span>
  return <span className="text-gray-400 font-mono text-xs">0</span>
}

function edgeCell(val) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  const formatted = (v > 0 ? '+' : '') + v.toFixed(2)
  if (v > 0) return <span className="text-green-400 font-mono text-xs">{formatted}</span>
  if (v < 0) return <span className="text-red-400 font-mono text-xs">{formatted}</span>
  return <span className="text-gray-400 font-mono text-xs">0</span>
}

function edgePctCell(val) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  const formatted = (v > 0 ? '+' : '') + (v * 100).toFixed(1) + '%'
  if (v > 0) return <span className="text-green-400 font-mono text-xs">{formatted}</span>
  if (v < 0) return <span className="text-red-400 font-mono text-xs">{formatted}</span>
  return <span className="text-gray-400 font-mono text-xs">0.0%</span>
}

function bayesianCell(val) {
  if (val == null || val === '') return <span className="text-gray-600">—</span>
  const v = String(val).trim().toLowerCase()
  if (v === 'same') return (
    <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-green-900/50 text-green-400">
      Same
    </span>
  )
  if (v === 'different') return (
    <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-gray-800 text-gray-500">
      Diff
    </span>
  )
  return <span className="text-gray-600 text-xs">{val}</span>
}

function pickPctCell(val, teamKey, row, allRows) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  // Get all pick pcts for this week to determine rank
  const week = row['Week_x'] ?? row['Week']
  const weekRows = allRows.filter(r => (r['Week_x'] ?? r['Week']) === week)
  const allPcts = weekRows.flatMap(r => [
    parseFloat(r['Away Pick %'] || 0),
    parseFloat(r['Home Pick %'] || 0),
  ]).filter(p => !isNaN(p)).sort((a, b) => b - a)
  const rank = allPcts.findIndex(p => p <= v) + 1
  const pct = (v * 100).toFixed(1) + '%'
  if (rank <= 2) return <span className="text-red-400 font-mono text-xs">{pct}</span>
  if (rank <= 5) return <span className="text-yellow-400 font-mono text-xs">{pct}</span>
  return <span className="text-green-400 font-mono text-xs">{pct}</span>
}

// ── Row highlight logic ───────────────────────────────────────
function rowClass(row) {
  const isTNF = String(row['Thursday Night Game'] || '').toLowerCase() === 'true'
  const isIntl = row['International Game']
  const isThanks = row['Away Team Thanksgiving Favorite'] || row['Home Team Thanksgiving Favorite'] ||
    row['Away Team Thanksgiving Underdog'] || row['Home Team Thanksgiving Underdog']
  const isXmas = row['Away Team Christmas Favorite'] || row['Home Team Christmas Favorite'] ||
    row['Away Team Christmas Underdog'] || row['Home Team Christmas Underdog']
  if (isIntl) return 'bg-red-950/20 hover:bg-red-950/30'
  if (isThanks || isXmas) return 'bg-yellow-950/20 hover:bg-yellow-950/30'
  if (isTNF) return 'bg-purple-950/20 hover:bg-purple-950/30'
  return 'hover:bg-gray-800/30'
}

// ── Sort state ────────────────────────────────────────────────
export default function ScheduleTable({ games, activeView }) {
  const [sortKey, setSortKey] = useState(null)
  const [sortDir, setSortDir] = useState('asc')

  const columns = COLUMNS[activeView] || COLUMNS['Overview']

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const sorted = [...games].sort((a, b) => {
    if (!sortKey) return 0
    const av = a[sortKey] ?? ''
    const bv = b[sortKey] ?? ''
    const an = parseFloat(av)
    const bn = parseFloat(bv)
    if (!isNaN(an) && !isNaN(bn)) return sortDir === 'asc' ? an - bn : bn - an
    return sortDir === 'asc'
      ? String(av).localeCompare(String(bv))
      : String(bv).localeCompare(String(av))
  })

  if (sorted.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center h-40">
        <p className="text-gray-500 text-sm">No games match your filters</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              {columns.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 cursor-pointer hover:text-white whitespace-nowrap select-none"
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span className="ml-1 text-green-400">
                      {sortDir === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, i) => (
              <tr
                key={i}
                className={`border-b border-gray-800/50 transition-colors ${rowClass(row)}`}
              >
                {columns.map(col => {
                  const raw = row[col.key]
                  const rendered = col.render
                    ? col.render(raw, row, sorted)
                    : raw ?? '—'
                  const cls = col.className
                    ? col.className(raw, row)
                    : 'text-gray-300'
                  return (
                    <td key={col.key} className={`px-4 py-2.5 text-xs whitespace-nowrap ${cls}`}>
                      {rendered}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="px-4 py-2.5 border-t border-gray-800 flex items-center gap-4 text-xs text-gray-500 flex-wrap">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-sm bg-yellow-900/50 inline-block" />
          Holiday game
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-sm bg-red-900/50 inline-block" />
          International game
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-sm bg-purple-900/50 inline-block" />
          Thursday Night Football
        </span>
        <span className="ml-auto">{sorted.length} games</span>
      </div>
    </div>
  )
}
