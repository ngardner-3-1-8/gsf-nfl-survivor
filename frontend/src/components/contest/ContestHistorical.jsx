import { useState, useEffect } from 'react'
import { fetchContestData } from '../../api/client'

const COLORS = [
  '#16a34a', '#2563eb', '#9333ea', '#ea580c',
  '#0891b2', '#dc2626', '#ca8a04',
]

// Simple SVG line chart
function SurvivalChart({ datasets, width = 600, height = 280 }) {
  if (!datasets || datasets.length === 0) return null
  const allPoints = datasets.flatMap(d => d.data)
  const maxWeek = Math.max(...allPoints.map(p => p.week))
  const pad = { top: 20, right: 20, bottom: 40, left: 45 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom

  const xScale = week => (week - 1) / (maxWeek - 1) * w
  const yScale = pct => h - (pct / 100) * h

  const weekTicks = Array.from({ length: maxWeek }, (_, i) => i + 1).filter(w => w % 2 === 0 || w === 1 || w === maxWeek)

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
      <g transform={`translate(${pad.left},${pad.top})`}>
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map(y => (
          <g key={y}>
            <line x1={0} y1={yScale(y)} x2={w} y2={yScale(y)} stroke="#374151" strokeWidth={0.5} />
            <text x={-6} y={yScale(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}%</text>
          </g>
        ))}
        {/* X ticks */}
        {weekTicks.map(wk => (
          <text key={wk} x={xScale(wk)} y={h + 16} textAnchor="middle" fontSize={9} fill="#6b7280">
            {wk}
          </text>
        ))}
        <text x={w / 2} y={h + 32} textAnchor="middle" fontSize={10} fill="#6b7280">Week</text>
        {/* Lines */}
        {datasets.map((ds, di) => {
          const pts = ds.data.map(p => `${xScale(p.week)},${yScale(p.pct_remaining)}`).join(' ')
          return (
            <polyline
              key={ds.year}
              points={pts}
              fill="none"
              stroke={COLORS[di % COLORS.length]}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )
        })}
        {/* Legend */}
        {datasets.map((ds, di) => (
          <g key={`leg-${ds.year}`} transform={`translate(${w - (datasets.length - di) * 48}, ${-16})`}>
            <line x1={0} y1={0} x2={16} y2={0} stroke={COLORS[di % COLORS.length]} strokeWidth={2} />
            <text x={20} y={4} fontSize={9} fill={COLORS[di % COLORS.length]}>{ds.year}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}

// Elimination heatmap
function EliminationHeatmap({ allData }) {
  const years = Object.keys(allData).sort()
  const maxWeeks = Math.max(...Object.values(allData).map(d => d.survival_curve.length))
  const weeks = Array.from({ length: maxWeeks }, (_, i) => i + 1)

  const getCell = (year, week) => {
    const curve = allData[year]?.survival_curve || []
    return curve.find(c => c.week === week)
  }

  const heatColor = (pct) => {
    if (pct >= 30) return 'bg-red-900 text-red-200'
    if (pct >= 15) return 'bg-orange-900 text-orange-200'
    if (pct >= 5)  return 'bg-yellow-900 text-yellow-200'
    if (pct >= 1)  return 'bg-gray-700 text-gray-300'
    return 'bg-gray-900 text-gray-600'
  }

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="text-left px-3 py-1.5 text-gray-500 font-medium sticky left-0 bg-gray-900">Year</th>
            {weeks.map(w => (
              <th key={w} className="px-2 py-1.5 text-gray-500 font-medium text-center min-w-[36px]">
                {w}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {years.map(year => (
            <tr key={year}>
              <td className="px-3 py-1 text-gray-300 font-medium sticky left-0 bg-gray-900">{year}</td>
              {weeks.map(week => {
                const cell = getCell(year, week)
                const pct = cell?.pct_eliminated || 0
                return (
                  <td
                    key={week}
                    title={`${year} Wk${week}: ${pct.toFixed(1)}% eliminated`}
                    className={`px-1 py-1 text-center rounded-sm m-0.5 ${heatColor(pct)}`}
                  >
                    {pct >= 1 ? pct.toFixed(0) : ''}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-600 mt-2">Values show % of entries eliminated that week. Darker = more carnage.</p>
    </div>
  )
}

export default function ContestHistorical({ years }) {
  const [allData, setAllData] = useState({})
  const [loading, setLoading] = useState(false)
  const [selectedYear, setSelectedYear] = useState(null)

  useEffect(() => {
    if (years.length === 0) return
    setLoading(true)
    setSelectedYear(years[0])
    Promise.all(years.map(y => fetchContestData(y).then(d => [y, d])))
      .then(results => {
        const map = {}
        results.forEach(([y, d]) => { map[y] = d })
        setAllData(map)
      })
      .finally(() => setLoading(false))
  }, [years])

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading historical data...</span>
    </div>
  )

  if (Object.keys(allData).length === 0) return (
    <div className="flex items-center justify-center h-40">
      <p className="text-gray-500 text-sm">No historical data available</p>
    </div>
  )

  const survivalDatasets = Object.entries(allData).map(([year, data]) => ({
    year,
    data: data.survival_curve,
  }))

  // Season summary table
  const summaries = Object.values(allData).map(d => d.summary).sort((a, b) => b.year - a.year)

  // Weekly picks for selected year
  const weeklyPicks = selectedYear && allData[selectedYear]
    ? allData[selectedYear].weekly_picks
    : {}

  return (
    <div className="flex flex-col gap-6">

      {/* Season summary table */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <p className="text-white font-semibold text-sm">Season Summary</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                {['Year', 'Entries', 'Contestants', 'Final Survivors', 'Biggest Elim Week', '% Eliminated', 'Median Survival', 'Survival Rate'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summaries.map(s => (
                <tr key={s.year} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                  <td className="px-4 py-2.5 text-white font-semibold text-xs">{s.year}</td>
                  <td className="px-4 py-2.5 text-gray-300 text-xs font-mono">{s.total_entries?.toLocaleString()}</td>
                  <td className="px-4 py-2.5 text-gray-300 text-xs font-mono">{s.total_contestants?.toLocaleString()}</td>
                  <td className="px-4 py-2.5 text-xs font-mono">
                    <span className={s.final_survivors > 0 ? 'text-green-400' : 'text-gray-500'}>
                      {s.final_survivors}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-yellow-400 text-xs font-mono">Wk {s.biggest_elimination_week}</td>
                  <td className="px-4 py-2.5 text-red-400 text-xs font-mono">{s.biggest_elimination_pct?.toFixed(1)}%</td>
                  <td className="px-4 py-2.5 text-gray-300 text-xs font-mono">Wk {s.median_survival_week}</td>
                  <td className="px-4 py-2.5 text-xs font-mono">
                    <span className="text-gray-400">
                      {((s.final_survivors / s.total_entries) * 100).toFixed(3)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Multi-year survival curves */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
        <p className="text-white font-semibold text-sm mb-4">Survival Curves — All Years</p>
        <SurvivalChart datasets={survivalDatasets} />
      </div>

      {/* Elimination heatmap */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
        <p className="text-white font-semibold text-sm mb-4">Weekly Elimination Heatmap</p>
        <p className="text-gray-500 text-xs mb-3">% of all entries eliminated each week</p>
        <EliminationHeatmap allData={allData} />
      </div>

      {/* Most popular picks by week */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <p className="text-white font-semibold text-sm">Most Popular Picks by Week</p>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400">Year</label>
            <select
              value={selectedYear || ''}
              onChange={e => setSelectedYear(Number(e.target.value))}
              className="bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-green-600"
            >
              {years.map(y => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left px-3 py-2 text-gray-500">Week</th>
                <th className="text-left px-3 py-2 text-gray-500">#1 Pick</th>
                <th className="text-left px-3 py-2 text-gray-500">#2 Pick</th>
                <th className="text-left px-3 py-2 text-gray-500">#3 Pick</th>
                <th className="text-left px-3 py-2 text-gray-500">#4 Pick</th>
                <th className="text-left px-3 py-2 text-gray-500">#5 Pick</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(weeklyPicks).sort((a, b) => Number(a[0]) - Number(b[0])).map(([week, picks]) => (
                <tr key={week} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                  <td className="px-3 py-2 text-gray-400 font-medium">Wk {week}</td>
                  {picks.slice(0, 5).map((p, i) => (
                    <td key={i} className="px-3 py-2">
                      <span className={`font-semibold ${i === 0 ? 'text-white' : 'text-gray-400'}`}>
                        {p.team}
                      </span>
                      <span className={`ml-1.5 ${
                        p.pct >= 40 ? 'text-red-400' :
                        p.pct >= 20 ? 'text-yellow-400' : 'text-gray-500'
                      }`}>
                        {p.pct.toFixed(1)}%
      </span>
                    </td>
                  ))}
                  {picks.length < 5 && Array.from({ length: 5 - picks.length }).map((_, i) => (
                    <td key={`empty-${i}`} className="px-3 py-2 text-gray-700">—</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  )
}
