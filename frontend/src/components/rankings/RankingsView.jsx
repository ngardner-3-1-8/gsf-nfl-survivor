import { useState, useEffect } from 'react'
import { fetchRankings } from '../../api/client'

const TEAM_FULL_NAMES = {
  ARI: 'Arizona Cardinals', ATL: 'Atlanta Falcons', BAL: 'Baltimore Ravens',
  BUF: 'Buffalo Bills', CAR: 'Carolina Panthers', CHI: 'Chicago Bears',
  CIN: 'Cincinnati Bengals', CLE: 'Cleveland Browns', DAL: 'Dallas Cowboys',
  DEN: 'Denver Broncos', DET: 'Detroit Lions', GB: 'Green Bay Packers',
  HOU: 'Houston Texans', IND: 'Indianapolis Colts', JAX: 'Jacksonville Jaguars',
  KC: 'Kansas City Chiefs', LA: 'Los Angeles Rams', LAC: 'Los Angeles Chargers',
  LV: 'Las Vegas Raiders', MIA: 'Miami Dolphins', MIN: 'Minnesota Vikings',
  NE: 'New England Patriots', NO: 'New Orleans Saints', NYG: 'New York Giants',
  NYJ: 'New York Jets', PHI: 'Philadelphia Eagles', PIT: 'Pittsburgh Steelers',
  SEA: 'Seattle Seahawks', SF: 'San Francisco 49ers', TB: 'Tampa Bay Buccaneers',
  TEN: 'Tennessee Titans', WAS: 'Washington Commanders',
}

function ratingCell(val, preseason) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  const pre = parseFloat(preseason)
  let changeEl = null
  if (!isNaN(pre)) {
    const diff = v - pre
    if (Math.abs(diff) >= 0.1) {
      changeEl = (
        <span className={`ml-1.5 text-xs ${diff > 0 ? 'text-green-400' : 'text-red-400'}`}>
          {diff > 0 ? '▲' : '▼'}{Math.abs(diff).toFixed(2)}
        </span>
      )
    }
  }
  const color = v >= 5 ? 'text-green-400' : v >= 0 ? 'text-yellow-400' : 'text-red-400'
  return (
    <span className={`font-mono text-xs font-medium ${color}`}>
      {v.toFixed(2)}{changeEl}
    </span>
  )
}

function rankChangeCell(current, last) {
  const c = parseInt(current)
  const l = parseInt(last)
  if (isNaN(c)) return <span className="text-gray-600">—</span>
  if (isNaN(l) || c === l) return <span className="font-mono text-xs text-gray-400">{c}</span>
  const diff = l - c // positive = moved up
  return (
    <span className="font-mono text-xs">
      <span className="text-white">{c}</span>
      <span className={`ml-1.5 text-xs ${diff > 0 ? 'text-green-400' : 'text-red-400'}`}>
        {diff > 0 ? '▲' : '▼'}{Math.abs(diff)}
      </span>
    </span>
  )
}

function epaCell(val) {
  const v = parseFloat(val)
  if (isNaN(v)) return <span className="text-gray-600">—</span>
  const color = v >= 3 ? 'text-green-400' : v >= 0 ? 'text-yellow-400' : 'text-red-400'
  return <span className={`font-mono text-xs ${color}`}>{(v > 0 ? '+' : '') + v.toFixed(2)}</span>
}

export default function RankingsView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sortKey, setSortKey] = useState('Rank')
  const [sortDir, setSortDir] = useState('asc')
  const [search, setSearch] = useState('')

  useEffect(() => {
    fetchRankings()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading rankings...</span>
    </div>
  )

  if (error) return (
    <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
      <p className="text-red-400 text-sm font-medium">Error loading rankings</p>
      <p className="text-red-300 text-sm mt-1">{error}</p>
    </div>
  )

  const rankings = data?.rankings || []
  const hasPreseason = data?.has_preseason

  // Filter by search
  const filtered = rankings.filter(r => {
    const q = search.trim().toLowerCase()
    if (!q) return true
    const abbr = String(r.Team || '').toLowerCase()
    const full = (TEAM_FULL_NAMES[r.Team] || '').toLowerCase()
    const qb = String(r['QB used'] || r['Projected QB'] || '').toLowerCase()
    return abbr.includes(q) || full.includes(q) || qb.includes(q)
  })

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey] ?? ''
    const bv = b[sortKey] ?? ''
    const an = parseFloat(av)
    const bn = parseFloat(bv)
    if (!isNaN(an) && !isNaN(bn)) return sortDir === 'asc' ? an - bn : bn - an
    return sortDir === 'asc'
      ? String(av).localeCompare(String(bv))
      : String(bv).localeCompare(String(av))
  })

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('asc') }
  }

  const SortTh = ({ col, label }) => (
    <th
      onClick={() => handleSort(col)}
      className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 cursor-pointer hover:text-white whitespace-nowrap select-none"
    >
      {label}
      {sortKey === col && (
        <span className="ml-1 text-green-400">{sortDir === 'asc' ? '↑' : '↓'}</span>
      )}
    </th>
  )

  return (
    <div className="flex flex-col gap-4">

      {/* Header bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div>
          <p className="text-white font-semibold text-sm">
            Power Rankings — Week {data?.upcoming_week} {data?.target_year}
          </p>
          <p className="text-gray-500 text-xs mt-0.5">
            {hasPreseason
              ? 'Arrows show change from preseason Week 1 ratings'
              : 'Rank change shows week-over-week movement'}
          </p>
        </div>
        <div className="ml-auto">
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search team or QB..."
            className="bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-2 w-48 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600"
          />
        </div>
      </div>

      {/* Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <SortTh col="Rank" label="Rank" />
                <th className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">Team</th>
                <SortTh col="Power Rating" label="Power Rating" />
                {hasPreseason && <SortTh col="Preseason Power Rating" label="Preseason PR" />}
                <SortTh col="MP_Rating" label="MP Rating" />
                {hasPreseason && <SortTh col="Preseason MP Rating" label="Preseason MP" />}
                <SortTh col="Rank_Last" label="Last Wk Rank" />
                <SortTh col="Change" label="Rank Δ" />
                <th className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">QB</th>
                <SortTh col="Offensive EPA/Game" label="Off EPA" />
                <SortTh col="Defensive EPA/Game" label="Def EPA" />
                <SortTh col="Special Teams EPA/Game" label="ST EPA" />
                <SortTh col="EPA Rating" label="EPA Rating" />
                <SortTh col="SR Rating (Pts)" label="SR Rating" />
                <SortTh col="Strength of Schedule" label="SOS" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => {
                const rankChange = parseInt(row['Rank_Last']) - parseInt(row['Rank'])
                return (
                  <tr
                    key={i}
                    className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
                  >
                    {/* Rank */}
                    <td className="px-4 py-2.5">
                      <span className="font-mono text-xs font-medium text-white">
                        {row['Rank'] ?? '—'}
                      </span>
                    </td>

                    {/* Team */}
                    <td className="px-4 py-2.5">
                      <div className="flex flex-col">
                        <span className="text-white font-semibold text-xs">{row['Team']}</span>
                        <span className="text-gray-500 text-xs">
                          {TEAM_FULL_NAMES[row['Team']] || ''}
                        </span>
                      </div>
                    </td>

                    {/* Power Rating with preseason comparison */}
                    <td className="px-4 py-2.5">
                      {ratingCell(row['Power Rating'], hasPreseason ? row['Preseason Power Rating'] : null)}
                    </td>
                    {hasPreseason && (
                      <td className="px-4 py-2.5 text-xs text-gray-500 font-mono">
                        {parseFloat(row['Preseason Power Rating'])?.toFixed(2) ?? '—'}
                      </td>
                    )}

                    {/* MP Rating */}
                    <td className="px-4 py-2.5">
                      {ratingCell(row['MP_Rating'], hasPreseason ? row['Preseason MP Rating'] : null)}
                    </td>
                    {hasPreseason && (
                      <td className="px-4 py-2.5 text-xs text-gray-500 font-mono">
                        {parseFloat(row['Preseason MP Rating'])?.toFixed(2) ?? '—'}
                      </td>
                    )}

                    {/* Last week rank + change */}
                    <td className="px-4 py-2.5 text-xs text-gray-400 font-mono">
                      {row['Rank_Last'] ?? '—'}
                    </td>
                    <td className="px-4 py-2.5">
                      {rankChangeCell(row['Rank'], row['Rank_Last'])}
                    </td>

                    {/* QB */}
                    <td className="px-4 py-2.5 text-xs text-gray-300">
                      {row['QB used'] || row['Projected QB'] || '—'}
                    </td>

                    {/* EPA columns */}
                    <td className="px-4 py-2.5">{epaCell(row['Offensive EPA/Game'])}</td>
                    <td className="px-4 py-2.5">{epaCell(row['Defensive EPA/Game'])}</td>
                    <td className="px-4 py-2.5">{epaCell(row['Special Teams EPA/Game'])}</td>
                    <td className="px-4 py-2.5 text-xs text-gray-400 font-mono">
                      {parseFloat(row['EPA Rating'])?.toFixed(2) ?? '—'}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-gray-400 font-mono">
                      {parseFloat(row['SR Rating (Pts)'])?.toFixed(2) ?? '—'}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-gray-400 font-mono">
                      {parseFloat(row['Strength of Schedule'])?.toFixed(3) ?? '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div className="px-4 py-2.5 border-t border-gray-800 flex items-center gap-6 text-xs text-gray-500 flex-wrap">
          <span className="flex items-center gap-1.5">
            <span className="text-green-400">▲</span> Rating improved vs preseason
          </span>
          <span className="flex items-center gap-1.5">
            <span className="text-red-400">▼</span> Rating declined vs preseason
          </span>
          <span className="ml-auto">{sorted.length} teams</span>
        </div>
      </div>
    </div>
  )
}
