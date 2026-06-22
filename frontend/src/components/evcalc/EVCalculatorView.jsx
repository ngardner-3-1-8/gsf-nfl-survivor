import { useState, useEffect, useMemo } from 'react'
import { fetchWeeks, fetchSchedule } from '../../api/client'
import { useAvailableYears } from '../../hooks/useAvailableYears'
import YearSelector from '../ui/YearSelector'

// Inside the component, replace the existing year-unaware fetch:
const { years, selectedYear, setSelectedYear, isHistorical } = useAvailableYears()

// Add year to all fetchSchedule and fetchWeeks calls:
useEffect(() => {
  if (!selectedYear) return
  fetchWeeks(selectedYear).then(...)
}, [selectedYear])

useEffect(() => {
  if (!selectedWeek || !selectedYear) return
  fetchSchedule(selectedWeek.week, selectedYear).then(...)
}, [selectedWeek, selectedYear])

// Add the YearSelector to the header bar JSX:
<YearSelector years={years} selectedYear={selectedYear} onChange={setSelectedYear} />

// ── EV Formula ────────────────────────────────────────────────
// EV(team) = win_pct / expected_survivors
// expected_survivors = sum of (pick_pct × win_pct) for all teams this week
function calculateEV(rows) {
  const expectedSurvivors = rows.reduce((sum, r) => {
    return sum + (r.pick_pct / 100) * (r.win_pct / 100)
  }, 0)
  if (expectedSurvivors === 0) return rows.map(r => ({ ...r, ev: 0 }))
  return rows.map(r => ({
    ...r,
    ev: (r.win_pct / 100) / expectedSurvivors,
  }))
}

// ── Inline editable number cell ───────────────────────────────
function EditableCell({ value, onChange, suffix = '', min = 0, max = 100, step = 0.1 }) {
  const [editing, setEditing] = useState(false)
  const [localVal, setLocalVal] = useState(value.toFixed(1))

  useEffect(() => {
    if (!editing) setLocalVal(Number(value).toFixed(1))
  }, [value, editing])

  const commit = () => {
    const parsed = parseFloat(localVal)
    if (!isNaN(parsed)) {
      const clamped = Math.min(max, Math.max(min, parsed))
      onChange(clamped)
      setLocalVal(clamped.toFixed(1))
    } else {
      setLocalVal(Number(value).toFixed(1))
    }
    setEditing(false)
  }

  if (editing) {
    return (
      <input
        autoFocus
        type="number"
        value={localVal}
        step={step}
        min={min}
        max={max}
        onChange={e => setLocalVal(e.target.value)}
        onBlur={commit}
        onKeyDown={e => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') { setEditing(false); setLocalVal(Number(value).toFixed(1)) } }}
        className="w-20 bg-gray-800 border border-green-600 text-white text-xs rounded px-2 py-1 font-mono focus:outline-none"
      />
    )
  }

  return (
    <button
      onClick={() => setEditing(true)}
      className="text-xs font-mono text-white hover:text-green-400 hover:underline transition-colors px-1 py-0.5 rounded"
      title="Click to edit"
    >
      {Number(value).toFixed(1)}{suffix}
    </button>
  )
}

// ── EV color ─────────────────────────────────────────────────
function evColor(ev) {
  if (ev >= 0.12) return 'text-green-400'
  if (ev >= 0.08) return 'text-yellow-400'
  if (ev >= 0.05) return 'text-orange-400'
  return 'text-red-400'
}

function winPctColor(pct) {
  if (pct >= 70) return 'text-green-400'
  if (pct >= 55) return 'text-yellow-400'
  return 'text-red-400'
}

// ── Main component ────────────────────────────────────────────
export default function EVCalculatorView() {
  const [weekOptions, setWeekOptions] = useState([])
  const [selectedWeek, setSelectedWeek] = useState(null)
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [hasEdits, setHasEdits] = useState(false)
  const [sortKey, setSortKey] = useState('ev')
  const [sortDir, setSortDir] = useState('desc')

  // Load week options on mount
  useEffect(() => {
    fetchWeeks()
      .then(data => {
        const options = data.weeks || []
        setWeekOptions(options)
        if (options.length > 0) {
          setSelectedWeek(options.find(w => w.week === data.upcoming_week) || options[0])
        }
      })
      .catch(() => {})
  }, [])

  // Load games when week changes
  useEffect(() => {
    if (!selectedWeek) return
    setLoading(true)
    setHasEdits(false)
    fetchSchedule(selectedWeek.week)
      .then(data => {
        const games = data.games || []
        // Build one row per team per game
        const built = []
        games.forEach(game => {
          const awayPick = parseFloat(game['Away Pick %']) || 0
          const homePick = parseFloat(game['Home Pick %']) || 0
          const awayWin = parseFloat(game['Consensus Away Win Pct']) * 100 || 50
          const homeWin = parseFloat(game['Consensus Home Win Pct']) * 100 || 50

          built.push({
            id: `${game['Away Team']}_${game['Week_x']}`,
            team: game['Away Team'],
            opponent: game['Home Team'],
            home_or_away: 'Away',
            game_label: `${game['Away Team']} @ ${game['Home Team']}`,
            circa_week: game['Circa Week'] || `Week ${game['Week_x']}`,
            pick_pct: awayPick * 100,    // store as 0-100
            win_pct: awayWin,            // store as 0-100
            // original values for reset
            _orig_pick: awayPick * 100,
            _orig_win: awayWin,
            // flags
            is_thanksgiving: !!(game['Away Team Thanksgiving Favorite'] || game['Away Team Thanksgiving Underdog']),
            is_christmas: !!(game['Away Team Christmas Favorite'] || game['Away Team Christmas Underdog']),
          })
          built.push({
            id: `${game['Home Team']}_${game['Week_x']}`,
            team: game['Home Team'],
            opponent: game['Away Team'],
            home_or_away: 'Home',
            game_label: `${game['Away Team']} @ ${game['Home Team']}`,
            circa_week: game['Circa Week'] || `Week ${game['Week_x']}`,
            pick_pct: homePick * 100,
            win_pct: homeWin,
            _orig_pick: homePick * 100,
            _orig_win: homeWin,
            is_thanksgiving: !!(game['Home Team Thanksgiving Favorite'] || game['Home Team Thanksgiving Underdog']),
            is_christmas: !!(game['Home Team Christmas Favorite'] || game['Home Team Christmas Underdog']),
          })
        })
        setRows(built)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedWeek])

  // Calculate EV live from current rows
  const rowsWithEV = useMemo(() => {
    const eligible = rows.filter(r => r.pick_pct > 0)
    const withEV = calculateEV(eligible)
    // Return all rows but set EV to null for ineligible teams
    return rows.map(r => {
      if (r.pick_pct <= 0) return { ...r, ev: null }
      return withEV.find(e => e.id === r.id) || { ...r, ev: null }
    })
  }, [rows])

  // Sort
  const sorted = useMemo(() => {
    return [...rowsWithEV].sort((a, b) => {
      const av = a[sortKey] ?? 0
      const bv = b[sortKey] ?? 0
      return sortDir === 'desc' ? bv - av : av - bv
    })
  }, [rowsWithEV, sortKey, sortDir])

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const updateRow = (id, field, value) => {
    setRows(prev => prev.map(r => r.id === id ? { ...r, [field]: value } : r))
    setHasEdits(true)
  }

  const resetAll = () => {
    setRows(prev => prev.map(r => ({ ...r, pick_pct: r._orig_pick, win_pct: r._orig_win })))
    setHasEdits(false)
  }

  // Summary stats
  const totalPickPct = rows.reduce((s, r) => s + r.pick_pct, 0)
  const expectedSurvivors = rows.reduce((s, r) => s + (r.pick_pct / 100) * (r.win_pct / 100), 0)
  const topEV = rowsWithEV.length > 0 ? Math.max(...rowsWithEV.map(r => r.ev)) : 0

  const SortTh = ({ col, label, right = false }) => (
    <th
      onClick={() => handleSort(col)}
      className={`px-4 py-2.5 text-xs font-medium text-gray-500 cursor-pointer hover:text-white select-none whitespace-nowrap ${right ? 'text-right' : 'text-left'}`}
    >
      {label}
      {sortKey === col && (
        <span className="ml-1 text-green-400">{sortDir === 'desc' ? '↓' : '↑'}</span>
      )}
    </th>
  )

  return (
    <div className="flex flex-col gap-4">

      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div>
          <p className="text-white font-semibold text-sm">EV Calculator</p>
          <p className="text-gray-500 text-xs mt-0.5">
            Edit pick% or win% for any team — EV recalculates instantly
          </p>
        </div>

        {/* Week selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400 uppercase tracking-wide">Week</label>
          <select
            value={selectedWeek?.week || ''}
            onChange={e => {
              const w = weekOptions.find(o => o.week === Number(e.target.value))
              setSelectedWeek(w)
            }}
            className="bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          >
            {weekOptions.map(w => (
              <option key={w.week} value={w.week}>{w.label}</option>
            ))}
          </select>
        </div>

        {/* Summary stats */}
        <div className="flex gap-4 text-xs">
          <div className="text-center">
            <p className="text-gray-500">Total Pick%</p>
            <p className={`font-semibold ${Math.abs(totalPickPct - 100) < 1 ? 'text-green-400' : 'text-yellow-400'}`}>
              {totalPickPct.toFixed(1)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-gray-500">Exp Survivors</p>
            <p className="text-white font-semibold">{(expectedSurvivors * 100).toFixed(1)}%</p>
          </div>
          <div className="text-center">
            <p className="text-gray-500">Top EV</p>
            <p className={`font-semibold ${evColor(topEV)}`}>{topEV.toFixed(4)}</p>
          </div>
        </div>

        {/* Reset button */}
        {hasEdits && (
          <button
            onClick={resetAll}
            className="ml-auto text-xs text-gray-400 hover:text-white border border-gray-700 hover:border-gray-500 px-3 py-1.5 rounded-lg transition-colors"
          >
            ↺ Reset to original values
          </button>
        )}
      </div>

      {/* Edit hint */}
      <div className="flex items-center gap-2 text-xs text-gray-600">
        <span className="w-2 h-2 rounded-full bg-green-600 inline-block" />
        Click any <span className="text-green-400 underline mx-1">value</span> to edit it — EV updates in real time
        {hasEdits && <span className="text-yellow-500 ml-2">· You have unsaved edits</span>}
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex items-center justify-center h-64 gap-3">
          <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-400 text-sm">Loading...</span>
        </div>
      ) : error ? (
        <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">Game</th>
                  <SortTh col="team" label="Team" />
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-gray-500">H/A</th>
                  <SortTh col="win_pct" label="Win % ✏️" right />
                  <SortTh col="pick_pct" label="Pick % ✏️" right />
                  <SortTh col="ev" label="EV" right />
                  <th className="text-right px-4 py-2.5 text-xs font-medium text-gray-500">EV Rank</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, i) => {
                  const rank = row.ev === null ? null :
                    [...rowsWithEV]
                      .filter(r => r.ev !== null)
                      .sort((a, b) => b.ev - a.ev)
                      .findIndex(r => r.id === row.id) + 1
                  const winEdited = Math.abs(row.win_pct - row._orig_win) > 0.05
                  const pickEdited = Math.abs(row.pick_pct - row._orig_pick) > 0.05

                  return (
                    <tr
                      key={row.id}
                      className={`border-b border-gray-800/50 transition-colors ${
                        row.pick_pct <= 0
                          ? 'opacity-40 bg-gray-900/20'
                          : 'hover:bg-gray-800/20'
                      }`}
                    >
                      {/* Game */}
                      <td className="px-4 py-2.5 text-xs text-gray-500 whitespace-nowrap">
                        {row.game_label}
                        {(row.is_thanksgiving || row.is_christmas) && (
                          <span className="ml-1">{row.is_thanksgiving ? '🦃' : '🎄'}</span>
                        )}
                      </td>

                      {/* Team */}
                      <td className="px-4 py-2.5">
                        <span className="text-white font-semibold text-xs">{row.team}</span>
                      </td>

                      {/* H/A */}
                      <td className="px-4 py-2.5">
                        <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                          row.home_or_away === 'Home'
                            ? 'bg-blue-900/50 text-blue-300'
                            : 'bg-gray-800 text-gray-400'
                        }`}>
                          {row.home_or_away}
                        </span>
                      </td>

                      {/* Win % — editable */}
                      <td className="px-4 py-2.5 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {winEdited && <span className="text-yellow-500 text-xs">*</span>}
                          <span className={winPctColor(row.win_pct)}>
                            <EditableCell
                              value={row.win_pct}
                              onChange={v => updateRow(row.id, 'win_pct', v)}
                              suffix="%"
                              min={0}
                              max={100}
                              step={0.1}
                            />
                          </span>
                        </div>
                      </td>

                      {/* Pick % — editable */}
                      <td className="px-4 py-2.5 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {pickEdited && <span className="text-yellow-500 text-xs">*</span>}
                          <EditableCell
                            value={row.pick_pct}
                            onChange={v => updateRow(row.id, 'pick_pct', v)}
                            suffix="%"
                            min={0}
                            max={100}
                            step={0.1}
                          />
                        </div>
                      </td>

                      {/* EV */}
                      <td className="px-4 py-2.5 text-right">
                        {row.ev === null ? (
                          <span className="text-xs text-gray-600 italic">N/A</span>
                        ) : (
                          <span className={`font-mono text-xs font-semibold ${evColor(row.ev)}`}>
                            {row.ev.toFixed(4)}
                          </span>
                        )}
                      </td>
                      
                      {/* EV Rank */}
                      <td className="px-4 py-2.5 text-right">
                        {row.ev === null ? (
                          <span className="text-xs text-gray-600">—</span>
                        ) : (
                          <span className={`font-mono text-xs ${
                            rank <= 3 ? 'text-green-400 font-semibold' :
                            rank <= 6 ? 'text-yellow-400' : 'text-gray-500'
                          }`}>
                            #{rank}
                          </span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Footer — pick% sum warning */}
          <div className="px-4 py-2.5 border-t border-gray-800 flex items-center gap-4 text-xs flex-wrap">
            {Math.abs(totalPickPct - 100) > 2 ? (
              <span className="text-yellow-400">
                ⚠️ Pick percentages sum to {totalPickPct.toFixed(1)}% — should be ~100% for accurate EV
              </span>
            ) : (
              <span className="text-gray-600">
                Pick% sum: {totalPickPct.toFixed(1)}% ✓
              </span>
            )}
            <span className="text-gray-600 ml-auto">
              {rows.length} teams · {rows.length / 2} games
            </span>
          </div>
        </div>
      )}

      {/* Formula explainer */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-xs text-gray-500">
        <p className="font-medium text-gray-400 mb-1">How EV is calculated</p>
        <p>
          EV(team) = Win%(team) ÷ Expected Survivors · · ·
          Expected Survivors = Σ (Pick% × Win%) for all teams this week · · ·
          A higher EV means better value relative to how many contest entries will be eliminated if that team loses.
        </p>
      </div>
    </div>
  )
}
