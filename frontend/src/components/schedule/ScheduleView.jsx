import { useState, useEffect, useMemo } from 'react'
import { fetchSchedule, fetchContestYears } from '../../api/client'
import ScheduleFilters from './ScheduleFilters'
import ScheduleTable from './ScheduleTable'
import { useContest } from '../../contexts/ContestContext'
import YearSelector from '../ui/YearSelector'

const COLUMN_VIEWS = ['Overview', 'Odds & Win%', 'Situational', 'Contest', 'Betting', 'Bayesian']

export default function ScheduleView() {
  const [games, setGames] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeView, setActiveView] = useState('Overview')
  const [selectedWeeks, setSelectedWeeks] = useState([])
  const [teamSearch, setTeamSearch] = useState('')
  const [showFilter, setShowFilter] = useState('all')
  const [weekOptions, setWeekOptions] = useState([])

  const { contest } = useContest()

  // Years are per contest: Circa runs back to 2020, the Splash contests only
  // exist from 2026 on. Reload the year list when the contest changes and
  // keep the selected year valid for the new contest (newest = current).
  const [years, setYears] = useState([])
  const [selectedYear, setSelectedYear] = useState(null)

  useEffect(() => {
    fetchContestYears(contest)
      .then(data => {
        const ys = (data.years || []).slice().sort((a, b) => b - a)
        setYears(ys)
        setSelectedYear(prev => (ys.includes(prev) ? prev : (ys[0] ?? null)))
      })
      .catch(() => { setYears([]); setSelectedYear(null) })
  }, [contest])

  // The newest year with data for a contest is its current season.
  const currentYear = years[0] ?? null
  const effectiveYear = years.includes(selectedYear) ? selectedYear : currentYear
  const isHistorical = effectiveYear != null && currentYear != null && effectiveYear !== currentYear

  // YearSelector expects objects: { year, label, is_current }.
  const yearOptions = useMemo(
    () => years.map(y => ({ year: y, label: String(y), is_current: y === currentYear })),
    [years, currentYear]
  )

  // Reload schedule when the effective year or the global contest changes.
  // `ignore` drops a stale response if year/contest changed before it resolved.
  useEffect(() => {
    if (!effectiveYear) return
    let ignore = false
    setLoading(true)
    setError(null)
    setSelectedWeeks([])
    fetchSchedule(null, effectiveYear, contest)
      .then(data => {
        if (ignore) return
        setGames(data.games || [])
        setWeekOptions(data.week_options || [])
      })
      .catch(e => { if (!ignore) setError(e.message) })
      .finally(() => { if (!ignore) setLoading(false) })
    return () => { ignore = true }
  }, [effectiveYear, contest])

  // Apply filters
  const filtered = useMemo(() => {
    let rows = [...games]

    if (selectedWeeks.length > 0) {
      rows = rows.filter(r => selectedWeeks.includes(r['_DisplayWeek'] ?? r['Week_x'] ?? r['Week']))
    }

    if (teamSearch.trim()) {
      const q = teamSearch.trim().toLowerCase()
      rows = rows.filter(r =>
        String(r['Away Team'] || '').toLowerCase().includes(q) ||
        String(r['Home Team'] || '').toLowerCase().includes(q)
      )
    }

    if (showFilter === 'home') {
      rows = rows.filter(r =>
        String(r['Home Team'] || '').toLowerCase().includes(teamSearch.trim().toLowerCase())
      )
    } else if (showFilter === 'away') {
      rows = rows.filter(r =>
        String(r['Away Team'] || '').toLowerCase().includes(teamSearch.trim().toLowerCase())
      )
    }

    if (showFilter === 'favorites') {
      rows = rows.filter(r => {
        const homeWin = parseFloat(r['Consensus Home Win Pct'] || 0)
        return homeWin > 0.5
      })
    }

    return rows
  }, [games, selectedWeeks, teamSearch, showFilter])

  const toggleWeek = (week) => {
    setSelectedWeeks(prev =>
      prev.includes(week) ? prev.filter(w => w !== week) : [...prev, week]
    )
  }

  if (loading) {
    return (
      <div className="flex flex-col gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-4 flex-wrap">
          <YearSelector years={yearOptions} selectedYear={selectedYear} onChange={setSelectedYear} />
        </div>
        <div className="flex items-center justify-center h-64 gap-3">
          <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-400 text-sm">
            Loading {isHistorical ? `${selectedYear} historical` : ''} schedule...
          </span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-4">
          <YearSelector years={yearOptions} selectedYear={selectedYear} onChange={setSelectedYear} />
        </div>
        <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
          <p className="text-red-400 text-sm font-medium">Error loading schedule</p>
          <p className="text-red-300 text-sm mt-1">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">

      {/* Year selector */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-4 flex-wrap">
        <YearSelector
          years={yearOptions}
          selectedYear={selectedYear}
          onChange={year => {
            setSelectedYear(year)
          }}
        />
        {isHistorical && (
          <span className="text-xs text-amber-400">
            📋 {selectedYear} season — showing actual results
          </span>
        )}
        <span className="ml-auto text-xs text-gray-500">
          {filtered.length} of {games.length} games
        </span>
      </div>

      <ScheduleFilters
        weekOptions={weekOptions}
        selectedWeeks={selectedWeeks}
        onToggleWeek={toggleWeek}
        onClearWeeks={() => setSelectedWeeks([])}
        teamSearch={teamSearch}
        onTeamSearch={setTeamSearch}
        showFilter={showFilter}
        onShowFilter={setShowFilter}
        activeView={activeView}
        onViewChange={setActiveView}
        columnViews={COLUMN_VIEWS}
        totalGames={games.length}
        filteredGames={filtered.length}
      />

      <ScheduleTable
        games={filtered}
        activeView={activeView}
      />
    </div>
  )
}
