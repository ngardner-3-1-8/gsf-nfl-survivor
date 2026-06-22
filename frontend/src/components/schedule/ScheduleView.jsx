import { useState, useEffect, useMemo } from 'react'
import { fetchSchedule, fetchWeeks, fetchAvailableYears } from '../../api/client'
import ScheduleFilters from './ScheduleFilters'
import ScheduleTable from './ScheduleTable'
import { useAvailableYears } from '../../hooks/useAvailableYears'
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
  const [availableWeeks, setAvailableWeeks] = useState([])

  const { years, selectedYear, setSelectedYear, isHistorical } = useAvailableYears()

  // Reload schedule when year changes
  useEffect(() => {
    if (!selectedYear) return
    setLoading(true)
    setError(null)
    setSelectedWeeks([])
    fetchSchedule(null, selectedYear)
      .then(data => {
        const g = data.games || []
        setGames(g)
        const weeks = [...new Set(g.map(r => r['Week_x'] ?? r['Week']))]
          .sort((a, b) => a - b)
        setAvailableWeeks(weeks)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedYear])

  // Apply filters
  const filtered = useMemo(() => {
    let rows = [...games]

    if (selectedWeeks.length > 0) {
      rows = rows.filter(r => selectedWeeks.includes(r['Week_x'] ?? r['Week']))
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
        <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-4">
          <YearSelector years={years} selectedYear={selectedYear} onChange={setSelectedYear} />
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
          <YearSelector years={years} selectedYear={selectedYear} onChange={setSelectedYear} />
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
          years={years}
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
        availableWeeks={availableWeeks}
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
