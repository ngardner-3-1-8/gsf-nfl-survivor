import { useState, useEffect, useMemo } from 'react'
import { fetchSchedule } from '../../api/client'
import ScheduleFilters from './ScheduleFilters'
import ScheduleTable from './ScheduleTable'
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

const COLUMN_VIEWS = ['Overview', 'Odds & Win%', 'Situational', 'Contest', 'Betting', 'Bayesian']

export default function ScheduleView() {
  const [games, setGames] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeView, setActiveView] = useState('Overview')
  const [selectedWeeks, setSelectedWeeks] = useState([])
  const [teamSearch, setTeamSearch] = useState('')
  const [showFilter, setShowFilter] = useState('all') // all, home, away, favorites
  const [availableWeeks, setAvailableWeeks] = useState([])

  useEffect(() => {
    setLoading(true)
    fetchSchedule()
      .then(data => {
        const g = data.games || []
        setGames(g)
        // Build unique week list
        const weeks = [...new Set(g.map(r => r['Week_x'] ?? r['Week']))].sort((a, b) => a - b)
        setAvailableWeeks(weeks)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  // Apply filters
  const filtered = useMemo(() => {
    let rows = [...games]

    // Week filter
    if (selectedWeeks.length > 0) {
      rows = rows.filter(r => selectedWeeks.includes(r['Week_x'] ?? r['Week']))
    }

    // Team search
    if (teamSearch.trim()) {
      const q = teamSearch.trim().toLowerCase()
      rows = rows.filter(r =>
        String(r['Away Team'] || '').toLowerCase().includes(q) ||
        String(r['Home Team'] || '').toLowerCase().includes(q)
      )
    }

    // Home/Away filter
    if (showFilter === 'home') {
      rows = rows.filter(r =>
        String(r['Home Team'] || '').toLowerCase().includes(teamSearch.trim().toLowerCase())
      )
    } else if (showFilter === 'away') {
      rows = rows.filter(r =>
        String(r['Away Team'] || '').toLowerCase().includes(teamSearch.trim().toLowerCase())
      )
    }

    // Favorites only — use consensus win pct > 0.5 for home team
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
      <div className="flex items-center justify-center h-64 gap-3">
        <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
        <span className="text-gray-400 text-sm">Loading schedule...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
        <p className="text-red-400 text-sm font-medium">Error loading schedule</p>
        <p className="text-red-300 text-sm mt-1">{error}</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
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
