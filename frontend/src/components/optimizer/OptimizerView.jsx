import { useState, useEffect } from 'react'
import ConstraintsPanel from './ConstraintsPanel'
import ResultsPanel from './ResultsPanel'
import { runOptimizer, fetchWeeks, fetchPickPercentages, fetchCircaContestStats } from '../../api/client'
import { useAvailableYears } from '../../hooks/useAvailableYears'
import YearSelector from '../ui/YearSelector'

export default function OptimizerView() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [upcomingWeek, setUpcomingWeek] = useState(1)
  const [allPickPcts, setAllPickPcts] = useState({})
  const [weekOptions, setWeekOptions] = useState([])
  const [circaStats, setCircaStats] = useState(null)

  const { years, selectedYear, setSelectedYear, isHistorical } = useAvailableYears()

  // Circa contest stats (live entries + static prize) for the summary bar.
  useEffect(() => {
    if (isHistorical) { setCircaStats(null); return }
    fetchCircaContestStats().then(setCircaStats).catch(() => setCircaStats(null))
  }, [isHistorical])

  // Reload weeks when year changes
  useEffect(() => {
    if (!selectedYear) return
    fetchWeeks(selectedYear)
      .then(data => {
        setUpcomingWeek(data.upcoming_week || 1)
        const weeks = data.weeks || []
        const options = weeks.map(w =>
          typeof w === 'object' ? w : { week: w, label: `Week ${w}` }
        )
        setWeekOptions(options)
      })
      .catch(() => {})
  }, [selectedYear])

  // Pick percentages — only relevant for current year
  useEffect(() => {
    if (isHistorical) return
    fetchPickPercentages()
      .then(data => {
        const lookup = {}
        data.picks.forEach(({ week, team, pick_pct }) => {
          if (!lookup[week]) lookup[week] = {}
          lookup[week][team] = pick_pct
        })
        setAllPickPcts(lookup)
      })
      .catch(() => {})
  }, [isHistorical])

  const handleSubmit = async (constraints) => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const data = await runOptimizer({ ...constraints, year: selectedYear })
      setResults(data)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  

  return (
    <div className="flex flex-col gap-4">

      {/* Year selector */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-4 flex-wrap">
        <YearSelector
          years={years}
          selectedYear={selectedYear}
          onChange={setSelectedYear}
        />
        {isHistorical && (
          <span className="text-xs text-amber-400 ml-2">
            ⚠️ Historical mode — optimizer runs against {selectedYear} actual data
          </span>
        )}
      </div>

      {/* Contest summary — live entries + static prize (Circa) */}
      {!isHistorical && circaStats && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-6 flex-wrap text-sm">
          <div>
            <p className="text-gray-500 text-xs">Total Entries</p>
            <p className="text-white font-mono">{circaStats.total_entries?.toLocaleString() ?? '\u2014'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Surviving Entries</p>
            <p className="text-white font-mono">{circaStats.surviving_entries?.toLocaleString() ?? '\u2014'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Entry Fee</p>
            <p className="text-white font-mono">${circaStats.entry_fee?.toLocaleString() ?? '\u2014'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Total Prizes</p>
            <p className="text-white font-mono">${circaStats.total_prize?.toLocaleString() ?? '\u2014'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Average Entry Value</p>
            <p className="text-white font-mono">
              {circaStats.total_prize && circaStats.surviving_entries
                ? `$${Math.round(circaStats.total_prize / circaStats.surviving_entries).toLocaleString()}`
                : '\u2014'}
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-[340px_1fr] gap-6 items-start">
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 sticky top-6">
          <h2 className="text-base font-semibold text-white mb-4">Constraints</h2>
          <ConstraintsPanel
            onSubmit={handleSubmit}
            loading={loading}
            upcomingWeek={upcomingWeek}
            weekOptions={weekOptions}
          />
        </div>
        <div className="min-h-[400px]">
          <h2 className="text-base font-semibold text-white mb-4">Results</h2>
          <ResultsPanel
            results={results}
            loading={loading}
            error={error}
            allPickPcts={allPickPcts}
          />
        </div>
      </div>
    </div>
  )
}
