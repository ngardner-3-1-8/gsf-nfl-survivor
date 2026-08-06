import { useState, useEffect } from 'react'
import ConstraintsPanel from './ConstraintsPanel'
import ResultsPanel from './ResultsPanel'
import { runOptimizer, fetchWeeks, fetchPickPercentages } from '../../api/client'
import { useAvailableYears } from '../../hooks/useAvailableYears'
import YearSelector from '../ui/YearSelector'

export default function OptimizerView() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [upcomingWeek, setUpcomingWeek] = useState(1)
  const [allPickPcts, setAllPickPcts] = useState({})
  const [weekOptions, setWeekOptions] = useState([])

  const { years, selectedYear, setSelectedYear, isHistorical } = useAvailableYears()

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
