import { useState, useEffect } from 'react'
import ConstraintsPanel from './ConstraintsPanel'
import ResultsPanel from './ResultsPanel'
import { runOptimizer, fetchWeeks, fetchPickPercentages } from '../../api/client'
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

export default function OptimizerView() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [upcomingWeek, setUpcomingWeek] = useState(1)
  const [allPickPcts, setAllPickPcts] = useState({})
  const [weekOptions, setWeekOptions] = useState([])
  

  useEffect(() => {
    fetchWeeks()
      .then(data => {
        setUpcomingWeek(data.upcoming_week || 1)
        // Handle both old format (array of ints) and new format (array of objects)
        const weeks = data.weeks || []
        const options = weeks.map(w =>
          typeof w === 'object' ? w : { week: w, label: `Week ${w}` }
        )
        setWeekOptions(options)
      })
      .catch(() => {})

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
  }, [])

  const handleSubmit = async (constraints) => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const data = await runOptimizer(constraints)
      setResults(data)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
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
  )
}
