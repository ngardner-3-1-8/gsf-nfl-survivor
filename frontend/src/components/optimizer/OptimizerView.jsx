import { useState, useEffect } from 'react'
import ConstraintsPanel from './ConstraintsPanel'
import ResultsPanel from './ResultsPanel'
import { runOptimizer, fetchWeeks, fetchPickPercentages } from '../../api/client'

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
        setWeekOptions(data.weeks || [])  // now an array of {week, label}
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
