import { useState, useEffect } from 'react'
import ConstraintsPanel from './ConstraintsPanel'
import ResultsPanel from './ResultsPanel'
import { runOptimizer, fetchWeeks } from '../../api/client'

export default function OptimizerView() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [upcomingWeek, setUpcomingWeek] = useState(1)

  useEffect(() => {
    fetchWeeks()
      .then(data => setUpcomingWeek(data.upcoming_week || 1))
      .catch(() => {}) // silently fail — default to week 1
  }, [])

  export async function fetchPickPercentages() {
    const res = await fetch(`${API_URL}/api/pick-percentages`)
    if (!res.ok) throw new Error('Failed to fetch pick percentages')
    return res.json()
  }

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

      {/* Left — constraints */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 sticky top-6">
        <h2 className="text-base font-semibold text-white mb-4">
          Constraints
        </h2>
        <ConstraintsPanel
          onSubmit={handleSubmit}
          loading={loading}
          upcomingWeek={upcomingWeek}
        />
      </div>

      {/* Right — results */}
      <div className="min-h-[400px]">
        <h2 className="text-base font-semibold text-white mb-4">
          Results
        </h2>
        <ResultsPanel
          results={results}
          loading={loading}
          error={error}
        />
      </div>

    </div>
  )
}
