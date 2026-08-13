import { useState, useEffect } from 'react'
import ConstraintsPanel from './ConstraintsPanel'
import ResultsPanel from './ResultsPanel'
import { runOptimizer, fetchSplashContests } from '../../api/client'

export default function SplashOptimizerView() {
  const [contests, setContests] = useState([])
  const [selectedContest, setSelectedContest] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchSplashContests()
      .then(d => {
        const cs = d.contests || []
        setContests(cs)
        if (cs.length) setSelectedContest(cs[0].key)
      })
      .catch(e => setError(e.message))
  }, [])

  const contest = contests.find(c => c.key === selectedContest)

  const handleSubmit = async (constraints) => {
    if (!selectedContest) return
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      // Splash: send contest key + its double-pick weeks. Backend also injects
      // the contest's manual pick data. NFL weeks (no holiday insertion).
      const data = await runOptimizer({
        ...constraints,
        contest: selectedContest,
        double_pick_weeks: contest?.double_pick_weeks || [],
      })
      setResults(data)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Contest selector */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-3 flex-wrap">
        <span className="text-xs text-gray-500 uppercase tracking-wide">Contest</span>
        <div className="flex gap-1.5 flex-wrap">
          {contests.map(c => (
            <button
              key={c.key}
              onClick={() => setSelectedContest(c.key)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
                selectedContest === c.key
                  ? 'bg-cyan-600 text-white border-cyan-600'
                  : 'border-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {c.display_name}
            </button>
          ))}
        </div>
      </div>

      {/* Contest summary — entries, survivors, double-pick weeks */}
      {contest && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-6 flex-wrap text-sm">
          <div>
            <p className="text-gray-500 text-xs">Total Entries</p>
            <p className="text-white font-mono">{contest.entries?.toLocaleString() ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Surviving Entries</p>
            <p className="text-white font-mono">{contest.survivors?.toLocaleString() ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Double-pick weeks</p>
            <p className="text-cyan-400 font-mono">
              {contest.double_pick_weeks?.length
                ? contest.double_pick_weeks.join(', ')
                : 'none set'}
            </p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Entry Fee</p>
            <p className="text-white font-mono">${contest.entry_fee?.toLocaleString() ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Total Prizes</p>
            <p className="text-white font-mono">${contest.total_prize?.toLocaleString() ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Average Entry Value</p>
            <p className="text-white font-mono">{contest.total_prize && contest.entries? `$${Math.round(contest.total_prize / contest.entries).toLocaleString()}`: '—'}</p>
          </div>
          <div className="ml-auto max-w-xs">
            <p className="text-gray-600 text-xs">
              NFL weeks · on double-pick weeks the optimizer selects two teams
              that must both win.
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
            upcomingWeek={1}
            weekOptions={Array.from({ length: 18 }, (_, i) => ({
              week: i + 1, label: `Week ${i + 1}`,
            }))}
          />
        </div>
        <div className="min-h-[400px]">
          <h2 className="text-base font-semibold text-white mb-4">Results</h2>
          {error && (
            <div className="bg-red-950/50 border border-red-800 rounded-xl p-4 mb-4">
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}
          <ResultsPanel results={results} loading={loading} error={error} allPickPcts={{}} />
        </div>
      </div>
    </div>
  )
}
