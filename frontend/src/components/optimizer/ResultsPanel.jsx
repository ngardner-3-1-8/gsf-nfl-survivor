import PickCard from './PickCard'

// Build a lookup of all pick percentages per week across all solutions
// so PickCard can rank each team's pick% within its week
function buildWeeklyPickPcts(solutions) {
  const weekly = {}
  solutions.forEach(solution => {
    solution.forEach(pick => {
      if (!weekly[pick.week]) weekly[pick.week] = []
      weekly[pick.week].push(pick.pick_pct)
    })
  })
  // Sort descending so rank = index + 1
  Object.keys(weekly).forEach(week => {
    weekly[week].sort((a, b) => b - a)
  })
  return weekly
}

export default function ResultsPanel({ results, loading, error }) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <div className="w-8 h-8 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
        <p className="text-gray-400 text-sm">Running optimizer...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
        <p className="text-red-400 text-sm font-medium">Error</p>
        <p className="text-red-300 text-sm mt-1">{error}</p>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-2 text-center">
        <span className="text-4xl">🏈</span>
        <p className="text-gray-400 text-sm">
          Set your constraints and click<br />
          <span className="text-white font-medium">Run Optimizer</span> to see results
        </p>
      </div>
    )
  }

  if (!results.feasible && results.ev_solutions.length === 0) {
    return (
      <div className="bg-yellow-950/50 border border-yellow-800 rounded-xl p-4">
        <p className="text-yellow-400 text-sm font-medium">No solution found</p>
        <p className="text-yellow-300 text-sm mt-1">{results.message}</p>
        <p className="text-gray-400 text-xs mt-2">
          Try relaxing some constraints — for example, turn off a few scheduling filters or expand the week range.
        </p>
      </div>
    )
  }

  const allSolutions = [...(results.ev_solutions || []), ...(results.ranking_solutions || [])]
  const allWeeklyPickPcts = buildWeeklyPickPcts(allSolutions)

  return (
    <div className="flex flex-col gap-6">
      {results.ev_solutions.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
            EV-Optimized Solutions
          </h3>
          <div className="flex flex-col gap-3">
            {results.ev_solutions.map((solution, i) => (
              <PickCard
                key={i}
                solution={solution}
                index={i}
                label="EV"
                allWeeklyPickPcts={allWeeklyPickPcts}
              />
            ))}
          </div>
        </div>
      )}

      {results.ranking_solutions.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
            Win%-Optimized Solutions
          </h3>
          <div className="flex flex-col gap-3">
            {results.ranking_solutions.map((solution, i) => (
              <PickCard
                key={i}
                solution={solution}
                index={i}
                label="Win%"
                allWeeklyPickPcts={allWeeklyPickPcts}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
