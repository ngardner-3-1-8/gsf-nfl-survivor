export default function ScheduleFilters({
  weekOptions, availableWeeks, selectedWeeks, onToggleWeek, onClearWeeks,
  teamSearch, onTeamSearch,
  showFilter, onShowFilter,
  activeView, onViewChange, columnViews,
  totalGames, filteredGames,
}) {
  // Accept the new week_options shape ([{week,label,double}]) and gracefully
  // fall back to a legacy plain-number list, so a version skew between this
  // component and its parent can never crash on an undefined array.
  const opts = (weekOptions && weekOptions.length ? weekOptions : (availableWeeks || []).map(w => ({
    week: w, label: String(w), double: false,
  }))) || []
  const views = columnViews || []
  const hasDouble = opts.some(o => o.double)
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex flex-col gap-3">

      {/* Row 1 — Week chips + Team search + Show filter */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-500 font-medium uppercase tracking-wide mr-1">
          Week
        </span>
        <button
          onClick={onClearWeeks}
          className={`text-xs px-3 py-1 rounded-full border transition-colors ${
            selectedWeeks.length === 0
              ? 'bg-green-600 text-white border-green-600'
              : 'border-gray-700 text-gray-400 hover:text-white hover:border-gray-500'
          }`}
        >
          All
        </button>
        {opts.map(opt => {
          const selected = selectedWeeks.includes(opt.week)
          return (
            <button
              key={opt.week}
              onClick={() => onToggleWeek(opt.week)}
              title={opt.double ? 'Double-pick week (2 picks required)' : undefined}
              className={`text-xs px-3 py-1 rounded-full border transition-colors inline-flex items-center gap-1 ${
                selected
                  ? 'bg-green-600 text-white border-green-600'
                  : opt.double
                    ? 'border-amber-600/60 text-amber-300 hover:text-white hover:border-amber-400'
                    : 'border-gray-700 text-gray-400 hover:text-white hover:border-gray-500'
              }`}
            >
              {opt.label}
              {opt.double && (
                <span className={`text-[10px] font-semibold leading-none px-1 py-0.5 rounded ${
                  selected ? 'bg-white/20 text-white' : 'bg-amber-500/20 text-amber-300'
                }`}>2×</span>
              )}
            </button>
          )
        })}
        {hasDouble && (
          <span className="text-[11px] text-amber-400/80 ml-1">2× = double-pick week</span>
        )}
      </div>

      {/* Row 2 — Team search + Show filter + Column view tabs */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">Team</span>
        <input
          type="text"
          value={teamSearch}
          onChange={e => onTeamSearch(e.target.value)}
          placeholder="Search team..."
          className="bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-3 py-1.5 w-40 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600"
        />

        <div className="w-px h-5 bg-gray-700" />

        <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">Show</span>
        {[
          { value: 'all', label: 'All games' },
          { value: 'favorites', label: 'Favorites only' },
          { value: 'home', label: 'Home only' },
          { value: 'away', label: 'Away only' },
        ].map(opt => (
          <button
            key={opt.value}
            onClick={() => onShowFilter(opt.value)}
            className={`text-xs px-3 py-1 rounded-full border transition-colors ${
              showFilter === opt.value
                ? 'bg-green-600 text-white border-green-600'
                : 'border-gray-700 text-gray-400 hover:text-white hover:border-gray-500'
            }`}
          >
            {opt.label}
          </button>
        ))}

        <div className="w-px h-5 bg-gray-700" />

        <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">View</span>
        {views.map(view => (
          <button
            key={view}
            onClick={() => onViewChange(view)}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              activeView === view
                ? 'bg-gray-700 text-white border-gray-600 font-medium'
                : 'border-gray-700 text-gray-400 hover:text-white hover:border-gray-500'
            }`}
          >
            {view}
          </button>
        ))}

        {/* Game count */}
        <span className="ml-auto text-xs text-gray-500">
          {filteredGames === totalGames
            ? `${totalGames} games`
            : `${filteredGames} of ${totalGames} games`}
        </span>
      </div>
    </div>
  )
}
