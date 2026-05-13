export default function ScheduleFilters({
  availableWeeks, selectedWeeks, onToggleWeek, onClearWeeks,
  teamSearch, onTeamSearch,
  showFilter, onShowFilter,
  activeView, onViewChange, columnViews,
  totalGames, filteredGames,
}) {
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
        {availableWeeks.map(week => (
          <button
            key={week}
            onClick={() => onToggleWeek(week)}
            className={`text-xs px-3 py-1 rounded-full border transition-colors ${
              selectedWeeks.includes(week)
                ? 'bg-green-600 text-white border-green-600'
                : 'border-gray-700 text-gray-400 hover:text-white hover:border-gray-500'
            }`}
          >
            {week}
          </button>
        ))}
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
        {columnViews.map(view => (
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
