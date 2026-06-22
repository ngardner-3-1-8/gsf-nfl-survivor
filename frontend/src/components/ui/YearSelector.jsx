export default function YearSelector({ years, selectedYear, onChange }) {
  if (!years || years.length <= 1) return null

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-gray-500 uppercase tracking-wide">Season</span>
      <div className="flex gap-1 flex-wrap">
        {years.map(y => (
          <button
            key={y.year}
            onClick={() => onChange(y.year)}
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors font-medium ${
              selectedYear === y.year
                ? 'bg-green-600 text-white border-green-600'
                : y.is_current
                  ? 'border-green-700 text-green-400 hover:bg-green-900/30'
                  : 'border-gray-700 text-gray-400 hover:text-white'
            }`}
          >
            {y.label}
          </button>
        ))}
      </div>
      {selectedYear && years.find(y => y.year === selectedYear && !y.is_current) && (
        <span className="text-xs text-amber-500 ml-1">
          📋 Historical — showing actual results
        </span>
      )}
    </div>
  )
}
