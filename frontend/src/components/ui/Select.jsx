export default function Select({ label, value, onChange, options }) {
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-xs text-gray-400 font-medium uppercase tracking-wide">
          {label}
        </label>
      )}
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="
          bg-gray-900 border border-gray-700 text-white text-sm rounded-lg
          px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-600
          focus:border-transparent cursor-pointer
        "
      >
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  )
}
