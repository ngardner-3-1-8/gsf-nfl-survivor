export default function Toggle({ label, description, checked, onChange }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2">
      <div className="flex-1">
        <div className="text-sm text-white font-medium">{label}</div>
        {description && (
          <div className="text-xs text-gray-500 mt-0.5">{description}</div>
        )}
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`
          relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full
          border-2 border-transparent transition-colors duration-200
          focus:outline-none focus:ring-2 focus:ring-green-600 focus:ring-offset-2
          focus:ring-offset-gray-900
          ${checked ? 'bg-green-600' : 'bg-gray-700'}
        `}
      >
        <span
          className={`
            pointer-events-none inline-block h-4 w-4 rounded-full bg-white
            shadow transform transition-transform duration-200
            ${checked ? 'translate-x-4' : 'translate-x-0'}
          `}
        />
      </button>
    </div>
  )
}
