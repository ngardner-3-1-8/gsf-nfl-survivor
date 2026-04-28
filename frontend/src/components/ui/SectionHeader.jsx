export default function SectionHeader({ title }) {
  return (
    <div className="flex items-center gap-2 mb-3 mt-5 first:mt-0">
      <div className="h-px flex-1 bg-gray-800" />
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-widest whitespace-nowrap">
        {title}
      </span>
      <div className="h-px flex-1 bg-gray-800" />
    </div>
  )
}
