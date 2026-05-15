import { useState, useEffect } from 'react'
import { fetchContestYears, fetchContestData } from '../../api/client'
import ContestHistorical from './ContestHistorical'
import ContestCurrent from './ContestCurrent'

export default function ContestView() {
  const [years, setYears] = useState([])
  const [activeSubTab, setActiveSubTab] = useState('historical')

  useEffect(() => {
    fetchContestYears()
      .then(d => setYears(d.years || []))
      .catch(() => {})
  }, [])

  return (
    <div className="flex flex-col gap-4">
      {/* Sub-tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        {[
          { id: 'historical', label: 'Historical Analysis' },
          { id: 'current',    label: 'Current Season' },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setActiveSubTab(t.id)}
            className={`text-sm px-4 py-3 border-b-2 transition-colors ${
              activeSubTab === t.id
                ? 'border-green-500 text-white font-medium'
                : 'border-transparent text-gray-500 hover:text-white'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeSubTab === 'historical' && <ContestHistorical years={years} />}
      {activeSubTab === 'current'    && <ContestCurrent years={years} />}
    </div>
  )
}
