import { useState, useEffect } from 'react'
import { fetchContestYears, fetchContestData, fetchContestCharts } from '../../api/client'
import ContestHistorical from './ContestHistorical'
import ContestCurrent from './ContestCurrent'
import { AvailabilityBarChart, PickPctByWeekChart } from './ContestCharts'

export default function ContestView() {
  const [years, setYears] = useState([])
  const [activeSubTab, setActiveSubTab] = useState('historical')
  const [charts, setCharts] = useState(null)

  useEffect(() => {
    fetchContestYears()
      .then(d => setYears(d.years || []))
      .catch(() => {})
  }, [])

  // Load chart data for the current season
  useEffect(() => {
    if (activeSubTab !== 'current') return
    fetchContestCharts()
      .then(d => setCharts(d))
      .catch(() => setCharts(null))
  }, [activeSubTab])

  return (
    <div className="flex flex-col gap-4">
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
      {activeSubTab === 'current' && (
        <div className="flex flex-col gap-4">
          <ContestCurrent years={years} />
          {charts && <AvailabilityBarChart availability={charts.availability} />}
          {charts && <PickPctByWeekChart pickByWeek={charts.pick_by_week} />}
        </div>
      )}
    </div>
  )
}
