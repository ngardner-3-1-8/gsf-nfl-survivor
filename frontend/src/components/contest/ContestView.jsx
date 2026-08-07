import { useState, useEffect } from 'react'
import { fetchContestYears, fetchContestCharts } from '../../api/client'
import ContestHistorical from './ContestHistorical'
import ContestCurrent from './ContestCurrent'
import { AvailabilityBarChart, PickPctByWeekChart } from './ContestCharts'

export default function ContestView() {
  const [years, setYears] = useState([])
  const [selectedYear, setSelectedYear] = useState(null)
  const [activeSubTab, setActiveSubTab] = useState('historical')
  const [charts, setCharts] = useState(null)
  const [chartsLoading, setChartsLoading] = useState(false)

  // Load available years; default the selection to the most recent one
  useEffect(() => {
    fetchContestYears()
      .then(d => {
        const ys = d.years || []
        setYears(ys)
        if (ys.length && selectedYear == null) {
          setSelectedYear(Math.max(...ys.map(Number)))
        }
      })
      .catch(() => {})
  }, [])

  // Fetch charts for the selected year (works for any year that has data).
  // The current, dataless season simply returns no charts and renders nothing.
  useEffect(() => {
    if (selectedYear == null) return
    setChartsLoading(true)
    setCharts(null)
    fetchContestCharts(selectedYear)
      .then(d => setCharts(d))
      .catch(() => setCharts(null))
      .finally(() => setChartsLoading(false))
  }, [selectedYear])

  const chartBlock = (
    <>
      {chartsLoading && (
        <div className="flex items-center justify-center h-40 gap-3">
          <div className="w-5 h-5 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-500 text-sm">Loading charts…</span>
        </div>
      )}
      {!chartsLoading && charts?.availability?.length > 0 && (
        <AvailabilityBarChart availability={charts.availability} />
      )}
      {!chartsLoading && charts?.pick_by_week?.length > 0 && (
        <PickPctByWeekChart pickByWeek={charts.pick_by_week} />
      )}
    </>
  )

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

      {activeSubTab === 'historical' && (
        <div className="flex flex-col gap-4">
          <ContestHistorical
            years={years}
            selectedYear={selectedYear}
            onYearChange={setSelectedYear}
          />
          {chartBlock}
        </div>
      )}

      {activeSubTab === 'current' && (
        <div className="flex flex-col gap-4">
          <ContestCurrent years={years} />
          {chartBlock}
        </div>
      )}
    </div>
  )
}
