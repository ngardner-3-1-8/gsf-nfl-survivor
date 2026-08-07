import { useState, useEffect } from 'react'
import { fetchContestYears, fetchContestCharts } from '../../api/client'
import ContestHistorical from './ContestHistorical'
import ContestCurrent from './ContestCurrent'
import { AvailabilityBarChart, PickPctByWeekChart } from './ContestCharts'

export default function ContestView() {
  const [years, setYears] = useState([])
  const [activeSubTab, setActiveSubTab] = useState('historical')

  // Current Season chart controls
  const [chartYear, setChartYear] = useState(null)
  const [throughWeek, setThroughWeek] = useState(null)
  const [maxWeek, setMaxWeek] = useState(null)
  const [charts, setCharts] = useState(null)
  const [chartsLoading, setChartsLoading] = useState(false)

  useEffect(() => {
    fetchContestYears()
      .then(d => {
        const ys = d.years || []
        setYears(ys)
        if (ys.length && chartYear == null) {
          setChartYear(Math.max(...ys.map(Number)))
        }
      })
      .catch(() => {})
  }, [])

  // Fetch charts for the selected year + "as of week".
  // Passing through_week scopes BOTH charts to that point in the season.
  useEffect(() => {
    if (activeSubTab !== 'current' || chartYear == null) return
    setChartsLoading(true)
    setCharts(null)
    fetchContestCharts(chartYear, throughWeek)
      .then(d => {
        setCharts(d)
        if (d?.max_week) setMaxWeek(d.max_week)
        // If no week chosen yet, default to the latest available week
        if (throughWeek == null && d?.through_week) setThroughWeek(d.through_week)
      })
      .catch(() => setCharts(null))
      .finally(() => setChartsLoading(false))
  }, [activeSubTab, chartYear, throughWeek])

  // Reset week when the year changes so we don't carry a stale week across seasons
  const handleYearChange = (y) => {
    setChartYear(y)
    setThroughWeek(null)
    setMaxWeek(null)
  }

  const weekButtons = maxWeek
    ? Array.from({ length: maxWeek }, (_, i) => i + 1)
    : []

  const currentSeasonCharts = (
    <div className="flex flex-col gap-4">
      {/* Year + As-of-Week selectors */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-3 flex-wrap">
        <span className="text-xs text-gray-500 uppercase tracking-wide">Season</span>
        <div className="flex gap-1.5 flex-wrap">
          {years.map(y => (
            <button
              key={y}
              onClick={() => handleYearChange(y)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
                chartYear === y
                  ? 'bg-green-600 text-white border-green-600'
                  : 'border-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {y}
            </button>
          ))}
        </div>

        {weekButtons.length > 0 && (
          <>
            <span className="text-xs text-gray-500 uppercase tracking-wide ml-2">As of week</span>
            <div className="flex gap-1 flex-wrap">
              {weekButtons.map(w => (
                <button
                  key={w}
                  onClick={() => setThroughWeek(w)}
                  className={`text-xs px-2.5 py-1.5 rounded-lg border transition-colors font-mono ${
                    throughWeek === w
                      ? 'bg-gray-700 text-white border-gray-600'
                      : 'border-gray-700 text-gray-500 hover:text-white'
                  }`}
                >
                  {w}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

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

      {!chartsLoading && !charts && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center h-40">
          <p className="text-gray-500 text-sm">
            No chart data for this season yet
          </p>
        </div>
      )}
    </div>
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

      {activeSubTab === 'historical' && <ContestHistorical years={years} />}
      {activeSubTab === 'current' && (
        <div className="flex flex-col gap-4">
          <ContestCurrent years={years} />
          {currentSeasonCharts}
        </div>
      )}
    </div>
  )
}
