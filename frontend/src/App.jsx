import OptimizerView from './components/optimizer/OptimizerView'
import { useState, useEffect } from 'react'
import { fetchLastUpdated } from './api/client'

export default function App() {
  const [activeTab, setActiveTab] = useState('optimizer')
  const [lastUpdated, setLastUpdated] = useState(null)

  useEffect(() => {
    fetchLastUpdated()
      .then(data => setLastUpdated(data))
      .catch(() => {})
  }, [])

  return (
    <div className="min-h-screen bg-gray-950 text-white">

      {/* Top navigation bar */}
      <nav className="border-b border-gray-800 bg-gray-900 sticky top-0 z-10">
        <div className="max-w-screen-2xl mx-auto px-6 flex items-center gap-8 h-14">

          <div className="flex items-center gap-2 mr-4">
            <span className="font-bold text-lg">🏈</span>
            <span className="font-semibold text-white tracking-tight">
              Generic Sports Fan Survivor
            </span>
          </div>

          {['optimizer', 'schedule', 'analytics'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`
                text-sm font-medium capitalize h-14 border-b-2 transition-colors px-1
                ${activeTab === tab
                  ? 'border-green-500 text-white'
                  : 'border-transparent text-gray-500 hover:text-white'}
              `}
            >
              {tab}
            </button>
          ))}
            {lastUpdated && (
              <div className="ml-auto flex flex-col items-end text-xs text-gray-500">
                <span>
                  <span className="text-gray-600">
                    Week {lastUpdated.upcoming_week} · Sim updated{' '}
                  </span>
                  {lastUpdated.sim_updated}
                </span>
                <span>
                  <span className="text-gray-600">MP Rankings updated{' '}</span>
                  {lastUpdated.mp_updated && lastUpdated.mp_updated !== 'Unknown'
                    ? lastUpdated.mp_updated
                    : 'Not yet uploaded this week'}
                </span>
              </div>
            )}
        </div>
      </nav>

      {/* Page content */}
      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        {activeTab === 'optimizer' && <OptimizerView />}
        {activeTab === 'schedule' && (
          <div className="text-gray-500 text-sm">Schedule view coming soon...</div>
        )}
        {activeTab === 'analytics' && (
          <div className="text-gray-500 text-sm">Analytics view coming soon...</div>
        )}
      </main>

    </div>
  )
}
