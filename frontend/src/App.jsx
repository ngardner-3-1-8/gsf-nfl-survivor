import { useState, useEffect } from 'react'
import { fetchLastUpdated } from './api/client'
import OptimizerView from './components/optimizer/OptimizerView'
import ScheduleView from './components/schedule/ScheduleView'
import RankingsView from './components/rankings/RankingsView'
import RecommendedBetsView from './components/bets/RecommendedBetsView'
import MyBetsView from './components/bets/MyBetsView'

const ComingSoon = ({ name }) => (
  <div className="flex items-center justify-center h-64">
    <p className="text-gray-500 text-sm">{name} coming soon...</p>
  </div>
)

const TABS = [
  { id: 'optimizer',     label: 'Optimizer' },
  { id: 'schedule',      label: 'Schedule' },
  { id: 'rankings',      label: 'Rankings' },
  { id: 'bets',          label: 'Bets' },
  { id: 'ev-calc',       label: 'EV Calc' },
  { id: 'contest',       label: 'Contest' },
  { id: 'transactions',  label: 'Transactions' },
  { id: 'analytics',     label: 'Analytics' },
  { id: 'faq',           label: 'FAQ' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('optimizer')
  const [lastUpdated, setLastUpdated] = useState(null)
  const [betsSubTab, setBetsSubTab] = useState('recommended')  // ← moved inside

  useEffect(() => {
    fetchLastUpdated()
      .then(data => setLastUpdated(data))
      .catch(() => {})
  }, [])

  const renderTab = () => {
    switch (activeTab) {
      case 'optimizer':    return <OptimizerView />
      case 'schedule':     return <ScheduleView />
      case 'rankings':     return <RankingsView />
      case 'bets':
        return (
          <div className="flex flex-col gap-4">
            <div className="flex gap-1 border-b border-gray-800 -mb-4 pb-0">
              {[
                { id: 'recommended', label: 'Recommended Bets' },
                { id: 'my-bets',     label: 'My Bets' },
                { id: 'history',     label: 'History & Performance' },
              ].map(t => (
                <button
                  key={t.id}
                  onClick={() => setBetsSubTab(t.id)}
                  className={`text-sm px-4 py-3 border-b-2 transition-colors ${
                    betsSubTab === t.id
                      ? 'border-green-500 text-white font-medium'
                      : 'border-transparent text-gray-500 hover:text-white'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            {betsSubTab === 'recommended' && <RecommendedBetsView />}
            {betsSubTab === 'my-bets'     && <MyBetsView />}
            {betsSubTab === 'history'     && (
              <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
                History & Performance coming next...
              </div>
            )}
          </div>
        )
      case 'ev-calc':      return <ComingSoon name="EV Calculator" />
      case 'contest':      return <ComingSoon name="Contest Analytics" />
      case 'transactions': return <ComingSoon name="Transactions" />
      case 'analytics':    return <ComingSoon name="Analytics" />
      case 'faq':          return <ComingSoon name="FAQ" />
      default:             return <OptimizerView />
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <nav className="border-b border-gray-800 bg-gray-900 sticky top-0 z-10">
        <div className="max-w-screen-2xl mx-auto px-6 flex items-center h-14">

          {/* Logo */}
          <div className="flex items-center gap-2 mr-6 flex-shrink-0">
            <span className="font-bold text-lg">🏈</span>
            <span className="font-semibold text-white tracking-tight hidden sm:block">
              GSF Survivor
            </span>
          </div>

          {/* Tabs — scrollable on small screens */}
          <div className="flex items-center gap-1 overflow-x-auto scrollbar-hide flex-1">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  text-sm font-medium h-14 border-b-2 transition-colors px-3 whitespace-nowrap flex-shrink-0
                  ${activeTab === tab.id
                    ? 'border-green-500 text-white'
                    : 'border-transparent text-gray-500 hover:text-white'}
                `}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Timestamp */}
          {lastUpdated && (
            <div className="ml-4 flex-shrink-0 flex flex-col items-end text-xs text-gray-500">
              <span>
                <span className="text-gray-600">Wk {lastUpdated.upcoming_week} · Sim </span>
                {lastUpdated.sim_updated}
              </span>
              {lastUpdated.mp_updated && lastUpdated.mp_updated !== 'Unknown' && (
                <span>
                  <span className="text-gray-600">MP </span>
                  {lastUpdated.mp_updated}
                </span>
              )}
            </div>
          )}
        </div>
      </nav>

      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        {renderTab()}
      </main>
    </div>
  )
}
