import { useState, useEffect } from 'react'
import { fetchRecommendedBets } from '../../api/client'
import { useAvailableYears } from '../../hooks/useAvailableYears'
import YearSelector from '../ui/YearSelector'

// Inside the component, replace the existing year-unaware fetch:
const { years, selectedYear, setSelectedYear, isHistorical } = useAvailableYears()

// Add year to all fetchSchedule and fetchWeeks calls:
useEffect(() => {
  if (!selectedYear) return
  fetchWeeks(selectedYear).then(...)
}, [selectedYear])

useEffect(() => {
  if (!selectedWeek || !selectedYear) return
  fetchSchedule(selectedWeek.week, selectedYear).then(...)
}, [selectedWeek, selectedYear])

// Add the YearSelector to the header bar JSX:
<YearSelector years={years} selectedYear={selectedYear} onChange={setSelectedYear} />

const TIER_CONFIG = {
  S: { label: 'S', bg: 'bg-green-900/60', text: 'text-green-300', border: 'border-green-700', desc: 'Highest confidence' },
  A: { label: 'A', bg: 'bg-blue-900/60', text: 'text-blue-300', border: 'border-blue-700', desc: 'Strong signal' },
  B: { label: 'B', bg: 'bg-yellow-900/60', text: 'text-yellow-300', border: 'border-yellow-700', desc: 'Worth watching' },
}

const BET_TYPE_COLORS = {
  'Spread': 'bg-purple-900/40 text-purple-300',
  'Moneyline': 'bg-blue-900/40 text-blue-300',
  'Total': 'bg-amber-900/40 text-amber-300',
}

function TierBadge({ tier }) {
  const cfg = TIER_CONFIG[tier] || TIER_CONFIG.B
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded border ${cfg.bg} ${cfg.text} ${cfg.border}`}>
      {cfg.label}
    </span>
  )
}

function WagerInfo({ bet }) {
  if (!bet.unit_wager && !bet.kelly_wager) return null
  return (
    <div className="mt-2 flex flex-wrap gap-3 text-xs">
      {bet.unit_wager != null && (
        <span className="text-gray-400">
          Unit: <span className="text-white font-medium">${Number(bet.unit_wager).toFixed(2)}</span>
          {bet.unit_to_win != null && (
            <span className="text-green-400"> → ${Number(bet.unit_to_win).toFixed(2)}</span>
          )}
        </span>
      )}
      {bet.kelly_wager != null && (
        <span className="text-gray-400">
          Kelly: <span className="text-white font-medium">${Number(bet.kelly_wager).toFixed(2)}</span>
          {bet.kelly_to_win != null && (
            <span className="text-green-400"> → ${Number(bet.kelly_to_win).toFixed(2)}</span>
          )}
        </span>
      )}
    </div>
  )
}

export default function RecommendedBetsView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filterTier, setFilterTier] = useState('all')
  const [filterType, setFilterType] = useState('all')

  useEffect(() => {
    fetchRecommendedBets()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading recommendations...</span>
    </div>
  )

  if (error) return (
    <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
      <p className="text-red-400 text-sm font-medium">Error</p>
      <p className="text-red-300 text-sm mt-1">{error}</p>
    </div>
  )

  const bets = data?.bets || []
  const counts = data?.counts || {}

  const filtered = bets.filter(b => {
    if (filterTier !== 'all' && b.tier !== filterTier) return false
    if (filterType !== 'all' && b.bet_type !== filterType) return false
    return true
  })

  return (
    <div className="flex flex-col gap-4">

      {/* Summary bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div>
          <p className="text-white font-semibold text-sm">
            Recommended Bets — Week {data?.upcoming_week} {data?.target_year}
          </p>
          <p className="text-gray-500 text-xs mt-0.5">
            Based on historical edge profitability analysis
          </p>
        </div>

        {/* Tier counts */}
        <div className="flex gap-3">
          {['S', 'A', 'B'].map(t => (
            <div key={t} className={`text-center px-3 py-1.5 rounded-lg border ${TIER_CONFIG[t].bg} ${TIER_CONFIG[t].border}`}>
              <p className={`text-lg font-bold ${TIER_CONFIG[t].text}`}>{counts[t] || 0}</p>
              <p className="text-xs text-gray-500">Tier {t}</p>
            </div>
          ))}
        </div>

        {/* Filters */}
        <div className="ml-auto flex items-center gap-2 flex-wrap">
          <span className="text-xs text-gray-500">Tier</span>
          {['all', 'S', 'A', 'B'].map(t => (
            <button
              key={t}
              onClick={() => setFilterTier(t)}
              className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                filterTier === t
                  ? 'bg-green-600 text-white border-green-600'
                  : 'border-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t === 'all' ? 'All' : `Tier ${t}`}
            </button>
          ))}
          <div className="w-px h-4 bg-gray-700" />
          <span className="text-xs text-gray-500">Type</span>
          {['all', 'Spread', 'Moneyline', 'Total'].map(t => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                filterType === t
                  ? 'bg-green-600 text-white border-green-600'
                  : 'border-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t === 'all' ? 'All' : t}
            </button>
          ))}
        </div>
      </div>

      {/* Bet cards */}
      {filtered.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center h-40">
          <p className="text-gray-500 text-sm">No bets match your filters this week</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3">
          {filtered.map((bet, i) => (
            <div
              key={i}
              className={`bg-gray-900 border rounded-xl p-4 ${
                bet.tier === 'S' ? 'border-green-800/60' :
                bet.tier === 'A' ? 'border-blue-800/60' : 'border-gray-800'
              }`}
            >
              <div className="flex items-start justify-between gap-4 flex-wrap">

                {/* Left — game info */}
                <div className="flex items-start gap-3">
                  <TierBadge tier={bet.tier} />
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-white font-semibold text-sm">
                        {bet.away_team} @ {bet.home_team}
                      </span>
                      <span className="text-gray-500 text-xs">
                        {bet.circa_week || `Wk ${bet.week}`}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded ${BET_TYPE_COLORS[bet.bet_type] || 'bg-gray-800 text-gray-300'}`}>
                        {bet.bet_type}
                      </span>
                      <span className="text-xs text-gray-400">{bet.model}</span>
                      {bet.note && (
                        <span className="text-xs text-green-400 font-medium">★ {bet.note}</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right — pick + edge */}
                <div className="text-right">
                  <p className="text-white font-bold text-base">{bet.pick}</p>
                  {bet.direction && (
                    <p className="text-xs text-amber-400 font-medium">{bet.direction}</p>
                  )}
                  <p className="text-xs text-gray-400 mt-0.5">
                    Edge: <span className="text-green-400 font-medium">
                      +{bet.edge}{bet.edge_unit || ' pts'}
                    </span>
                  </p>
                </div>
              </div>

              <WagerInfo bet={bet} />

              {/* Historical context bar */}
              <div className="mt-2 pt-2 border-t border-gray-800 flex items-center gap-4 text-xs text-gray-600 flex-wrap">
                {bet.model === 'Monte Carlo' && bet.bet_type === 'Spread' && (
                  <>
                    {parseFloat(bet.edge) >= 4.0 && <span>Historical: 69.5% win rate at edge ≥ 4.0</span>}
                    {parseFloat(bet.edge) >= 1.0 && parseFloat(bet.edge) < 2.0 && <span>Historical: 72.0% win rate at edge 1.0–2.0</span>}
                  </>
                )}
                {bet.model === 'Monte Carlo' && bet.bet_type === 'Moneyline' && (
                  <>
                    {parseFloat(bet.edge) >= 20 && <span>Historical: 70.0% win rate at edge ≥ 20%</span>}
                    {parseFloat(bet.edge) >= 15 && parseFloat(bet.edge) < 20 && <span>Historical: 76.5% win rate at edge 15–20%</span>}
                    {parseFloat(bet.edge) >= 10 && parseFloat(bet.edge) < 15 && <span>Historical: 68.4% win rate at edge 10–15%</span>}
                  </>
                )}
                {bet.model === 'Monte Carlo' && bet.bet_type === 'Total' && (
                  <>
                    {parseFloat(bet.edge) >= 5.0 && <span>Historical: 62.1% win rate at edge ≥ 5.0</span>}
                    {parseFloat(bet.edge) >= 3.0 && parseFloat(bet.edge) < 5.0 && <span>Historical: 61.8% win rate at edge 3.0–5.0</span>}
                  </>
                )}
                {bet.model === 'GSF' && <span>Historical: 64.4% win rate at GSF edge 2.0–3.0</span>}
                {bet.note && <span className="text-green-600">72.8% win rate when MC+GSF agree</span>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Methodology note */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-xs text-gray-500">
        <p className="font-medium text-gray-400 mb-1">Recommendation methodology</p>
        <p>Tier S: MC Spread ≥4.0pt edge, MC ML ≥15% edge, MC Under ≥5.0pt edge. Tier A: MC Spread 1.0-2.0pt, MC Total ≥3.0pt, GSF Spread 2.0-3.0pt only, combined MC+GSF agreement. GSF Moneyline excluded entirely — negative historical profit at all edge tiers.</p>
      </div>
    </div>
  )
}
