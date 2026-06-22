import { useState, useEffect } from 'react'
import { fetchRecommendedBets } from '../../api/client'
import { useAvailableYears } from '../../hooks/useAvailableYears'
import YearSelector from '../ui/YearSelector'

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

  const { years, selectedYear, setSelectedYear, isHistorical } = useAvailableYears()

  // Reload bets when year changes
  useEffect(() => {
    if (!selectedYear) return
    setLoading(true)
    setError(null)
    fetchRecommendedBets(selectedYear)
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedYear])

  const bets = data?.bets || []
  const counts = data?.counts || {}

  const filtered = bets.filter(b => {
    if (filterTier !== 'all' && b.tier !== filterTier) return false
    if (filterType !== 'all' && b.bet_type !== filterType) return false
    return true
  })

  const yearSelectorBar = (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-4 flex-wrap">
      <YearSelector
        years={years}
        selectedYear={selectedYear}
        onChange={setSelectedYear}
      />
      {isHistorical && (
        <span className="text-xs text-amber-400">
          📋 {selectedYear} season — showing backtest results
        </span>
      )}
    </div>
  )

  if (loading) return (
    <div className="flex flex-col gap-4">
      {yearSelectorBar}
      <div className="flex items-center justify-center h-64 gap-3">
        <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
        <span className="text-gray-400 text-sm">Loading recommendations...</span>
      </div>
    </div>
  )

  if (error) return (
    <div className="flex flex-col gap-4">
      {yearSelectorBar}
      <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
        <p className="text-red-400 text-sm font-medium">Error</p>
        <p className="text-red-300 text-sm mt-1">{error}</p>
      </div>
    </div>
  )

  return (
    <div className="flex flex-col gap-4">

      {/* Year selector */}
      {yearSelectorBar}

      {/* Summary bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div>
          <p className="text-white font-semibold text-sm">
            {isHistorical ? 'Backtest' : 'Recommended'} Bets — {data?.target_year}
            {!isHistorical && ` Week ${data?.upcoming_week}`}
          </p>
          <p className="text-gray-500 text-xs mt-0.5">
            {isHistorical
              ? `Full ${data?.target_year} season — what the model recommended with actual results`
              : 'Based on historical edge profitability analysis'}
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

      {/* Season backtest summary — historical mode only */}
      {isHistorical && data?.season_summary && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-white font-semibold text-sm">
              Season Backtest Results — {data.target_year}
            </p>
            <p className="text-gray-500 text-xs mt-0.5">
              Actual Win/Loss and P/L across all {data.target_year} bets
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  {['Model', 'Record', 'Win%', 'Total P/L', 'Per Bet Avg'].map(h => (
                    <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.season_summary).map(([key, s]) => {
                  const settled = s.wins + s.losses + s.pushes
                  const winPct = (settled - s.pushes) > 0
                    ? ((s.wins / (settled - s.pushes)) * 100).toFixed(1)
                    : '—'
                  const perBet = settled > 0
                    ? (s.total_pl / settled).toFixed(2)
                    : '—'
                  return (
                    <tr key={key} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                      <td className="px-4 py-2.5 text-white text-xs font-medium">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </td>
                      <td className="px-4 py-2.5 text-xs font-mono text-gray-300">
                        {s.wins}W-{s.losses}L{s.pushes > 0 ? `-${s.pushes}P` : ''}
                        {s.no_bets > 0 && (
                          <span className="text-gray-600 ml-1">({s.no_bets} NB)</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-xs font-mono">
                        <span className={
                          parseFloat(winPct) >= 55 ? 'text-green-400' :
                          parseFloat(winPct) >= 50 ? 'text-yellow-400' :
                          'text-red-400'
                        }>
                          {winPct}{winPct !== '—' ? '%' : ''}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-xs font-mono">
                        <span className={s.total_pl >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {s.total_pl >= 0 ? '+' : ''}${Math.abs(s.total_pl).toLocaleString()}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-xs font-mono text-gray-400">
                        {perBet !== '—' ? `$${perBet}` : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Bet cards */}
      {filtered.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center h-40">
          <p className="text-gray-500 text-sm">No bets match your filters</p>
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

                {/* Right — pick + edge + historical result */}
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
                  {/* Historical W/L result inline */}
                  {isHistorical && bet.win_loss && (
                    <p className={`text-xs font-semibold mt-1 ${
                      bet.win_loss === 'Win'  ? 'text-green-400' :
                      bet.win_loss === 'Loss' ? 'text-red-400'   :
                      bet.win_loss === 'Push' ? 'text-yellow-400': 'text-gray-500'
                    }`}>
                      {bet.win_loss}
                      {bet.pnl != null && (
                        <span className="ml-1 font-normal">
                          ({bet.pnl >= 0 ? '+' : ''}${Number(bet.pnl).toFixed(2)})
                        </span>
                      )}
                    </p>
                  )}
                </div>
              </div>

              <WagerInfo bet={bet} />

              {/* Context bar */}
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
