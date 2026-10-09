import { useState, useEffect } from 'react'
import { fetchRecommendedBets, fetchBettingHistory, fetchBetWeekRange } from '../../api/client'
import { useAvailableYears } from '../../hooks/useAvailableYears'
import YearSelector from '../ui/YearSelector'

// All 12 tracked bet types, grouped for display. Labels match the
// betting-history endpoint + weekly_10 analysis.
const BET_TYPE_ORDER = [
  'Sim Spread', 'GSF Spread', 'MP Spread', 'Consensus Spread',
  'Sim Moneyline', 'GSF Moneyline', 'MP Moneyline', 'Consensus Moneyline',
  'Sim Total',
  'Sim Spread (Kelly)', 'Sim Moneyline (Kelly)', 'Sim Total (Kelly)',
]
const BET_DISPLAY = {
  'Consensus Spread': 'Consensus Spread · MC+GSF agree',
  'Consensus Moneyline': 'Consensus ML · MC+GSF agree',
}
const CATEGORY_OF = {
  'Sim Spread': 'Spread', 'GSF Spread': 'Spread', 'MP Spread': 'Spread',
  'Consensus Spread': 'Spread', 'Sim Spread (Kelly)': 'Spread',
  'Sim Moneyline': 'Moneyline', 'GSF Moneyline': 'Moneyline', 'MP Moneyline': 'Moneyline',
  'Consensus Moneyline': 'Moneyline', 'Sim Moneyline (Kelly)': 'Moneyline',
  'Sim Total': 'Total', 'Sim Total (Kelly)': 'Total',
}

// Map a recommended-bet card to its betting-history label.
function betHistoryLabel(bet) {
  if (bet.model === 'GSF' && bet.bet_type === 'Spread') return 'GSF Spread'
  if (bet.model === 'Combined (MC+GSF)') return 'Consensus Spread'
  if (bet.model === 'Monte Carlo') {
    if (bet.bet_type === 'Spread') return 'Sim Spread'
    if (bet.bet_type === 'Moneyline') return 'Sim Moneyline'
    if (bet.bet_type === 'Total') return 'Sim Total'
  }
  return null
}

function winPctColor(v) {
  if (v == null) return 'text-gray-500'
  if (v >= 55) return 'text-green-400'
  if (v >= 50) return 'text-yellow-400'
  return 'text-red-400'
}

const RANGE_CLASS_STYLE = {
  strong:  { chip: 'bg-green-900/50 text-green-300 border border-green-700/50', label: 'Strong' },
  weak:    { chip: 'bg-red-900/50 text-red-300 border border-red-700/50',       label: 'Weak' },
  average: { chip: 'bg-gray-800 text-gray-400 border border-gray-700',          label: 'Average' },
}

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

function DetailRow({ label, val }) {
  if (val == null) return null
  return (
    <div className="flex justify-between gap-3">
      <span className="text-gray-500">{label}</span>
      <span className="text-white font-mono">{val}</span>
    </div>
  )
}

// Shows the sportsbook line + all model estimates for a bet, by type.
function BetDetails({ bet }) {
  const d = bet.details
  if (!d) return null
  const pct = v => (v == null ? null : `${(v * 100).toFixed(1)}%`)

  return (
    <div className="mt-2 pt-2 border-t border-gray-800 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
      {/* Sportsbook line — shown for every type */}
      {d.sportsbook_line != null && (
        <div className="flex justify-between gap-3 col-span-2">
          <span className="text-gray-400 font-medium">Sportsbook {bet.bet_type}</span>
          <span className="text-white font-mono font-medium">{d.sportsbook_line}</span>
        </div>
      )}

      {bet.bet_type === 'Spread' && (
        <>
          <DetailRow label="Sim mean"    val={d.sim_spread_mean} />
          <DetailRow label="Sim median"  val={d.sim_spread_median} />
          <DetailRow label="Monte Carlo" val={d.mc_spread} />
          <DetailRow label="Consensus"   val={d.consensus_spread} />
          <DetailRow label="MP"          val={d.mp_spread} />
          <DetailRow label="GSF"         val={d.gsf_spread} />
        </>
      )}

      {bet.bet_type === 'Total' && (
        <>
          <DetailRow label="Sim mean"    val={d.sim_total_mean} />
          <DetailRow label="Sim median"  val={d.sim_total_median} />
          <DetailRow label="Monte Carlo" val={d.mc_total} />
          <DetailRow label="P(Over)"     val={pct(d.sim_prob_over)} />
        </>
      )}

      {bet.bet_type === 'Moneyline' && (
        <>
          <DetailRow label="Implied (Book)"      val={pct(d.implied_sportsbook)} />
          <DetailRow label="Implied (Market)"    val={pct(d.implied_market)} />
          <DetailRow label="Implied (Consensus)" val={pct(d.implied_consensus)} />
          <DetailRow label="Implied (MP)"        val={pct(d.implied_mp)} />
          <DetailRow label="Implied (GSF)"       val={pct(d.implied_gsf)} />
        </>
      )}
    </div>
  )
}

// ── Lifetime + per-year hit-rate cards for every tracked bet type ──────────
function HitRateCards({ history }) {
  if (!history?.total?.by_bet_type) return null
  const total = history.total.by_bet_type
  const byYear = history.by_year || {}
  const years = (history.available_years || []).slice().sort((a, b) => b - a)
  const labels = BET_TYPE_ORDER.filter(l => total[l])
  if (!labels.length) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
      <p className="text-white font-semibold text-sm">Hit rate by bet type</p>
      <p className="text-gray-500 text-xs mt-0.5 mb-3">
        Lifetime and per-season win rate across all settled bets — updates as results come in
      </p>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3">
        {labels.map(label => {
          const s = total[label]
          const settled = (s.wins || 0) + (s.losses || 0)
          return (
            <div key={label} className="bg-gray-950/40 border border-gray-800 rounded-xl p-3">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs font-medium text-white leading-tight">
                  {BET_DISPLAY[label] || label}
                </span>
                <span className={`text-lg font-bold font-mono ${winPctColor(s.win_pct)}`}>
                  {s.win_pct != null ? `${s.win_pct}%` : '—'}
                </span>
              </div>
              <p className="text-xs text-gray-600 mt-0.5">
                {s.wins}-{s.losses}{s.pushes ? `-${s.pushes}P` : ''} · {settled} bets
                {s.roi != null && (
                  <span className={s.roi >= 0 ? ' text-green-500/80' : ' text-red-500/80'}>
                    {' '}· {s.roi >= 0 ? '+' : ''}{s.roi}% ROI
                  </span>
                )}
              </p>
              {years.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-800/80 flex flex-wrap gap-x-2 gap-y-0.5">
                  {years.map(y => {
                    const ys = byYear[String(y)]?.by_bet_type?.[label]
                    if (!ys || ys.win_pct == null) return null
                    return (
                      <span key={y} className="text-xs font-mono text-gray-500" title={`${ys.wins}-${ys.losses}`}>
                        {String(y).slice(2)}:
                        <span className={winPctColor(ys.win_pct)}> {Math.round(ys.win_pct)}%</span>
                      </span>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Auto-detected strong/average/weak week ranges per bet type ─────────────
function WeekRangeSection({ weekRange }) {
  const byType = weekRange?.by_bet_type
  if (!byType) return null
  const labels = BET_TYPE_ORDER.filter(l => byType[l])
  if (!labels.length) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
      <p className="text-white font-semibold text-sm">Best &amp; worst weeks of the season</p>
      <p className="text-gray-500 text-xs mt-0.5 mb-3">
        Win rate by part of the season vs each bet's own baseline, pooled across{' '}
        {(weekRange.seasons_used || []).length} seasons (weeks 1–{weekRange.max_week}).
        A stretch is flagged when it beats/trails the baseline by {weekRange.margin_pts}+ points.
      </p>
      <div className="flex flex-col gap-2.5">
        {labels.map(label => {
          const d = byType[label]
          return (
            <div key={label} className="flex items-start gap-3 flex-wrap border-b border-gray-800/60 pb-2.5 last:border-0">
              <div className="w-44 shrink-0">
                <p className="text-xs font-medium text-white">{BET_DISPLAY[label] || label}</p>
                <p className="text-xs text-gray-600">baseline {d.baseline_win_pct}% · {d.n} bets</p>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {d.ranges.map((r, i) => {
                  const st = RANGE_CLASS_STYLE[r.class] || RANGE_CLASS_STYLE.average
                  return (
                    <span key={i} className={`text-xs px-2 py-1 rounded-lg ${st.chip}`}
                      title={`${st.label} · ${r.n} bets`}>
                      <span className="font-medium">{r.label}</span>
                      {r.win_pct != null && <span className="font-mono"> · {r.win_pct}%</span>}
                    </span>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
      <p className="text-gray-600 text-xs mt-3">
        Green = beats baseline, red = trails it, gray = in line. Short one-week blips are smoothed out.
      </p>
    </div>
  )
}

export default function RecommendedBetsView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filterTier, setFilterTier] = useState('all')
  const [filterType, setFilterType] = useState('all')
  // Lifetime/per-year hit rates and the week-range analysis are season-agnostic,
  // so they load once and are non-fatal (sections just hide if unavailable).
  const [history, setHistory] = useState(null)
  const [weekRange, setWeekRange] = useState(null)

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

  // Hit-rate history + week-range analysis — fetched once, best-effort.
  useEffect(() => {
    fetchBettingHistory().then(setHistory).catch(() => setHistory(null))
    fetchBetWeekRange().then(setWeekRange).catch(() => setWeekRange(null))
  }, [])

  // Live lifetime win rate for a recommended bet: prefer the exact tier cell,
  // fall back to the bet type's overall lifetime rate. Always realized results.
  const liveTierWinRate = (bet) => {
    const label = betHistoryLabel(bet)
    if (!label || !history?.total) return null
    const tierCell = history.total.by_tier?.[bet.tier]?.[label]
    if (tierCell && tierCell.win_pct != null) {
      return { scope: `Tier ${bet.tier}`, win_pct: tierCell.win_pct,
               wins: tierCell.wins, losses: tierCell.losses }
    }
    const overall = history.total.by_bet_type?.[label]
    if (overall && overall.win_pct != null) {
      return { scope: 'Lifetime', win_pct: overall.win_pct,
               wins: overall.wins, losses: overall.losses }
    }
    return null
  }

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

      {/* Lifetime + per-year hit rate for every bet type */}
      <HitRateCards history={history} />

      {/* Auto-detected strong/weak week ranges (precomputed weekly) */}
      <WeekRangeSection weekRange={weekRange} />

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

              <BetDetails bet={bet} />

              {/* Context bar — LIVE lifetime win rate for this bet type at this
                  tier, computed from realized results (updates each week). */}
              {(() => {
                const wr = liveTierWinRate(bet)
                if (!wr) return null
                return (
                  <div className="mt-2 pt-2 border-t border-gray-800 flex items-center gap-2 text-xs flex-wrap">
                    <span className="text-gray-500">{wr.scope} win rate:</span>
                    <span className={`font-mono font-medium ${winPctColor(wr.win_pct)}`}>
                      {wr.win_pct}%
                    </span>
                    <span className="text-gray-600">({wr.wins}-{wr.losses})</span>
                    {bet.note && <span className="text-green-500/80">· MC+GSF agree</span>}
                  </div>
                )
              })()}
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
