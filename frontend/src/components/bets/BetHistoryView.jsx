import { useState, useEffect } from 'react'
import { fetchBettingHistory } from '../../api/client'

const TIER_CONFIG = {
  S: { bg: 'bg-green-900/40', text: 'text-green-300', border: 'border-green-700' },
  A: { bg: 'bg-blue-900/40', text: 'text-blue-300', border: 'border-blue-700' },
  B: { bg: 'bg-yellow-900/40', text: 'text-yellow-300', border: 'border-yellow-700' },
}

const CATEGORY_COLORS = {
  Spread: 'text-purple-300',
  Moneyline: 'text-blue-300',
  Total: 'text-amber-300',
}

function plColor(v) {
  if (v == null) return 'text-gray-500'
  return v >= 0 ? 'text-green-400' : 'text-red-400'
}

function fmtPL(v) {
  if (v == null) return '—'
  const sign = v >= 0 ? '+' : ''
  return `${sign}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function fmtPct(v) {
  return v == null ? '—' : `${v}%`
}

// ── Record cell ──────────────────────────────────────────────
function RecordCell({ data }) {
  if (!data || (data.wins === 0 && data.losses === 0 && data.pushes === 0)) {
    return <span className="text-gray-600 text-xs">—</span>
  }
  return (
    <div className="flex flex-col">
      <span className="font-mono text-xs text-gray-200">
        {data.wins}-{data.losses}{data.pushes > 0 ? `-${data.pushes}` : ''}
      </span>
      <span className={`font-mono text-xs ${plColor(data.total_pl)}`}>
        {fmtPL(data.total_pl)}
      </span>
    </div>
  )
}

// ── By bet type table ────────────────────────────────────────
function ByBetTypeTable({ summary }) {
  if (!summary?.by_bet_type) return null

  const entries = Object.entries(summary.by_bet_type)
  // Separate flat and Kelly
  const flat = entries.filter(([, d]) => !d.is_kelly)
  const kelly = entries.filter(([, d]) => d.is_kelly)

  const renderRows = (list) => list.map(([label, d]) => {
    const settled = d.wins + d.losses
    return (
      <tr key={label} className="border-b border-gray-800/50 hover:bg-gray-800/20">
        <td className="px-4 py-2.5">
          <span className="text-white text-xs font-medium">{label}</span>
        </td>
        <td className="px-4 py-2.5">
          <span className={`text-xs ${CATEGORY_COLORS[d.category] || 'text-gray-400'}`}>
            {d.category}
          </span>
        </td>
        <td className="px-4 py-2.5 text-xs font-mono text-gray-300">
          {d.wins}-{d.losses}{d.pushes > 0 ? `-${d.pushes}` : ''}
          {d.no_bets > 0 && <span className="text-gray-600 ml-1">({d.no_bets} NB)</span>}
        </td>
        <td className="px-4 py-2.5 text-xs font-mono">
          <span className={
            d.win_pct == null ? 'text-gray-500' :
            d.win_pct >= 55 ? 'text-green-400' :
            d.win_pct >= 50 ? 'text-yellow-400' : 'text-red-400'
          }>
            {fmtPct(d.win_pct)}
          </span>
        </td>
        <td className="px-4 py-2.5 text-xs font-mono">
          <span className={plColor(d.total_pl)}>{fmtPL(d.total_pl)}</span>
        </td>
        <td className="px-4 py-2.5 text-xs font-mono">
          <span className={plColor(d.roi)}>{fmtPct(d.roi)}</span>
        </td>
      </tr>
    )
  })

  return (
    <div className="flex flex-col gap-4">
      {/* Flat unit bets */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <p className="text-white font-semibold text-sm">Flat Unit Betting ($100/bet)</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                {['Bet Type', 'Category', 'Record', 'Win%', 'P/L', 'ROI'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>{renderRows(flat)}</tbody>
          </table>
        </div>
      </div>

      {/* Kelly bets */}
      {kelly.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-white font-semibold text-sm">Kelly Criterion Betting (variable stake)</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  {['Bet Type', 'Category', 'Record', 'Win%', 'P/L', 'ROI'].map(h => (
                    <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>{renderRows(kelly)}</tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ── By tier grid ─────────────────────────────────────────────
function ByTierGrid({ summary }) {
  if (!summary?.by_tier) return null

  // Collect all bet labels that appear in any tier
  const allLabels = new Set()
  for (const tier of ['S', 'A', 'B']) {
    Object.keys(summary.by_tier[tier] || {}).forEach(l => allLabels.add(l))
  }
  const labels = [...allLabels]

  if (labels.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center h-40">
        <p className="text-gray-500 text-sm">No tier-classified bets for this period</p>
      </div>
    )
  }

  // Per-tier totals
  const tierTotals = {}
  for (const tier of ['S', 'A', 'B']) {
    let pl = 0, w = 0, l = 0
    for (const cell of Object.values(summary.by_tier[tier] || {})) {
      pl += cell.total_pl || 0
      w += cell.wins || 0
      l += cell.losses || 0
    }
    tierTotals[tier] = { pl, w, l }
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Tier summary cards */}
      <div className="grid grid-cols-3 gap-3">
        {['S', 'A', 'B'].map(tier => {
          const t = tierTotals[tier]
          const settled = t.w + t.l
          const cfg = TIER_CONFIG[tier]
          return (
            <div key={tier} className={`rounded-xl border p-4 ${cfg.bg} ${cfg.border}`}>
              <div className="flex items-center justify-between">
                <span className={`text-lg font-bold ${cfg.text}`}>Tier {tier}</span>
                <span className={`font-mono text-sm ${plColor(t.pl)}`}>{fmtPL(t.pl)}</span>
              </div>
              <p className="text-xs text-gray-400 mt-1 font-mono">
                {t.w}-{t.l} · {settled > 0 ? `${(t.w / settled * 100).toFixed(1)}%` : '—'}
              </p>
            </div>
          )
        })}
      </div>

      {/* Grid: bet types × tiers */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left px-4 py-2.5 text-xs font-medium text-gray-500">Bet Type</th>
                {['S', 'A', 'B'].map(tier => (
                  <th key={tier} className={`text-left px-4 py-2.5 text-xs font-medium ${TIER_CONFIG[tier].text}`}>
                    Tier {tier}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {labels.map(label => (
                <tr key={label} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                  <td className="px-4 py-2.5">
                    <span className="text-white text-xs font-medium">{label}</span>
                  </td>
                  {['S', 'A', 'B'].map(tier => (
                    <td key={tier} className="px-4 py-2.5">
                      <RecordCell data={summary.by_tier[tier]?.[label]} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────
export default function BetHistoryView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeYear, setActiveYear] = useState('total')
  const [view, setView] = useState('by_tier') // 'by_tier' | 'by_bet_type'

  useEffect(() => {
    fetchBettingHistory()
      .then(d => {
        setData(d)
        // Default to total tab
        setActiveYear('total')
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading betting history...</span>
    </div>
  )

  if (error) return (
    <div className="bg-red-950/50 border border-red-800 rounded-xl p-4">
      <p className="text-red-400 text-sm font-medium">Error loading betting history</p>
      <p className="text-red-300 text-sm mt-1">{error}</p>
    </div>
  )

  const years = data?.available_years || []
  const currentSummary = activeYear === 'total'
    ? data?.total
    : data?.by_year?.[activeYear]

  return (
    <div className="flex flex-col gap-4">

      {/* Year tabs + view toggle */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1.5 flex-wrap">
          <button
            onClick={() => setActiveYear('total')}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
              activeYear === 'total'
                ? 'bg-green-600 text-white border-green-600'
                : 'border-gray-700 text-gray-400 hover:text-white'
            }`}
          >
            All Years
          </button>
          {years.map(y => (
            <button
              key={y}
              onClick={() => setActiveYear(String(y))}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
                activeYear === String(y)
                  ? 'bg-green-600 text-white border-green-600'
                  : 'border-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {y}
            </button>
          ))}
        </div>

        {/* View toggle */}
        <div className="ml-auto flex items-center gap-1.5">
          <span className="text-xs text-gray-500">View</span>
          <button
            onClick={() => setView('by_tier')}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              view === 'by_tier'
                ? 'bg-gray-700 text-white border-gray-600'
                : 'border-gray-700 text-gray-400 hover:text-white'
            }`}
          >
            By Tier
          </button>
          <button
            onClick={() => setView('by_bet_type')}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              view === 'by_bet_type'
                ? 'bg-gray-700 text-white border-gray-600'
                : 'border-gray-700 text-gray-400 hover:text-white'
            }`}
          >
            By Bet Type
          </button>
        </div>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-white font-semibold text-sm">
          {activeYear === 'total' ? 'All-Time Performance' : `${activeYear} Season Performance`}
        </p>
        <p className="text-xs text-gray-500">
          {view === 'by_tier'
            ? 'Records and P/L grouped by edge tier (S/A/B)'
            : 'Records and P/L grouped by bet type'}
        </p>
      </div>

      {/* Content */}
      {!currentSummary ? (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center h-40">
          <p className="text-gray-500 text-sm">No data available for this period</p>
        </div>
      ) : view === 'by_tier' ? (
        <ByTierGrid summary={currentSummary} />
      ) : (
        <ByBetTypeTable summary={currentSummary} />
      )}

      {/* Methodology note */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-xs text-gray-500">
        <p className="font-medium text-gray-400 mb-1">About this data</p>
        <p>
          Performance is calculated from actual closing sportsbook lines and final scores.
          Tiers are assigned using the same edge thresholds as the Recommended Bets tab.
          Flat unit assumes $100 per bet (−110 juice on spreads/totals); Kelly uses the
          fractional Kelly stake computed at bet time. ROI = total P/L ÷ total amount risked.
        </p>
      </div>
    </div>
  )
}
