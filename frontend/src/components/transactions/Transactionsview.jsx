import { useState, useEffect } from 'react'
import { fetchTransactions, fetchTransactionYears } from '../../api/client'

const TEAM_FULL_NAMES = {
  ARI: 'Arizona Cardinals', ATL: 'Atlanta Falcons', BAL: 'Baltimore Ravens',
  BUF: 'Buffalo Bills', CAR: 'Carolina Panthers', CHI: 'Chicago Bears',
  CIN: 'Cincinnati Bengals', CLE: 'Cleveland Browns', DAL: 'Dallas Cowboys',
  DEN: 'Denver Broncos', DET: 'Detroit Lions', GB: 'Green Bay Packers',
  HOU: 'Houston Texans', IND: 'Indianapolis Colts', JAX: 'Jacksonville Jaguars',
  KC: 'Kansas City Chiefs', LA: 'Los Angeles Rams', LAC: 'Los Angeles Chargers',
  LV: 'Las Vegas Raiders', MIA: 'Miami Dolphins', MIN: 'Minnesota Vikings',
  NE: 'New England Patriots', NO: 'New Orleans Saints', NYG: 'New York Giants',
  NYJ: 'New York Jets', PHI: 'Philadelphia Eagles', PIT: 'Pittsburgh Steelers',
  SEA: 'Seattle Seahawks', SF: 'San Francisco 49ers', TB: 'Tampa Bay Buccaneers',
  TEN: 'Tennessee Titans', WAS: 'Washington Commanders',
}

const TYPE_COLORS = {
  'Trade (player)':       'bg-purple-900/40 text-purple-300',
  'Trade (pick)':         'bg-purple-900/30 text-purple-400',
  'Free Agent Signing':   'bg-green-900/40 text-green-300',
  'Released / Retired':   'bg-red-900/40 text-red-300',
  'Draft Pick':           'bg-blue-900/40 text-blue-300',
}

function deltaColor(v) {
  if (v == null) return 'text-gray-500'
  return v >= 0 ? 'text-green-400' : 'text-red-400'
}

function fmtDelta(v) {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}`
}

export default function TransactionsView() {
  const [years, setYears] = useState([])
  const [selectedYear, setSelectedYear] = useState(null)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [view, setView] = useState('leaderboard') // 'leaderboard' | 'log'
  const [teamFilter, setTeamFilter] = useState('all')
  const [typeFilter, setTypeFilter] = useState('all')

  // Load available years
  useEffect(() => {
    fetchTransactionYears()
      .then(d => {
        const yrs = d.years || []
        setYears(yrs)
        if (yrs.length > 0) setSelectedYear(yrs[0])
        else { setLoading(false); setError('No transaction data available yet') }
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  // Load data for selected year
  useEffect(() => {
    if (!selectedYear) return
    setLoading(true)
    setError(null)
    fetchTransactions(selectedYear)
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedYear])

  const leaderboard = data?.leaderboard || []
  const transactions = data?.transactions || []

  const filteredTx = transactions.filter(t => {
    if (teamFilter !== 'all' && t.to_team !== teamFilter && t.from_team !== teamFilter) return false
    if (typeFilter !== 'all' && t.type !== typeFilter) return false
    return true
  })

  const txTypes = [...new Set(transactions.map(t => t.type))].filter(Boolean)

  const yearBar = (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl px-4 py-3 flex items-center gap-3 flex-wrap">
      <span className="text-xs text-gray-500 uppercase tracking-wide">Season</span>
      <div className="flex gap-1.5 flex-wrap">
        {years.map(y => (
          <button
            key={y}
            onClick={() => setSelectedYear(y)}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors font-medium ${
              selectedYear === y
                ? 'bg-green-600 text-white border-green-600'
                : 'border-gray-700 text-gray-400 hover:text-white'
            }`}
          >
            {y}
          </button>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-1.5">
        <button
          onClick={() => setView('leaderboard')}
          className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
            view === 'leaderboard'
              ? 'bg-gray-700 text-white border-gray-600'
              : 'border-gray-700 text-gray-400 hover:text-white'
          }`}
        >
          Team Leaderboard
        </button>
        <button
          onClick={() => setView('log')}
          className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
            view === 'log'
              ? 'bg-gray-700 text-white border-gray-600'
              : 'border-gray-700 text-gray-400 hover:text-white'
          }`}
        >
          Transaction Log
        </button>
      </div>
    </div>
  )

  if (loading) return (
    <div className="flex flex-col gap-4">
      {years.length > 0 && yearBar}
      <div className="flex items-center justify-center h-64 gap-3">
        <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
        <span className="text-gray-400 text-sm">Loading transactions...</span>
      </div>
    </div>
  )

  if (error) return (
    <div className="flex flex-col gap-4">
      {years.length > 0 && yearBar}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center h-40">
        <p className="text-gray-500 text-sm">{error}</p>
      </div>
    </div>
  )

  // Max absolute delta for bar scaling
  const maxAbs = Math.max(1, ...leaderboard.map(t => Math.abs(t.net_delta || 0)))

  return (
    <div className="flex flex-col gap-4">

      {yearBar}

      {/* Methodology note */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-xs text-gray-500">
        <p className="font-medium text-gray-400 mb-1">How transactions are valued</p>
        <p>
          Each move is quantified in power-rating points. Skill players are valued by
          their prior-season EPA; defenders and linemen use position baselines (public
          defensive EPA is unreliable); draft picks use an expected-value-by-slot curve;
          departures subtract value. Net delta estimates how much each team gained or lost
          this offseason. These are directional estimates, not exact projections.
        </p>
      </div>

      {view === 'leaderboard' ? (
        /* ── Team leaderboard ── */
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-white font-semibold text-sm">
              {selectedYear} Offseason Net Point Deltas
            </p>
            <p className="text-gray-500 text-xs mt-0.5">
              Ranked by total estimated points added through all transactions
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  {['Rank', 'Team', 'Net Δ', '', 'Offense', 'Defense', 'Added', 'Lost', 'Moves'].map((h, i) => (
                    <th key={i} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((t, i) => (
                  <tr key={t.team} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                    <td className="px-4 py-2.5 text-xs font-mono text-gray-400">{i + 1}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex flex-col">
                        <span className="text-white text-xs font-semibold">{t.team}</span>
                        <span className="text-gray-600 text-xs">{TEAM_FULL_NAMES[t.team] || ''}</span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5">
                      <span className={`font-mono text-xs font-semibold ${deltaColor(t.net_delta)}`}>
                        {fmtDelta(t.net_delta)}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 w-32">
                      {/* Bar viz */}
                      <div className="relative h-3 w-28 bg-gray-800/50 rounded-full overflow-hidden">
                        <div
                          className={`absolute top-0 bottom-0 ${t.net_delta >= 0 ? 'left-1/2 bg-green-500/70' : 'right-1/2 bg-red-500/70'}`}
                          style={{ width: `${Math.abs(t.net_delta || 0) / maxAbs * 50}%` }}
                        />
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-600" />
                      </div>
                    </td>
                    <td className="px-4 py-2.5">
                      <span className={`font-mono text-xs ${deltaColor(t.offense_delta)}`}>
                        {fmtDelta(t.offense_delta)}
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <span className={`font-mono text-xs ${deltaColor(t.defense_delta)}`}>
                        {fmtDelta(t.defense_delta)}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs text-green-400/80">
                      +{(t.additions || 0).toFixed(1)}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs text-red-400/80">
                      {(t.subtractions || 0).toFixed(1)}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs text-gray-400">
                      {t.num_moves}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        /* ── Transaction log ── */
        <div className="flex flex-col gap-3">
          {/* Filters */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-gray-500">Team</span>
            <select
              value={teamFilter}
              onChange={e => setTeamFilter(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-green-600"
            >
              <option value="all">All teams</option>
              {Object.keys(TEAM_FULL_NAMES).map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <span className="text-xs text-gray-500 ml-2">Type</span>
            <select
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-white text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-green-600"
            >
              <option value="all">All types</option>
              {txTypes.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <span className="text-xs text-gray-600 ml-auto">{filteredTx.length} transactions</span>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    {['Type', 'Player', 'Pos', 'From', 'To', 'Value'].map(h => (
                      <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredTx.map((t, i) => (
                    <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                      <td className="px-4 py-2.5">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded ${TYPE_COLORS[t.type] || 'bg-gray-800 text-gray-300'}`}>
                          {t.type}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-white text-xs">{t.player}</td>
                      <td className="px-4 py-2.5 text-gray-400 text-xs">{t.position || '—'}</td>
                      <td className="px-4 py-2.5 text-gray-400 text-xs font-mono">{t.from_team || '—'}</td>
                      <td className="px-4 py-2.5 text-gray-400 text-xs font-mono">{t.to_team || '—'}</td>
                      <td className="px-4 py-2.5">
                        <span className={`font-mono text-xs font-medium ${deltaColor(t.value)}`}>
                          {fmtDelta(t.value)}
                        </span>
                        {t.epa != null && (
                          <span className="text-gray-600 text-xs ml-1">({t.epa} EPA)</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
