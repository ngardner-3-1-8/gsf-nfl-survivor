import { useState, useEffect } from 'react'
import { getBets, addBet, updateBet, deleteBet } from '../../api/client'

const RESULT_CONFIG = {
  pending: { label: 'Pending', bg: 'bg-gray-800', text: 'text-gray-400' },
  won:     { label: 'Won',     bg: 'bg-green-900/60', text: 'text-green-300' },
  lost:    { label: 'Lost',    bg: 'bg-red-900/60',   text: 'text-red-300' },
  push:    { label: 'Push',    bg: 'bg-yellow-900/60', text: 'text-yellow-300' },
}

function calcProfit(wager, odds, result) {
  if (result === 'push') return 0
  if (result === 'lost') return -Math.abs(parseFloat(wager) || 0)
  if (result === 'won') {
    const o = parseFloat(odds)
    const w = parseFloat(wager) || 0
    if (isNaN(o)) return 0
    return o > 0 ? (w * o / 100) : (w * 100 / Math.abs(o))
  }
  return 0
}

function profitCell(wager, odds, result) {
  if (result === 'pending') return <span className="text-gray-500">—</span>
  const p = calcProfit(wager, odds, result)
  const formatted = (p >= 0 ? '+' : '') + '$' + Math.abs(p).toFixed(2)
  return (
    <span className={p > 0 ? 'text-green-400 font-medium' : p < 0 ? 'text-red-400 font-medium' : 'text-gray-400'}>
      {formatted}
    </span>
  )
}

// ── Username gate ─────────────────────────────────────────────
function UsernameGate({ onSet }) {
  const [val, setVal] = useState('')
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <p className="text-white font-semibold text-sm">Enter your name to track bets</p>
      <p className="text-gray-500 text-xs text-center max-w-xs">
        Your bets are saved to this name across all devices. Pick something memorable.
      </p>
      <div className="flex gap-2">
        <input
          type="text"
          value={val}
          onChange={e => setVal(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && val.trim() && onSet(val.trim())}
          placeholder="e.g. Nick"
          className="bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-4 py-2 w-48 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600"
        />
        <button
          onClick={() => val.trim() && onSet(val.trim())}
          className="bg-green-600 hover:bg-green-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
        >
          Continue
        </button>
      </div>
    </div>
  )
}

// ── Add Bet Form ──────────────────────────────────────────────
function AddBetForm({ onAdd, onCancel, weeks }) {
  const [form, setForm] = useState({
    away_team: '', home_team: '', week: '', circa_week: '',
    bet_type: 'Spread', pick: '', odds: '-110', wager: '',
    model: '', notes: '',
  })
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  const handleSubmit = () => {
    if (!form.away_team || !form.home_team || !form.pick || !form.wager) return
    onAdd({
      ...form,
      wager: parseFloat(form.wager),
      odds: form.odds,
    })
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-5 flex flex-col gap-4">
      <p className="text-white font-semibold text-sm">Add a bet</p>

      {/* Game */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Away Team</label>
          <input
            value={form.away_team}
            onChange={e => set('away_team', e.target.value)}
            placeholder="e.g. KC"
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Home Team</label>
          <input
            value={form.home_team}
            onChange={e => set('home_team', e.target.value)}
            placeholder="e.g. BAL"
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          />
        </div>
      </div>

      {/* Week + Bet type */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Week</label>
          <input
            value={form.circa_week || form.week}
            onChange={e => set('circa_week', e.target.value)}
            placeholder="e.g. Week 3"
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Bet Type</label>
          <select
            value={form.bet_type}
            onChange={e => set('bet_type', e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          >
            <option>Spread</option>
            <option>Moneyline</option>
            <option>Total</option>
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Model</label>
          <select
            value={form.model}
            onChange={e => set('model', e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          >
            <option value="">Manual</option>
            <option>Monte Carlo</option>
            <option>GSF</option>
            <option>Combined</option>
            <option>Massey-Peabody</option>
          </select>
        </div>
      </div>

      {/* Pick + Odds + Wager */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Pick</label>
          <input
            value={form.pick}
            onChange={e => set('pick', e.target.value)}
            placeholder="e.g. KC -3.5 or OVER"
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Odds</label>
          <input
            value={form.odds}
            onChange={e => set('odds', e.target.value)}
            placeholder="-110"
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Wager ($)</label>
          <input
            type="number"
            value={form.wager}
            onChange={e => set('wager', e.target.value)}
            placeholder="100"
            className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
          />
        </div>
      </div>

      {/* Notes */}
      <div>
        <label className="text-xs text-gray-400 uppercase tracking-wide block mb-1">Notes (optional)</label>
        <input
          value={form.notes}
          onChange={e => set('notes', e.target.value)}
          placeholder="e.g. S-tier edge 4.2, Kelly sizing"
          className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-green-600"
        />
      </div>

      <div className="flex gap-2 justify-end">
        <button
          onClick={onCancel}
          className="text-sm text-gray-400 hover:text-white px-4 py-2 rounded-lg border border-gray-700 transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          className="bg-green-600 hover:bg-green-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
        >
          Save Bet
        </button>
      </div>
    </div>
  )
}

// ── Main view ─────────────────────────────────────────────────
export default function MyBetsView() {
  const [username, setUsername] = useState(() => localStorage.getItem('gsf_username') || '')
  const [bets, setBets] = useState([])
  const [loading, setLoading] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [filterResult, setFilterResult] = useState('all')
  const [saving, setSaving] = useState(false)

  const handleSetUsername = (name) => {
    localStorage.setItem('gsf_username', name)
    setUsername(name)
  }

  useEffect(() => {
    if (!username) return
    setLoading(true)
    getBets(username)
      .then(d => setBets(d.bets || []))
      .catch(() => setBets([]))
      .finally(() => setLoading(false))
  }, [username])

  const handleAdd = async (bet) => {
    setSaving(true)
    try {
      const res = await addBet(username, bet)
      setBets(prev => [...prev, res.bet])
      setShowForm(false)
    } finally {
      setSaving(false)
    }
  }

  const handleResult = async (bet, result) => {
    const updated = await updateBet(username, bet.id, { result })
    setBets(prev => prev.map(b => b.id === bet.id ? updated.bet : b))
  }

  const handleDelete = async (betId) => {
    await deleteBet(username, betId)
    setBets(prev => prev.filter(b => b.id !== betId))
  }

  if (!username) return <UsernameGate onSet={handleSetUsername} />

  // Stats
  const settled = bets.filter(b => b.result !== 'pending')
  const wins = settled.filter(b => b.result === 'won').length
  const losses = settled.filter(b => b.result === 'lost').length
  const pushes = settled.filter(b => b.result === 'push').length
  const totalProfit = settled.reduce((sum, b) => sum + calcProfit(b.wager, b.odds, b.result), 0)
  const totalWagered = bets.reduce((sum, b) => sum + (parseFloat(b.wager) || 0), 0)
  const winRate = settled.length > 0 ? ((wins / (settled.length - pushes)) * 100).toFixed(1) : '—'

  const filtered = bets.filter(b => filterResult === 'all' || b.result === filterResult)
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))

  return (
    <div className="flex flex-col gap-4">

      {/* Header + stats */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 flex flex-col gap-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <p className="text-white font-semibold text-sm">My Bets — {username}</p>
            <button
              onClick={() => { localStorage.removeItem('gsf_username'); setUsername('') }}
              className="text-xs text-gray-600 hover:text-gray-400 mt-0.5"
            >
              Switch user
            </button>
          </div>
          <button
            onClick={() => setShowForm(true)}
            className="bg-green-600 hover:bg-green-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            + Add Bet
          </button>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {[
            { label: 'Record', value: settled.length > 0 ? `${wins}W-${losses}L${pushes > 0 ? `-${pushes}P` : ''}` : '—' },
            { label: 'Win Rate', value: winRate === '—' ? '—' : `${winRate}%` },
            { label: 'Total P/L', value: settled.length > 0 ? `${totalProfit >= 0 ? '+' : ''}$${Math.abs(totalProfit).toFixed(2)}` : '—', color: totalProfit > 0 ? 'text-green-400' : totalProfit < 0 ? 'text-red-400' : '' },
            { label: 'Total Wagered', value: `$${totalWagered.toFixed(2)}` },
            { label: 'Pending', value: bets.filter(b => b.result === 'pending').length },
          ].map(s => (
            <div key={s.label} className="bg-gray-800 rounded-xl p-3 text-center">
              <p className="text-xs text-gray-500 mb-1">{s.label}</p>
              <p className={`text-sm font-semibold ${s.color || 'text-white'}`}>{s.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Add bet form */}
      {showForm && (
        <AddBetForm
          onAdd={handleAdd}
          onCancel={() => setShowForm(false)}
        />
      )}

      {/* Filter tabs */}
      <div className="flex items-center gap-2 flex-wrap">
        {['all', 'pending', 'won', 'lost', 'push'].map(r => (
          <button
            key={r}
            onClick={() => setFilterResult(r)}
            className={`text-xs px-3 py-1 rounded-full border transition-colors capitalize ${
              filterResult === r
                ? 'bg-green-600 text-white border-green-600'
                : 'border-gray-700 text-gray-400 hover:text-white'
            }`}
          >
            {r === 'all' ? `All (${bets.length})` : `${r.charAt(0).toUpperCase() + r.slice(1)} (${bets.filter(b => b.result === r).length})`}
          </button>
        ))}
      </div>

      {/* Bets table */}
      {loading ? (
        <div className="flex items-center justify-center h-40 gap-3">
          <div className="w-6 h-6 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-400 text-sm">Loading bets...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl flex flex-col items-center justify-center h-40 gap-2">
          <p className="text-gray-500 text-sm">No bets yet</p>
          <button
            onClick={() => setShowForm(true)}
            className="text-green-400 text-xs hover:text-green-300"
          >
            Add your first bet →
          </button>
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  {['Week', 'Game', 'Type', 'Pick', 'Odds', 'Wager', 'P/L', 'Model', 'Result', ''].map(h => (
                    <th key={h} className="text-left px-4 py-2.5 text-xs font-medium text-gray-500 whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map(bet => (
                  <tr key={bet.id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                    <td className="px-4 py-3 text-xs text-gray-400">{bet.circa_week || `Wk ${bet.week}`}</td>
                    <td className="px-4 py-3">
                      <div className="text-xs text-white font-medium">{bet.away_team} @ {bet.home_team}</div>
                      {bet.notes && <div className="text-xs text-gray-500 mt-0.5">{bet.notes}</div>}
                    </td>
                    <td className="px-4 py-3 text-xs text-gray-400">{bet.bet_type}</td>
                    <td className="px-4 py-3 text-xs text-white font-medium">{bet.pick}</td>
                    <td className="px-4 py-3 text-xs font-mono text-gray-300">{bet.odds}</td>
                    <td className="px-4 py-3 text-xs font-mono text-gray-300">${parseFloat(bet.wager).toFixed(2)}</td>
                    <td className="px-4 py-3 text-xs font-mono">{profitCell(bet.wager, bet.odds, bet.result)}</td>
                    <td className="px-4 py-3 text-xs text-gray-500">{bet.model || '—'}</td>
                    <td className="px-4 py-3">
                      <select
                        value={bet.result}
                        onChange={e => handleResult(bet, e.target.value)}
                        className={`text-xs rounded-lg px-2 py-1 border focus:outline-none focus:ring-1 focus:ring-green-600 ${
                          RESULT_CONFIG[bet.result]?.bg || 'bg-gray-800'
                        } ${RESULT_CONFIG[bet.result]?.text || 'text-gray-400'} border-gray-700`}
                      >
                        <option value="pending">Pending</option>
                        <option value="won">Won</option>
                        <option value="lost">Lost</option>
                        <option value="push">Push</option>
                      </select>
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleDelete(bet.id)}
                        className="text-gray-600 hover:text-red-400 text-xs transition-colors"
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
