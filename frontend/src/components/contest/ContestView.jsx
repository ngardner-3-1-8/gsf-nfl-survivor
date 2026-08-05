import { useState, useEffect } from 'react'
import { fetchContestYears, fetchContestData } from '../../api/client'
import ContestHistorical from './ContestHistorical'
import ContestCurrent from './ContestCurrent'

export default function ContestView() {
  const [years, setYears] = useState([])
  const [activeSubTab, setActiveSubTab] = useState('historical')

  useEffect(() => {
    fetchContestYears()
      .then(d => setYears(d.years || []))
      .catch(() => {})
  }, [])

  return (
    <div className="flex flex-col gap-4">
      {/* Sub-tabs */}
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
      {activeSubTab === 'current'    && <ContestCurrent years={years} />}
    </div>
  )
}

import {
  BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, Tooltip,
  LineChart, Line, CartesianGrid, Legend,
} from 'recharts'

// ── Primary team colors (keyed by abbreviation) ─────────────────────────────
export const TEAM_COLORS = {
  ARI: '#97233F', ATL: '#A71930', BAL: '#241773', BUF: '#00338D',
  CAR: '#0085CA', CHI: '#0B162A', CIN: '#FB4F14', CLE: '#311D00',
  DAL: '#003594', DEN: '#FB4F14', DET: '#0076B6', GB: '#203731',
  HOU: '#03202F', IND: '#002C5F', JAX: '#101820', KC: '#E31837',
  LA: '#003594', LAC: '#0080C6', LV: '#000000', MIA: '#008E97',
  MIN: '#4F2683', NE: '#002244', NO: '#D3BC8D', NYG: '#0B2265',
  NYJ: '#125740', PHI: '#004C54', PIT: '#FFB612', SEA: '#002244',
  SF: '#AA0000', TB: '#D50A0A', TEN: '#0C2340', WAS: '#5A1414',
}

// A few teams have very dark primaries; give their bars a lighter edge so
// they're visible on a dark background.
const NEEDS_OUTLINE = new Set(['CHI', 'JAX', 'LV', 'NE', 'HOU', 'BAL', 'NYG', 'TEN', 'SEA'])

// ── 1. Current Availability by Team ─────────────────────────────────────────
// Shows how many entries still have each team available to pick.
// Expects: availability = [{ team: 'KC', available: 4120 }, ...]
export function AvailabilityBarChart({ availability }) {
  const data = useMemo(
    () => [...(availability || [])].sort((a, b) => b.available - a.available),
    [availability]
  )

  if (!data.length) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
      <p className="text-white font-semibold text-sm mb-1">Current Availability by Team</p>
      <p className="text-gray-500 text-xs mb-4">
        How many alive entries still have each team available to pick
      </p>
      <ResponsiveContainer width="100%" height={Math.max(260, data.length * 20)}>
        <BarChart data={data} layout="vertical"
          margin={{ top: 0, right: 24, bottom: 0, left: 8 }}>
          <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={{ stroke: '#374151' }} tickLine={false} />
          <YAxis type="category" dataKey="team" width={44}
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
            contentStyle={{ background: '#111827', border: '1px solid #374151',
              borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#e5e7eb' }}
            formatter={(v) => [v.toLocaleString(), 'entries']}
          />
          <Bar dataKey="available" radius={[0, 4, 4, 0]}>
            {data.map(d => (
              <Cell key={d.team}
                fill={TEAM_COLORS[d.team] || '#4b5563'}
                stroke={NEEDS_OUTLINE.has(d.team) ? '#6b7280' : 'none'}
                strokeWidth={NEEDS_OUTLINE.has(d.team) ? 1 : 0} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── 2. Team Pick % by Week ──────────────────────────────────────────────────
// Line chart: each selected team's pick share across weeks.
// Expects: pickByWeek = [{ week: 1, KC: 0.12, BUF: 0.08, ... }, ...]
//   teams = ['KC','BUF',...] the teams to draw (default: top movers)
export function PickPctByWeekChart({ pickByWeek, teams }) {
  const drawnTeams = useMemo(() => {
    if (teams && teams.length) return teams
    // Auto-pick the 6 teams with the highest peak pick% across the season
    if (!pickByWeek?.length) return []
    const peaks = {}
    pickByWeek.forEach(row => {
      Object.keys(row).forEach(k => {
        if (k === 'week') return
        peaks[k] = Math.max(peaks[k] || 0, row[k] || 0)
      })
    })
    return Object.entries(peaks).sort((a, b) => b[1] - a[1]).slice(0, 6).map(e => e[0])
  }, [pickByWeek, teams])

  if (!pickByWeek?.length || !drawnTeams.length) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
      <p className="text-white font-semibold text-sm mb-1">Team Pick % by Week</p>
      <p className="text-gray-500 text-xs mb-4">
        Share of the field picking each team, week by week
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={pickByWeek} margin={{ top: 8, right: 16, bottom: 0, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="week" tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={{ stroke: '#374151' }} tickLine={false}
            label={{ value: 'Week', position: 'insideBottom', offset: -2,
              fill: '#6b7280', fontSize: 11 }} />
          <YAxis tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={false} tickLine={false}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151',
              borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#e5e7eb' }}
            labelFormatter={w => `Week ${w}`}
            formatter={(v, name) => [`${(v * 100).toFixed(1)}%`, name]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {drawnTeams.map(t => (
            <Line key={t} type="monotone" dataKey={t}
              stroke={TEAM_COLORS[t] || '#4b5563'} strokeWidth={2}
              dot={{ r: 2 }} activeDot={{ r: 4 }} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

