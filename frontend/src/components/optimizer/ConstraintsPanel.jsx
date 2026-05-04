import { useState, useEffect } from 'react'
import Toggle from '../ui/Toggle'
import Select from '../ui/Select'
import SectionHeader from '../ui/SectionHeader'


const OBJECTIVE_OPTIONS = [
  { value: 'consensus', label: 'Consensus (EV + Win Odds)' },
  { value: 'sportsbook', label: 'Sportsbook (EV + Win Odds)' },
  { value: 'mp', label: 'Massey-Peabody (EV + Win Odds)' },
  { value: 'gsf', label: 'Generic Sports Fan (EV + Win Odds)' },
  { value: 'sim', label: 'Simulation (EV + Win Odds)' },
]

const SOLUTIONS_OPTIONS = [
  { value: 1,   label: '1 solution' },
  { value: 5,   label: '5 solutions' },
  { value: 10,  label: '10 solutions' },
  { value: 25,  label: '25 solutions' },
  { value: 50,  label: '50 solutions' },
  { value: 100, label: '100 solutions' },
]

const FAVORED_OPTIONS = [
  { value: 'sportsbook', label: 'Sportsbook' },
  { value: 'mp',         label: 'Massey-Peabody' },
  { value: 'gsf',        label: 'Generic Sports Fan' },
  { value: 'sim',        label: 'Simulation' },
  { value: 'consensus',  label: 'Consensus' },
  { value: 'all',        label: 'All Models' },
]

export default function ConstraintsPanel({ onSubmit, loading, upcomingWeek, weekOptions }) {
  const [objective, setObjective] = useState('consensus')
  const [numSolutions, setNumSolutions] = useState(1)
  const [startWeek, setStartWeek] = useState(upcomingWeek || 1)
  const [endWeek, setEndWeek] = useState(20)

  const [mustBeFavored, setMustBeFavored] = useState(false)
  const [favoredQualifier, setFavoredQualifier] = useState('sportsbook')

  const [scheduling, setScheduling] = useState({
    avoid_away_short_rest: false,
    avoid_away_divisional: false,
    avoid_3_in_10: false,
    avoid_4_in_17: false,
    avoid_cumulative_rest: false,
    avoid_thursday_all: false,
    avoid_thursday_away: false,
    avoid_back_to_back_away: false,
    avoid_international: false,
    avoid_weekly_rest_disadvantage: false,
    avoid_travel_disadvantage: false,
    avoid_close_divisional: false,
    min_div_spread: 3.0,
    avoid_away_close: false,
    min_away_spread: 3.0,
  })

  const [bayesian, setBayesian] = useState({
    mp_bayesian_all_metrics: false,
    mp_bayesian_preseason_and_current: false,
    mp_bayesian_current_and_adjusted: false,
    gsf_bayesian_adjusted: false,
    gsf_bayesian_preseason_and_current: false,
    gsf_bayesian_current_and_adjusted: false,
    sportsbook_bayesian_preseason_and_current: false,
    sim_bayesian_preseason_and_current: false,
    consensus_bayesian_preseason_and_current: false,
    bayesian_require_all: false,
  })

  const [prohibitedTeams, setProhibitedTeams] = useState('')
  const [requiredPicks, setRequiredPicks] = useState('')
  const [prohibitedWeeklyPicks, setProhibitedWeeklyPicks] = useState('')

  // Update week range defaults when weekOptions loads
  useEffect(() => {
    if (upcomingWeek) setStartWeek(upcomingWeek)
    if (weekOptions?.length > 0) {
      setEndWeek(weekOptions[weekOptions.length - 1].week)
    }
  }, [upcomingWeek, weekOptions])

  const toggleScheduling = (key) => {
    setScheduling(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const toggleBayesian = (key) => {
    setBayesian(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const handleSubmit = () => {
    const prohibited = prohibitedTeams
      .split(',')
      .map(t => t.trim())
      .filter(Boolean)

    const required = {}
    requiredPicks.split('\n').forEach(line => {
      const [team, week] = line.split(':').map(s => s.trim())
      if (team && week && !isNaN(week)) {
        required[team] = parseInt(week)
      }
    })

    const prohibitedWeekly = {}
    prohibitedWeeklyPicks.split('\n').forEach(line => {
      const colonIdx = line.indexOf(':')
      if (colonIdx === -1) return
      const team = line.slice(0, colonIdx).trim()
      const weeks = line.slice(colonIdx + 1).split(',')
        .map(w => parseInt(w.trim()))
        .filter(w => !isNaN(w))
      if (team && weeks.length > 0) {
        prohibitedWeekly[team] = weeks
      }
    })

    onSubmit({
      objective,
      number_solutions: numSolutions,
      start_week: parseInt(startWeek),
      end_week: parseInt(endWeek),
      must_be_favored: mustBeFavored,
      favored_qualifier: favoredQualifier,
      prohibited_teams: prohibited,
      required_picks: required,
      prohibited_weekly_picks: prohibitedWeekly,
      scheduling: {
        ...scheduling,
        ...bayesian,
      },
    })
  }

  return (
    <div className="flex flex-col gap-1">

      {/* Objective */}
      <SectionHeader title="Objective" />
      <Select
        label="Optimize by"
        value={objective}
        onChange={setObjective}
        options={OBJECTIVE_OPTIONS}
      />
      <div className="mt-3">
        <Select
          label="Number of solutions"
          value={numSolutions}
          onChange={v => setNumSolutions(Number(v))}
          options={SOLUTIONS_OPTIONS}
        />
      </div>

      {/* Week range */}
      <SectionHeader title="Week Range" />
      <div className="grid grid-cols-2 gap-3">
        <Select
          label="Start week"
          value={startWeek}
          onChange={v => setStartWeek(Number(v))}
          options={(weekOptions || []).map(w => ({
            value: w.week,
            label: w.label,
          }))}
        />
        <Select
          label="End week"
          value={endWeek}
          onChange={v => setEndWeek(Number(v))}
          options={(weekOptions || []).map(w => ({
            value: w.week,
            label: w.label,
          }))}
        />
      </div>

      {/* Must be favored */}
      <SectionHeader title="Favored" />
      <Toggle
        label="Must be favored"
        description="Only pick teams favored to win"
        checked={mustBeFavored}
        onChange={setMustBeFavored}
      />
      {mustBeFavored && (
        <div className="mt-2">
          <Select
            label="Favored according to"
            value={favoredQualifier}
            onChange={setFavoredQualifier}
            options={FAVORED_OPTIONS}
          />
        </div>
      )}

      {/* Scheduling constraints */}
      <SectionHeader title="Scheduling" />
      <Toggle
        label="Avoid Thursday Night Football"
        description="Exclude all TNF games"
        checked={scheduling.avoid_thursday_all}
        onChange={() => toggleScheduling('avoid_thursday_all')}
      />
      <Toggle
        label="Avoid Away TNF"
        description="Exclude away teams in TNF only"
        checked={scheduling.avoid_thursday_away}
        onChange={() => toggleScheduling('avoid_thursday_away')}
      />
      <Toggle
        label="Avoid away short rest"
        description="Away team on less than 7 days rest"
        checked={scheduling.avoid_away_short_rest}
        onChange={() => toggleScheduling('avoid_away_short_rest')}
      />
      <Toggle
        label="Avoid 3 games in 10 days"
        description="Team playing their 3rd game in 10 days"
        checked={scheduling.avoid_3_in_10}
        onChange={() => toggleScheduling('avoid_3_in_10')}
      />
      <Toggle
        label="Avoid 4 games in 17 days"
        checked={scheduling.avoid_4_in_17}
        onChange={() => toggleScheduling('avoid_4_in_17')}
      />
      <Toggle
        label="Avoid back-to-back away games"
        checked={scheduling.avoid_back_to_back_away}
        onChange={() => toggleScheduling('avoid_back_to_back_away')}
      />
      <Toggle
        label="Avoid international games"
        description="London, Munich, Madrid"
        checked={scheduling.avoid_international}
        onChange={() => toggleScheduling('avoid_international')}
      />
      <Toggle
        label="Avoid weekly rest disadvantage"
        checked={scheduling.avoid_weekly_rest_disadvantage}
        onChange={() => toggleScheduling('avoid_weekly_rest_disadvantage')}
      />
      <Toggle
        label="Avoid cumulative rest disadvantage"
        checked={scheduling.avoid_cumulative_rest}
        onChange={() => toggleScheduling('avoid_cumulative_rest')}
      />
      <Toggle
        label="Avoid travel disadvantage"
        checked={scheduling.avoid_travel_disadvantage}
        onChange={() => toggleScheduling('avoid_travel_disadvantage')}
      />
      <Toggle
        label="Avoid away divisional games"
        checked={scheduling.avoid_away_divisional}
        onChange={() => toggleScheduling('avoid_away_divisional')}
      />
      <Toggle
        label="Avoid close divisional matchups"
        description={`Only pick if favored by ${scheduling.min_div_spread}+ points`}
        checked={scheduling.avoid_close_divisional}
        onChange={() => toggleScheduling('avoid_close_divisional')}
      />
      <Toggle
        label="Avoid close away matchups"
        description={`Only pick away teams favored by ${scheduling.min_away_spread}+ points`}
        checked={scheduling.avoid_away_close}
        onChange={() => toggleScheduling('avoid_away_close')}
      />

      {/* Bayesian constraints */}
      <SectionHeader title="Bayesian — Massey-Peabody" />
      <Toggle
        label="Same winner across all metrics"
        checked={bayesian.mp_bayesian_all_metrics}
        onChange={() => toggleBayesian('mp_bayesian_all_metrics')}
      />
      <Toggle
        label="Same current and preseason adjusted winner"
        checked={bayesian.mp_bayesian_preseason_and_current}
        onChange={() => toggleBayesian('mp_bayesian_preseason_and_current')}
      />
      <Toggle
        label="Same current and adjusted current winner"
        checked={bayesian.mp_bayesian_current_and_adjusted}
        onChange={() => toggleBayesian('mp_bayesian_current_and_adjusted')}
      />

      <SectionHeader title="Bayesian — Generic Sports Fan" />
      <Toggle
        label="Same adjusted winner across all metrics"
        checked={bayesian.gsf_bayesian_adjusted}
        onChange={() => toggleBayesian('gsf_bayesian_adjusted')}
      />
      <Toggle
        label="Current and preseason adjusted winner"
        checked={bayesian.gsf_bayesian_preseason_and_current}
        onChange={() => toggleBayesian('gsf_bayesian_preseason_and_current')}
      />
      <Toggle
        label="Same current and adjusted current winner"
        checked={bayesian.gsf_bayesian_current_and_adjusted}
        onChange={() => toggleBayesian('gsf_bayesian_current_and_adjusted')}
      />

      <SectionHeader title="Bayesian — Cross Model" />
      <Toggle
        label="Sportsbook: current and preseason adjusted"
        checked={bayesian.sportsbook_bayesian_preseason_and_current}
        onChange={() => toggleBayesian('sportsbook_bayesian_preseason_and_current')}
      />
      <Toggle
        label="Simulation: current and preseason adjusted"
        checked={bayesian.sim_bayesian_preseason_and_current}
        onChange={() => toggleBayesian('sim_bayesian_preseason_and_current')}
      />
      <Toggle
        label="Consensus: current and preseason adjusted"
        checked={bayesian.consensus_bayesian_preseason_and_current}
        onChange={() => toggleBayesian('consensus_bayesian_preseason_and_current')}
      />
      <Toggle
        label="Require ALL Bayesian constraints"
        description="Off = any one must pass. On = all must pass."
        checked={bayesian.bayesian_require_all}
        onChange={() => toggleBayesian('bayesian_require_all')}
      />

      {/* Team constraints */}
      <SectionHeader title="Team Constraints" />
      <div className="flex flex-col gap-1.5">
        <label className="text-xs text-gray-400 font-medium uppercase tracking-wide">
          Prohibited teams (season-long)
        </label>
        <textarea
          value={prohibitedTeams}
          onChange={e => setProhibitedTeams(e.target.value)}
          placeholder="Kansas City Chiefs, Cleveland Browns..."
          rows={3}
          className="bg-gray-900 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-600 focus:border-transparent resize-none placeholder-gray-600"
        />
        <p className="text-xs text-gray-600">Comma-separated team names</p>
      </div>

      <div className="flex flex-col gap-1.5 mt-3">
        <label className="text-xs text-gray-400 font-medium uppercase tracking-wide">
          Required picks
        </label>
        <textarea
          value={requiredPicks}
          onChange={e => setRequiredPicks(e.target.value)}
          placeholder={"Kansas City Chiefs: 3\nBuffalo Bills: 7"}
          rows={3}
          className="bg-gray-900 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-600 focus:border-transparent resize-none placeholder-gray-600"
        />
        <p className="text-xs text-gray-600">One per line: Team Name: week number</p>
      </div>

      <div className="flex flex-col gap-1.5 mt-3">
        <label className="text-xs text-gray-400 font-medium uppercase tracking-wide">
          Avoid team on specific week
        </label>
        <textarea
          value={prohibitedWeeklyPicks}
          onChange={e => setProhibitedWeeklyPicks(e.target.value)}
          placeholder={"Kansas City Chiefs: 3\nBuffalo Bills: 7, 12"}
          rows={3}
          className="bg-gray-900 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-600 focus:border-transparent resize-none placeholder-gray-600"
        />
        <p className="text-xs text-gray-600">One per line: Team Name: week, week...</p>
      </div>

      {/* Run button */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="
          mt-6 w-full py-3 rounded-xl font-semibold text-sm
          bg-green-600 hover:bg-green-500 text-white
          disabled:opacity-50 disabled:cursor-not-allowed
          transition-colors duration-150
          focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2
          focus:ring-offset-gray-950
        "
      >
        {loading ? 'Running...' : 'Run Optimizer'}
      </button>
    </div>
  )
}
