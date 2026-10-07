// Trap-team detection shared by the Schedule and Optimizer views.
//
// A "trap" is a below-average team that is also one of the week's most-picked
// teams — the kind of pick that busts a lot of entries at once. "Below average"
// means a rating below TRAP_RATING_THRESHOLD on Massey-Peabody OR Generic Sports
// Fan (the "...Current Rank" columns, which are point-style ratings centered at
// 0, NOT 1–32 ranks). Severity is driven by how popular the team is: the more
// picked it is, the bigger the trap.

export const TRAP_RATING_THRESHOLD = 3.0

const num = (v) => {
  if (v == null || v === '') return NaN
  const n = Number(v)
  return Number.isNaN(n) ? NaN : n
}

// Weak if EITHER rating is below the threshold (requires at least one real
// rating; a team with no ratings at all is never flagged).
export function isWeakRating(mp, gsf, threshold = TRAP_RATING_THRESHOLD) {
  const m = num(mp)
  const g = num(gsf)
  const hasM = !Number.isNaN(m)
  const hasG = !Number.isNaN(g)
  if (!hasM && !hasG) return false
  return (hasM && m < threshold) || (hasG && g < threshold)
}

// Severity by popularity rank (1 = most-picked). null = not a trap-popular rank.
//   red    = #1 pick       (biggest trap)
//   orange = #2–#3 picks
//   yellow = #4–#5 picks
export function popularitySeverity(rank) {
  const r = num(rank)
  if (Number.isNaN(r) || r < 1) return null
  if (r <= 1) return 'red'
  if (r <= 3) return 'orange'
  if (r <= 5) return 'yellow'
  return null
}

// Combined gate: a team is a trap only when it is BOTH weak AND popular.
// Returns { severity: 'red'|'orange'|'yellow'|null, rank }.
export function trapInfo({ mp, gsf, rank, threshold = TRAP_RATING_THRESHOLD }) {
  const r = num(rank)
  const outRank = Number.isNaN(r) ? null : r
  if (!isWeakRating(mp, gsf, threshold)) return { severity: null, rank: outRank }
  return { severity: popularitySeverity(rank), rank: outRank }
}

// Popularity rank (1 = highest) of `pct` within a list of pick %s, ties sharing
// the better rank. Pass the full week's pick %s.
export function rankWithin(pct, allPcts) {
  const v = num(pct)
  if (Number.isNaN(v)) return null
  let greater = 0
  for (const p of allPcts) {
    const n = num(p)
    if (!Number.isNaN(n) && n > v) greater += 1
  }
  return greater + 1
}

// Tailwind classes per severity. `text` colors a value; `chip` styles a badge;
// `cell` is a subtle background tint for a whole cell; `legend` for the key.
export const TRAP_STYLES = {
  red: {
    text: 'text-red-400',
    chip: 'bg-red-900/60 text-red-200 border border-red-700/60',
    cell: 'bg-red-950/40',
    legend: 'bg-red-900/60',
    label: 'Top pick',
  },
  orange: {
    text: 'text-orange-400',
    chip: 'bg-orange-900/60 text-orange-200 border border-orange-700/60',
    cell: 'bg-orange-950/40',
    legend: 'bg-orange-900/60',
    label: '#2–3 pick',
  },
  yellow: {
    text: 'text-yellow-400',
    chip: 'bg-yellow-900/60 text-yellow-200 border border-yellow-700/60',
    cell: 'bg-yellow-950/40',
    legend: 'bg-yellow-900/60',
    label: '#4–5 pick',
  },
}

// Hover text explaining why a team is flagged.
export function trapTitle(rank, mp, gsf) {
  const parts = []
  const m = num(mp)
  const g = num(gsf)
  if (!Number.isNaN(m)) parts.push(`MP ${m.toFixed(1)}`)
  if (!Number.isNaN(g)) parts.push(`GSF ${g.toFixed(1)}`)
  const ratingStr = parts.length ? ` (${parts.join(', ')})` : ''
  const rankStr = rank ? `the #${rank} most-picked team` : 'a popular pick'
  return `Trap alert: below-average team${ratingStr} that is ${rankStr} this week — high elimination risk if it loses. Consider avoiding.`
}
