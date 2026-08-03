import { useState, useMemo } from 'react'

// ── Content ────────────────────────────────────────────────────────────────
// Grouped FAQ + documentation. Each item: { q, a } where a is JSX-safe string(s).
const SECTIONS = [
  {
    id: 'survivor',
    label: 'Survivor Basics',
    blurb: 'How the Circa Survivor contest works.',
    items: [
      {
        q: 'What is an NFL survivor contest?',
        a: [
          'You pick one team each week that you think will win. If your team wins, you advance. If it loses (or ties), you\'re eliminated.',
          'The catch: you can only use each team once all season. Pick the Chiefs in Week 1 and they\'re gone from your pool for the rest of the year. The last entries standing split the pot.',
        ],
      },
      {
        q: 'What happens if everyone gets eliminated in the same week?',
        a: [
          'Circa\'s rule is that if all remaining entries are eliminated in the same week, the entries with the most total wins split the pot. So surviving the most weeks matters even if nobody runs the table.',
          'This is why the fair-value model prices in correlated eliminations — a week where most of the field dies can be the most valuable week to have survived.',
        ],
      },
      {
        q: 'Why does picking the same popular team as everyone else hurt me?',
        a: [
          'If 40% of the field picks one team and that team loses, 40% of the field is gone at once — and if you weren\'t on it, your share of the pot just grew. Conversely, if you were on it and it won, you\'ve gained nothing on the field.',
          'This is the core tension of survivor: the safest pick by win probability is often the worst pick by expected value, because everyone else is on it too.',
        ],
      },
      {
        q: 'What does "future value" mean when picking?',
        a: [
          'Some teams are strong all season. Using a top team early "burns" it — you can\'t save it for a week when your other options are weak. Sophisticated players hoard strong teams for later weeks, especially short holiday slates.',
          'The tools track this: a team\'s future value is how much you\'d give up by using it now instead of saving it.',
        ],
      },
    ],
  },
  {
    id: 'tools',
    label: 'Using the Tools',
    blurb: 'What each tab does and how to read it.',
    items: [
      {
        q: 'What does the Optimizer do?',
        a: [
          'It searches for the pick path that maximizes your survival odds (or expected value) across the rest of the season, respecting the no-reuse rule. Think of it as planning several weeks ahead instead of one pick at a time.',
        ],
      },
      {
        q: 'What am I looking at in the Analytics tab?',
        a: [
          'Every alive entry in the contest, ranked by fair value. For each entry you see its wins so far, teams remaining, best-path win probability, survival probability, and a fair-value estimate — plus the model\'s prediction of which teams that entry is likely to pick next.',
          'You can scrub through past weeks of any season to watch how rankings shifted after big upsets, and the Final tab shows who won and how.',
        ],
      },
      {
        q: 'What is "fair value" and why would I use it?',
        a: [
          'Fair value is what an entry is worth in dollars, based on its probability of winning a share of the pot. If someone wants to buy or sell an entry, this is a reference price.',
          'It accounts for how many entries remain and how much is in the pot — a contrarian entry with slightly worse odds can be worth more than a chalk entry, because it tends to survive into a smaller field.',
        ],
      },
      {
        q: 'What does the Transactions tab measure?',
        a: [
          'Offseason roster changes, each quantified as net points-per-game of margin a team gained or lost. Inbound value is talent added; outbound is talent lost; net delta is the difference.',
          'It\'s a directional estimate of how much each team improved or declined, useful for adjusting expectations before the season.',
        ],
      },
      {
        q: 'What\'s in the Bets tab?',
        a: [
          'Recommended bets from the models (spreads, moneylines, totals) tiered by edge, a place to track your own bets, and a History & Performance view showing how each bet type and tier has actually done, with both flat-stake and Kelly results.',
        ],
      },
    ],
  },
  {
    id: 'models',
    label: 'How It\'s Calculated',
    blurb: 'The methodology behind the numbers.',
    items: [
      {
        q: 'How are win probabilities calculated?',
        a: [
          'Team power ratings (a consensus of several rating systems) produce a point spread for each game, which converts to a win probability. These feed a Monte Carlo simulation that plays the season thousands of times.',
        ],
      },
      {
        q: 'How does the model predict which teams each entry will pick?',
        a: [
          'Each entry is profiled from its own pick history along a few behavioral axes: does it follow the crowd or go contrarian, favor home teams, chase favorites, or optimize expected value. Those traits become weights in a choice model that outputs a probability for each available team.',
          'The weights themselves were fit on hundreds of thousands of real historical pick decisions, not hand-tuned — so the model reflects how the field has actually behaved.',
        ],
      },
      {
        q: 'How accurate are the pick predictions?',
        a: [
          'On out-of-sample backtests the model correctly identifies an entry\'s exact pick around 45–50% of the time out of roughly a dozen options, and its field-level pick percentages beat the standard top-down approach when blended. Accuracy improves as the season progresses and each entry accumulates history.',
          'Early-season predictions lean more on league-average behavior, since there isn\'t much pick history to profile yet.',
        ],
      },
      {
        q: 'How is survival probability computed?',
        a: [
          'The season is simulated many times. In each simulation, game outcomes are drawn from the win probabilities, and each entry\'s likely pick path is played out. Survival probability is the fraction of simulations in which an entry makes it to the end.',
          'Crucially, all entries share the same simulated game outcomes, so correlated eliminations — chalk entries dying together — are captured rather than assumed independent.',
        ],
      },
      {
        q: 'How is fair value computed?',
        a: [
          'For each simulation, if an entry survives, it splits the pot with however many other entries also survived in that simulation. Fair value averages that payout across all simulations.',
          'Because survivors are counted per-simulation, an entry that tends to survive when few others do gets credited with a larger share — which is why contrarian entries can carry high value.',
        ],
      },
      {
        q: 'How are offseason player values calculated?',
        a: [
          'Skill players (QB, RB, WR, TE) are valued by their prior-season expected points added, expressed per game. Defenders and linemen — where public data can\'t cleanly credit individuals — use position baselines scaled by how many snaps they played.',
          'Only the top 20% of players by playing time get a value; everyone below is treated as depth worth roughly zero, so practice-squad shuffles don\'t distort the totals. Coaching changes and standout players can be hand-adjusted.',
        ],
      },
      {
        q: 'How are the betting recommendations graded?',
        a: [
          'Each model produces a projected spread, total, or win probability, compared against the sportsbook line to find an edge. Bets are tiered by edge size. Performance is tracked against actual closing lines and final scores, with results shown for both flat staking and Kelly staking.',
        ],
      },
      {
        q: 'Where does the data come from?',
        a: [
          'Game results, rosters, and player stats come from public NFL data. Odds come from sportsbook feeds. Contest pick data comes from the Circa survivor pick history. Everything updates through the week as games are played and lines move.',
        ],
      },
    ],
  },
  {
    id: 'caveats',
    label: 'Limits & Caveats',
    blurb: 'What the models can and can\'t do.',
    items: [
      {
        q: 'How much should I trust these numbers?',
        a: [
          'Treat them as informed estimates, not certainties. The win probabilities, pick predictions, and fair values are model outputs built on public data and historical behavior — they capture real signal, but football is high-variance and people are unpredictable.',
          'Use them to inform your own judgment, not to replace it.',
        ],
      },
      {
        q: 'Why do early-season predictions look less confident?',
        a: [
          'The behavioral model needs pick history to tell entries apart. In Weeks 2–4 it leans heavily on league-average behavior, so predictions are fuzzier. By midseason each entry has a clearer profile and accuracy improves.',
        ],
      },
      {
        q: 'Why might a player\'s transaction value look off?',
        a: [
          'Player valuation from public data is genuinely hard, especially for defenders and players in small sample sizes. A breakout player who was a backup last season is valued on last year\'s limited role. Notable players can be hand-corrected in the manual value file.',
        ],
      },
      {
        q: 'Is this affiliated with Circa or the NFL?',
        a: [
          'No. This is an independent analytics tool built for tracking and analyzing the survivor contest. It isn\'t affiliated with, endorsed by, or operated by Circa, the NFL, or any sportsbook.',
        ],
      },
    ],
  },
]

export default function FAQView() {
  const [search, setSearch] = useState('')
  const [openId, setOpenId] = useState(null)   // `${sectionId}-${idx}`

  const q = search.trim().toLowerCase()

  const filteredSections = useMemo(() => {
    if (!q) return SECTIONS
    return SECTIONS
      .map(s => ({
        ...s,
        items: s.items.filter(it =>
          it.q.toLowerCase().includes(q) ||
          it.a.join(' ').toLowerCase().includes(q)),
      }))
      .filter(s => s.items.length > 0)
  }, [q])

  const totalHits = filteredSections.reduce((n, s) => n + s.items.length, 0)

  return (
    <div className="flex flex-col gap-5 max-w-4xl">

      {/* Header */}
      <div>
        <h1 className="text-white font-semibold text-xl">FAQ & Documentation</h1>
        <p className="text-gray-500 text-sm mt-1">
          How the survivor contest works, how to use these tools, and how every
          number is calculated.
        </p>
      </div>

      {/* Search */}
      <div className="relative">
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search questions..."
          className="w-full bg-gray-900 border border-gray-800 text-white text-sm rounded-xl px-4 py-3 focus:outline-none focus:ring-1 focus:ring-green-600 placeholder-gray-600"
        />
        {q && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-gray-600">
            {totalHits} result{totalHits === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {/* Jump nav (hidden while searching) */}
      {!q && (
        <div className="flex gap-2 flex-wrap">
          {SECTIONS.map(s => (
            <a key={s.id} href={`#faq-${s.id}`}
              className="text-xs px-3 py-1.5 rounded-lg border border-gray-800 text-gray-400 hover:text-white hover:border-gray-700 transition-colors">
              {s.label}
            </a>
          ))}
        </div>
      )}

      {/* Sections */}
      {filteredSections.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-2xl flex flex-col items-center justify-center h-40 gap-1">
          <p className="text-gray-400 text-sm">No questions match "{search}"</p>
          <p className="text-gray-600 text-xs">Try a different term, or clear the search</p>
        </div>
      ) : (
        filteredSections.map(section => (
          <section key={section.id} id={`faq-${section.id}`} className="scroll-mt-4">
            <div className="mb-2">
              <h2 className="text-white font-semibold text-sm">{section.label}</h2>
              <p className="text-gray-600 text-xs">{section.blurb}</p>
            </div>

            <div className="flex flex-col gap-2">
              {section.items.map((item, idx) => {
                const id = `${section.id}-${idx}`
                const open = openId === id
                return (
                  <div key={id}
                    className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                    <button
                      onClick={() => setOpenId(open ? null : id)}
                      className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-800/30 transition-colors">
                      <span className={`text-green-500 text-xs transition-transform ${open ? 'rotate-90' : ''}`}>
                        ▶
                      </span>
                      <span className="text-white text-sm font-medium flex-1">{item.q}</span>
                    </button>
                    {open && (
                      <div className="px-4 pb-4 pl-11 flex flex-col gap-2">
                        {item.a.map((para, i) => (
                          <p key={i} className="text-gray-400 text-sm leading-relaxed">{para}</p>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </section>
        ))
      )}

      {/* Footer note */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-xs text-gray-500 mt-2">
        Independent analytics tool, not affiliated with Circa, the NFL, or any
        sportsbook. All projections are estimates for informational purposes.
      </div>
    </div>
  )
}
