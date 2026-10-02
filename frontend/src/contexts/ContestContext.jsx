import { createContext, useContext, useState } from 'react'

// Canonical contest keys (match contest_config / the schedule + optimizer APIs).
export const CONTESTS = [
  { value: 'circa', label: 'Circa' },
  { value: 'big_splash', label: 'Big Splash' },
  { value: 'world_championship', label: 'World Championship' },
]

const ContestContext = createContext(null)

export function ContestProvider({ children }) {
  const [contest, setContest] = useState('circa')
  return (
    <ContestContext.Provider value={{ contest, setContest, contests: CONTESTS }}>
      {children}
    </ContestContext.Provider>
  )
}

export function useContest() {
  const ctx = useContext(ContestContext)
  if (!ctx) throw new Error('useContest must be used within a ContestProvider')
  return ctx
}
