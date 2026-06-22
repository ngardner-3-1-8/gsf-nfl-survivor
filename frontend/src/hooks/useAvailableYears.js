import { useState, useEffect } from 'react'
import { fetchAvailableYears } from '../api/client'

export function useAvailableYears() {
  const [years, setYears] = useState([])
  const [selectedYear, setSelectedYear] = useState(null)
  const [currentYear, setCurrentYear] = useState(null)

  useEffect(() => {
    fetchAvailableYears()
      .then(data => {
        setYears(data.years || [])
        setCurrentYear(data.current_year)
        setSelectedYear(data.current_year) // default to current year
      })
      .catch(() => {})
  }, [])

  const isHistorical = selectedYear && selectedYear !== currentYear

  return { years, selectedYear, setSelectedYear, currentYear, isHistorical }
}
