import { useState } from 'react'

export default function App() {
  const [activeTab, setActiveTab] = useState('optimizer')

  return (
    <div className="min-h-screen bg-brand-dark text-white">

      {/* Top navigation bar */}
      <nav className="border-b border-brand-border bg-brand-card">
        <div className="max-w-screen-2xl mx-auto px-6 flex items-center gap-8 h-14">

          {/* Logo */}
          <div className="flex items-center gap-2 mr-4">
            <span className="text-brand-green font-bold text-lg">🏈</span>
            <span className="font-semibold text-white tracking-tight">
              Circa Survivor
            </span>
          </div>

          {/* Tabs */}
          {['optimizer', 'schedule', 'analytics'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`
                text-sm font-medium capitalize h-14 border-b-2 transition-colors px-1
                ${activeTab === tab
                  ? 'border-brand-green text-white'
                  : 'border-transparent text-brand-muted hover:text-white'}
              `}
            >
              {tab}
            </button>
          ))}
        </div>
      </nav>

      {/* Page content */}
      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        {activeTab === 'optimizer' && (
          <div className="text-brand-muted text-sm">
            Optimizer view coming soon...
          </div>
        )}
        {activeTab === 'schedule' && (
          <div className="text-brand-muted text-sm">
            Schedule view coming soon...
          </div>
        )}
        {activeTab === 'analytics' && (
          <div className="text-brand-muted text-sm">
            Analytics view coming soon...
          </div>
        )}
      </main>
    </div>
  )
}
