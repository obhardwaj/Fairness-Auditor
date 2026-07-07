import { useEffect, useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

function App() {
  const [apiStatus, setApiStatus] = useState('checking...')

  useEffect(() => {
    axios
      .get(`${API_BASE}/health`)
      .then((res) => setApiStatus(res.data.status))
      .catch(() => setApiStatus('unreachable'))
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-8">
      <h1 className="text-2xl font-semibold text-gray-800 mb-2">
        Bias & Fairness Auditor
      </h1>
      <p className="text-gray-500 mb-6">
        Week 1 scaffold — dataset ingestion, baseline models, and audit run wiring.
      </p>
      <div className="rounded-lg border border-gray-200 bg-white px-4 py-2 text-sm">
        API status: <span className="font-mono">{apiStatus}</span>
      </div>
      {/* TODO(week 5): dataset upload, Pareto frontier chart (recharts),
          per-group calibration plots, audit report viewer, agent reasoning trace */}
    </div>
  )
}

export default App
