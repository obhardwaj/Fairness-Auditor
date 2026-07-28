import { useEffect, useState } from 'react'
import axios from 'axios'
import ParetoChart from './components/ParetoChart'
import AuditReport from './components/AuditReport'
import AuditRunSelector from './components/AuditRunSelector'
import CalibrationChart from './components/CalibrationChart'

const API_BASE = 'http://localhost:8000'

function App() {
  const [apiStatus, setApiStatus] = useState('checking...')
  const [selectedAuditRunId, setSelectedAuditRunId] = useState(null)

  useEffect(() => {
    axios
      .get(`${API_BASE}/health`)
      .then((res) => setApiStatus(res.data.status))
      .catch(() => setApiStatus('unreachable'))
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-2xl font-semibold text-gray-800 mb-1">
          Bias & Fairness Auditor
        </h1>
        <p className="text-gray-500 mb-2">
          COMPAS recidivism model — mitigation method comparison.
        </p>
        <div className="mb-6 inline-block rounded-lg border border-gray-200 bg-white px-4 py-2 text-sm">
          API status: <span className="font-mono">{apiStatus}</span>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <ParetoChart />
        </div>

        <div className="mt-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <AuditRunSelector selectedId={selectedAuditRunId} onSelect={setSelectedAuditRunId} />
          <AuditReport auditRunId={selectedAuditRunId} />
        </div>

        <div className="mt-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <CalibrationChart auditRunId={selectedAuditRunId} />
        </div>
      </div>
    </div>
  )
}

export default App  