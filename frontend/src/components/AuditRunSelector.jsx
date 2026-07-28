import { useEffect, useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

export default function AuditRunSelector({ selectedId, onSelect }) {
  const [runs, setRuns] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios
      .get(`${API_BASE}/audit`)
      .then((res) => {
        setRuns(res.data)
        // Default to the most recent run if nothing's selected yet
        if (!selectedId && res.data.length > 0) {
          onSelect(res.data[0].id)
        }
      })
      .catch(() => setError('Could not load audit runs.'))
  }, [])

  if (error) return <div className="text-sm text-red-500">{error}</div>
  if (!runs) return <div className="text-sm text-gray-400">Loading audit runs…</div>
  if (runs.length === 0) return <div className="text-sm text-gray-400">No audit runs yet.</div>

  return (
    <div className="mb-4">
      <label className="mb-1 block text-sm font-medium text-gray-600">
        Audit run
      </label>
      <select
        value={selectedId || ''}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full max-w-md rounded-md border border-gray-300 bg-white px-3 py-2 text-sm text-gray-700 focus:border-gray-400 focus:outline-none"
      >
        {runs.map((run) => (
          <option key={run.id} value={run.id}>
            {run.model_name} ({run.algorithm}) — {run.status} — {new Date(run.created_at).toLocaleString()}
          </option>
        ))}
      </select>
    </div>
  )
}