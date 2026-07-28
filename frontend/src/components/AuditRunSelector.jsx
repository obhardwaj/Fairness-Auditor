import { useEffect, useState, useRef } from 'react'
import axios from 'axios'
import { LoadingSkeleton, ErrorBox } from './StatusBox'

const API_BASE = 'http://localhost:8000'
const POLL_INTERVAL_MS = 5000

export default function AuditRunSelector({ selectedId, onSelect }) {
  const [runs, setRuns] = useState(null)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)
  const hasSetDefaultRef = useRef(false)

  useEffect(() => {
    const fetchRuns = () => {
      axios
        .get(`${API_BASE}/audit`)
        .then((res) => {
          setRuns(res.data)

          // Only auto-select a default once, on first successful load —
          // don't override the user's manual selection on later poll ticks.
          if (!hasSetDefaultRef.current && !selectedId && res.data.length > 0) {
            onSelect(res.data[0].id)
            hasSetDefaultRef.current = true
          }

          // Stop polling once every run has reached a terminal state —
          // nothing left that could still change.
          const anyInProgress = res.data.some(
            (r) => r.status !== 'completed' && r.status !== 'failed'
          )
          if (!anyInProgress) {
            clearInterval(intervalRef.current)
          }
        })
        .catch(() => {
          setError('Could not load audit runs.')
          clearInterval(intervalRef.current)
        })
    }

    fetchRuns()
    intervalRef.current = setInterval(fetchRuns, POLL_INTERVAL_MS)

    return () => clearInterval(intervalRef.current)
  }, [])

  if (error) return <ErrorBox message={error} />
  if (!runs) return <LoadingSkeleton lines={1} />
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