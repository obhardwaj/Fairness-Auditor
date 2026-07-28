import { useEffect, useState, useRef } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'
const POLL_INTERVAL_MS = 3000

export default function AuditReport({ auditRunId }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (!auditRunId) return

    setData(null)
    setError(null)

    const fetchReport = () => {
      axios
        .get(`${API_BASE}/audit/${auditRunId}/report`)
        .then((res) => {
          setData(res.data)
          // Stop polling once the run reaches a terminal state.
          if (res.data.status === 'completed' || res.data.status === 'failed') {
            clearInterval(intervalRef.current)
          }
        })
        .catch(() => {
          setError('Could not load audit report.')
          clearInterval(intervalRef.current)
        })
    }

    fetchReport() // immediate first fetch, don't wait for the first interval tick
    intervalRef.current = setInterval(fetchReport, POLL_INTERVAL_MS)

    return () => clearInterval(intervalRef.current) // cleanup on unmount or auditRunId change
  }, [auditRunId])

  if (error) return <div className="text-sm text-red-500">{error}</div>
  if (!data) return <div className="text-sm text-gray-400">Loading audit report…</div>

  if (data.status !== 'completed') {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-gray-300 border-t-gray-500" />
        Audit in progress — status: <span className="font-mono">{data.status}</span>
      </div>
    )
  }

  const trace = data.agent_trace || {}

  return (
    <div className="space-y-6">
      <div>
        <h2 className="mb-2 text-lg font-semibold text-gray-800">Audit Report</h2>
        <div className="whitespace-pre-line rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm leading-relaxed text-gray-700">
          {data.report}
        </div>
      </div>

      {trace.profiler_notes && (
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
            Profiler Reasoning
          </h3>
          <p className="text-sm text-gray-600">{trace.profiler_notes}</p>
        </div>
      )}

      {trace.contradictions && trace.contradictions.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
            Critic — Flagged Contradictions
          </h3>
          <ul className="space-y-2">
            {trace.contradictions.map((c, i) => (
              <li key={i} className="rounded-md border-l-4 border-amber-400 bg-amber-50 px-3 py-2 text-sm text-gray-700">
                {c}
              </li>
            ))}
          </ul>
        </div>
      )}

      {trace.chosen_mitigation && (
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
            Mitigation Chosen
          </h3>
          <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-gray-700">
            <span className="font-medium capitalize">{trace.chosen_mitigation.replace('_', ' ')}</span>
            {trace.mitigation_reasoning && <> — {trace.mitigation_reasoning}</>}
          </div>
        </div>
      )}
    </div>
  )
}