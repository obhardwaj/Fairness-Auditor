import { useEffect, useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

export default function AuditReport({ auditRunId }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!auditRunId) return
    axios
      .get(`${API_BASE}/audit/${auditRunId}/report`)
      .then((res) => setData(res.data))
      .catch(() => setError('Could not load audit report.'))
  }, [auditRunId])

  if (error) return <div className="text-sm text-red-500">{error}</div>
  if (!data) return <div className="text-sm text-gray-400">Loading audit report…</div>
  if (data.status !== 'completed') {
    return <div className="text-sm text-gray-400">Audit status: {data.status}</div>
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