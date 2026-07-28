import { useEffect, useState } from 'react'
import axios from 'axios'
import { LoadingSkeleton, ErrorBox } from './StatusBox'

const API_BASE = 'http://localhost:8000'

export default function RunAuditButton({ onAuditCreated }) {
  const [models, setModels] = useState(null)
  const [selectedModelId, setSelectedModelId] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios
      .get(`${API_BASE}/datasets/models`)
      .then((res) => {
        setModels(res.data)
        if (res.data.length > 0) setSelectedModelId(res.data[0].id)
      })
      .catch(() => setError('Could not load models.'))
  }, [])

  const handleRun = () => {
    if (!selectedModelId) return
    setSubmitting(true)
    setError(null)
    axios
      .post(`${API_BASE}/audit`, { model_id: selectedModelId })
      .then((res) => {
        setSubmitting(false)
        onAuditCreated?.(res.data.id)
      })
      .catch(() => {
        setSubmitting(false)
        setError('Could not start audit run.')
      })
  }

  if (error) return <ErrorBox message={error} />
  if (!models) return <LoadingSkeleton lines={1} />
  if (models.length === 0) {
    return <div className="text-sm text-gray-400">No trained models yet.</div>
  }

  return (
    <div className="flex items-end gap-3">
      <div className="flex-1">
        <label className="mb-1 block text-sm font-medium text-gray-600">Model</label>
        <select
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          className="w-full max-w-xs rounded-md border border-gray-300 bg-white px-3 py-2 text-sm text-gray-700 focus:border-gray-400 focus:outline-none"
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} ({m.algorithm}) — {(m.baseline_accuracy * 100).toFixed(1)}% acc
            </option>
          ))}
        </select>
      </div>
      <button
        onClick={handleRun}
        disabled={submitting}
        className="rounded-md bg-gray-800 px-4 py-2 text-sm font-medium text-white hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting ? 'Starting…' : 'Run New Audit'}
      </button>
    </div>
  )
}