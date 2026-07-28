import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { LoadingSkeleton, ErrorBox } from './StatusBox'

const API_BASE = 'http://localhost:8000'

const GROUP_COLORS = {
  'African-American': '#8B5CF6',
  'Caucasian': '#F59E0B',
}

export default function CalibrationChart({ auditRunId }) {
  const [curves, setCurves] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!auditRunId) return
    setCurves(null)
    axios
      .get(`${API_BASE}/audit/${auditRunId}/calibration`)
      .then((res) => setCurves(res.data))
      .catch(() => setError('Could not load calibration data.'))
  }, [auditRunId])

  if (error) return <ErrorBox message={error} />
  if (!curves) return <LoadingSkeleton lines={4} />

  const groups = Object.keys(curves)
  if (groups.length === 0) {
    return <div className="text-sm text-gray-400">Not enough samples per group to compute calibration.</div>
  }

  // Recharts wants one array of points per line; each group's curve is
  // plotted as its own <Line>, keyed by its own x (prob_pred) values.
  const seriesByGroup = groups.map((group) => ({
    group,
    points: curves[group].prob_pred.map((pp, i) => ({
      prob_pred: pp,
      prob_true: curves[group].prob_true[i],
    })),
  }))

  // A single combined dataset for the perfect-calibration reference line
  const diagonal = [{ prob_pred: 0, prob_true: 0 }, { prob_pred: 1, prob_true: 1 }]

  return (
    <div className="w-full">
      <h2 className="mb-1 text-lg font-semibold text-gray-800">Calibration by Group</h2>
      <p className="mb-4 text-sm text-gray-500">
        Predicted vs. actual reoffense rate, binned by decile. The dashed diagonal is perfect calibration.
      </p>
      <ResponsiveContainer width="100%" height={360}>
        <LineChart margin={{ top: 10, right: 30, bottom: 50, left: 10 }}>
        <Legend wrapperStyle={{ paddingTop: 16 }} />
          <XAxis
            type="number"
            dataKey="prob_pred"
            domain={[0, 1]}
            label={{ value: 'Predicted probability', position: 'bottom', fill: '#6B7280' }}
            allowDuplicatedCategory={false}
          />
          <YAxis
            type="number"
            dataKey="prob_true"
            domain={[0, 1]}
            label={{ value: 'Observed frequency', angle: -90, position: 'insideLeft', fill: '#6B7280' }}
          />
          <Tooltip formatter={(value) => value.toFixed(3)} />
          <Legend />
          <Line
            data={diagonal}
            dataKey="prob_true"
            stroke="#D1D5DB"
            strokeDasharray="4 4"
            dot={false}
            name="Perfect calibration"
            legendType="none"
          />
          {seriesByGroup.map(({ group, points }) => (
            <Line
              key={group}
              data={points}
              dataKey="prob_true"
              name={group}
              stroke={GROUP_COLORS[group] || '#6B7280'}
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}