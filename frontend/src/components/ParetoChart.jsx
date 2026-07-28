import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts'
import { LoadingSkeleton, ErrorBox } from './StatusBox'

const API_BASE = 'http://localhost:8000'

// One fixed color per method — colors are the primary visual encoding here,
// since method (not algorithm) is the thing we most want the eye to group by.
const METHOD_COLORS = {
  baseline: '#94A3B8',              // slate — the "before" reference point
  reweighing: '#F59E0B',            // amber
  exponentiated_gradient: '#8B5CF6', // violet
  threshold_optimizer: '#10B981',   // emerald — the strongest performer, given its own clear color
}

const METHOD_LABELS = {
    baseline: 'Baseline',
    reweighing: 'Reweighing',
    exponentiated_gradient: 'Exp. Gradient',
    threshold_optimizer: 'Threshold Opt.',
  }

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null
  const d = payload[0].payload
  return (
    <div className="rounded-md border border-gray-200 bg-white px-3 py-2 text-sm shadow-md">
      <div className="font-medium text-gray-800">{METHOD_LABELS[d.method]}</div>
      <div className="text-gray-500 capitalize">{d.algorithm.replace('_', ' ')}</div>
      <div className="mt-1 text-gray-600">
        Accuracy: <span className="font-mono">{(d.accuracy * 100).toFixed(1)}%</span>
      </div>
      <div className="text-gray-600">
        Disparate impact: <span className="font-mono">{d.disparate_impact_ratio.toFixed(3)}</span>
      </div>
    </div>
  )
}

export default function ParetoChart() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios
      .get(`${API_BASE}/audit/mitigation-comparison`)
      .then((res) => setData(res.data))
      .catch(() => setError('Could not load mitigation results.'))
  }, [])

  if (error) return <ErrorBox message={error} />
  if (!data) return <LoadingSkeleton lines={4} />

  // Separate series per method so each renders with a distinct color + legend entry.
  const methods = [...new Set(data.map((d) => d.method))]

  return (
    <div className="w-full">
      <h2 className="mb-1 text-lg font-semibold text-gray-800">
        Accuracy vs. Fairness Tradeoff
      </h2>
      <p className="mb-4 text-sm text-gray-500">
        Disparate impact ratio by mitigation method — the dashed line marks the 0.8 legal threshold.
      </p>
      <ResponsiveContainer width="100%" height={420}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 10 }}>
          <XAxis
            type="number"
            dataKey="accuracy"
            name="Accuracy"
            domain={[0.60, 0.72]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            label={{ value: 'Accuracy', position: 'bottom', fill: '#6B7280' }}
          />
          <YAxis
            type="number"
            dataKey="disparate_impact_ratio"
            name="Disparate Impact Ratio"
            domain={[0, 1]}
            label={{ value: 'Disparate Impact Ratio', angle: -90, position: 'insideLeft', fill: '#6B7280' }}
          />
          <ZAxis range={[120, 120]} />
          <ReferenceLine
            y={0.8}
            stroke="#DC2626"
            strokeDasharray="4 4"
            label={{ value: '80% rule threshold', position: 'insideTopRight', fill: '#DC2626', fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
          <Legend
            formatter={(value) => METHOD_LABELS[value] || value}
            wrapperStyle={{ fontSize: 13, paddingTop: 12}}
          />
          {methods.map((method) => (
            <Scatter
              key={method}
              name={method}
              data={data.filter((d) => d.method === method)}
              fill={METHOD_COLORS[method] || '#6B7280'}
              shape={(props) => {
                // Circle for logistic_regression, diamond for gradient_boosting —
                // so algorithm is encoded by shape, method by color, independently.
                const { cx, cy, fill } = props
                const isGB = props.payload.algorithm === 'gradient_boosting'
                return isGB ? (
                  <rect x={cx - 5} y={cy - 5} width={10} height={10} fill={fill}
                        transform={`rotate(45 ${cx} ${cy})`} />
                ) : (
                  <circle cx={cx} cy={cy} r={6} fill={fill} />
                )
              }}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="mt-2 text-xs text-gray-400">
        ● Logistic Regression &nbsp;&nbsp; ◆ Gradient Boosting
      </div>
    </div>
  )
}