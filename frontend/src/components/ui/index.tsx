import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import clsx from 'clsx'

// ── MetricBadge ──────────────────────────────────────────────────────────────
interface MetricBadgeProps {
  label: string
  value: string | number
  color?: 'green' | 'yellow' | 'blue' | 'red' | 'gray'
}
export function MetricBadge({ label, value, color = 'gray' }: MetricBadgeProps) {
  const colors = {
    green:  'border-emerald-800 text-emerald-300 bg-emerald-950',
    yellow: 'border-yellow-800 text-yellow-300 bg-yellow-950',
    blue:   'border-blue-800 text-blue-300 bg-blue-950',
    red:    'border-red-800 text-red-300 bg-red-950',
    gray:   'border-gray-700 text-gray-300 bg-gray-900',
  }
  return (
    <span className={clsx('metric-chip border', colors[color])}>
      <span className="text-gray-500">{label}</span>
      <span className="font-semibold">{value}</span>
    </span>
  )
}

// ── JsonViewer ───────────────────────────────────────────────────────────────
interface JsonViewerProps {
  data: unknown
  title?: string
  collapsed?: boolean
}
export function JsonViewer({ data, title, collapsed = false }: JsonViewerProps) {
  const [open, setOpen] = useState(!collapsed)
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden text-xs font-mono">
      {title && (
        <button
          onClick={() => setOpen(v => !v)}
          className="w-full flex items-center gap-2 px-3 py-2 bg-gray-900 text-gray-400
                     hover:bg-gray-800 transition-colors text-left"
        >
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          {title}
        </button>
      )}
      {open && (
        <pre className="p-3 bg-gray-950 text-gray-300 overflow-x-auto max-h-64 text-xs leading-relaxed">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}

// ── StageCard ────────────────────────────────────────────────────────────────
interface StageCardProps {
  title: string
  icon?: string
  status?: 'idle' | 'running' | 'done' | 'error'
  color?: string
  children: React.ReactNode
  className?: string
}
export function StageCard({ title, icon, status = 'idle', color = 'emerald', children, className }: StageCardProps) {
  const statusDot = {
    idle:    'bg-gray-600',
    running: 'bg-yellow-400 animate-pulse',
    done:    'bg-emerald-400',
    error:   'bg-red-400',
  }[status]

  return (
    <div className={clsx('card', className)}>
      <div className="card-header">
        <span className={clsx('stage-dot', statusDot)} />
        {icon && <span>{icon}</span>}
        <span>{title}</span>
        <span className={clsx('ml-auto text-xs', `status-${status}`)}>
          {status !== 'idle' && status}
        </span>
      </div>
      {children}
    </div>
  )
}

// ── ParamSlider ───────────────────────────────────────────────────────────────
interface ParamSliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  unit?: string
}
export function ParamSlider({ label, value, min, max, step = 1, onChange, unit }: ParamSliderProps) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="param-label">{label}</span>
        <span className="text-xs font-mono text-gray-300">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                   [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-emerald-400"
      />
    </div>
  )
}

// ── ScoreBar ─────────────────────────────────────────────────────────────────
interface ScoreBarProps {
  label: string
  score: number   // 0–1
  maxScore?: number
}
export function ScoreBar({ label, score, maxScore = 1 }: ScoreBarProps) {
  const pct = Math.min((score / maxScore) * 100, 100)
  const color = pct > 66 ? 'bg-emerald-500' : pct > 33 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className="w-40 truncate text-gray-400">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={clsx('h-full rounded-full transition-all', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-12 text-right text-gray-300">{score.toFixed(3)}</span>
    </div>
  )
}

// ── StreamingText ────────────────────────────────────────────────────────────
interface StreamingTextProps {
  text: string
  isStreaming: boolean
}
export function StreamingText({ text, isStreaming }: StreamingTextProps) {
  return (
    <div className="code-block min-h-16 relative">
      {text || <span className="text-gray-600">Response will appear here...</span>}
      {isStreaming && (
        <span className="inline-block w-2 h-4 bg-emerald-400 animate-pulse ml-0.5 align-middle" />
      )}
    </div>
  )
}

// ── TokenBar ─────────────────────────────────────────────────────────────────
interface TokenBarProps {
  used: number
  total: number
}
export function TokenBar({ used, total }: TokenBarProps) {
  const pct = Math.min((used / total) * 100, 100)
  const color = pct > 90 ? 'bg-red-500' : pct > 70 ? 'bg-yellow-500' : 'bg-emerald-500'
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-mono text-gray-400">
        <span>Context window</span>
        <span>{used.toLocaleString()} / {total.toLocaleString()} tokens ({pct.toFixed(1)}%)</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className={clsx('h-full rounded-full transition-all', color)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}
