import { useState, useRef } from 'react'
import { StageCard, MetricBadge, TokenBar, ParamSlider, StreamingText } from '../components/ui'
import { complete, streamCompletion, ollamaHealth, getModels } from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'

export default function InferenceDashboard() {
  const store = usePipelineStore()

  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [streamMode, setStreamMode] = useState(true)

  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null)

  const [streamText, setStreamText] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const t0Ref = useRef<number>(0)
  const tokenCountRef = useRef<number>(0)

  // Demo messages if pipeline hasn't been run
  const demoMessages = [
    {
      role: 'system',
      content: 'You are a helpful assistant. Answer using only the provided context.',
    },
    {
      role: 'user',
      content: `CONTEXT:\n[Chunk 1]\nTransformer models use self-attention to process tokens in parallel.\n\nQUESTION: How do transformers differ from RNNs?`,
    },
  ]

  const messages = store.promptMessages.length > 0 ? store.promptMessages : demoMessages

  const handleCheckOllama = async () => {
    const res = await ollamaHealth()
    setOllamaOk(res.data.ollama_running)
    if (res.data.ollama_running) {
      const mRes = await getModels()
      setModels(mRes.data.models || [])
    }
  }

  const handleRun = async () => {
    if (streamMode) {
      setStreamText('')
      setIsStreaming(true)
      setMetrics(null)
      t0Ref.current = Date.now()
      tokenCountRef.current = 0

      await streamCompletion(
        messages,
        (token) => {
          setStreamText(t => t + token)
          tokenCountRef.current++
        },
        () => {
          const elapsed = (Date.now() - t0Ref.current) / 1000
          setMetrics({
            latency_ms: Date.now() - t0Ref.current,
            tokens_per_second: (tokenCountRef.current / elapsed).toFixed(1),
            completion_tokens: tokenCountRef.current,
            finish_reason: 'stop',
          })
          setIsStreaming(false)
          store.setResponse(streamText, Date.now() - t0Ref.current, tokenCountRef.current / elapsed)
        }
      )
    } else {
      setLoading(true)
      setMetrics(null)
      try {
        const res = await complete(messages, temperature, maxTokens)
        setStreamText(res.data.response)
        setMetrics(res.data)
        store.setResponse(res.data.response, res.data.latency_ms, res.data.tokens_per_second)
      } catch (e: any) {
        alert(e?.response?.data?.detail || e.message)
      } finally { setLoading(false) }
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-lg font-mono text-gray-100">Inference Dashboard</h1>
        <p className="text-xs text-gray-500 mt-1">
          Token streaming · Latency · Context window · Model internals
        </p>
      </div>

      {/* ── Ollama Status ─────────────────────────────────────────────────── */}
      <StageCard title="Ollama status" icon="🖥">
        <div className="flex items-center gap-3 flex-wrap">
          <button className="btn-secondary" onClick={handleCheckOllama}>Check Ollama</button>
          {ollamaOk === true && <MetricBadge label="ollama" value="running ✓" color="green" />}
          {ollamaOk === false && (
            <div className="text-xs text-red-400 font-mono">
              ✗ Ollama not reachable. Run: <code className="bg-gray-900 px-1 rounded">ollama serve</code>
              &nbsp;and pull a model: <code className="bg-gray-900 px-1 rounded">ollama pull llama3.2</code>
            </div>
          )}
          {models.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="param-label">Model:</span>
              <select
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-300"
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value)}
              >
                <option value="">default</option>
                {models.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          )}
        </div>
      </StageCard>

      {/* ── Sampling Parameters ───────────────────────────────────────────── */}
      <StageCard title="Sampling parameters" icon="⚙">
        <div className="grid grid-cols-3 gap-6">
          <ParamSlider label="Temperature" value={temperature} min={0} max={2} step={0.1}
            onChange={setTemperature} />
          <ParamSlider label="Max tokens" value={maxTokens} min={64} max={2048} step={64}
            onChange={setMaxTokens} unit=" tok" />
          <div className="flex flex-col gap-2">
            <p className="param-label">Output mode</p>
            <label className="flex items-center gap-2 text-xs font-mono text-gray-400 cursor-pointer">
              <input type="checkbox" checked={streamMode} onChange={e => setStreamMode(e.target.checked)}
                className="accent-emerald-400" />
              Token streaming (SSE)
            </label>
          </div>
        </div>
        <div className="mt-3 text-xs text-gray-600 font-mono space-y-0.5">
          <p>temperature=0 → deterministic greedy decoding. temperature&gt;1 → more random (creative/hallucinates more).</p>
          <p>Streaming uses Server-Sent Events: the backend yields each token as it's decoded.</p>
        </div>
      </StageCard>

      {/* ── Context window ────────────────────────────────────────────────── */}
      <StageCard title="Context window utilization" icon="◧">
        <div className="space-y-2">
          <TokenBar used={store.totalTokens || 847} total={8192} />
          <div className="flex gap-2 flex-wrap">
            <MetricBadge label="system" value={`${store.builtPrompt ? '~120' : '~120'} tok`} color="blue" />
            <MetricBadge label="context" value={`${store.totalTokens ? store.totalTokens - 120 - 30 : '~680'} tok`} color="green" />
            <MetricBadge label="query" value="~30 tok" color="yellow" />
            <MetricBadge label="max output" value={`${maxTokens} tok`} />
          </div>
          <p className="text-xs text-gray-600 font-mono">
            Context window = total capacity. Used tokens = prompt. Remaining = available for output.
            RAG works by fitting retrieved chunks into this window.
          </p>
        </div>
      </StageCard>

      {/* ── Messages preview ─────────────────────────────────────────────── */}
      <StageCard title="Messages sent to model" icon="📨">
        <div className="space-y-1">
          {messages.map((m: any, i) => (
            <div key={i} className={`border rounded p-2 text-xs font-mono ${
              m.role === 'system' ? 'border-blue-900 bg-blue-950/20 text-blue-200' :
              m.role === 'assistant' ? 'border-emerald-900 bg-emerald-950/20 text-emerald-200' :
              'border-yellow-900 bg-yellow-950/20 text-yellow-200'
            }`}>
              <span className="opacity-60 uppercase text-xs tracking-widest">{m.role}</span>
              <p className="mt-1 text-gray-300 line-clamp-3">{m.content}</p>
            </div>
          ))}
        </div>
      </StageCard>

      {/* ── Run + Output ──────────────────────────────────────────────────── */}
      <StageCard title="Generated output" icon="▶" status={metrics ? 'done' : isStreaming ? 'running' : 'idle'}>
        <div className="space-y-3">
          <button
            className="btn-primary text-sm"
            onClick={handleRun}
            disabled={isStreaming || loading}
          >
            {isStreaming ? '⬤ Streaming...' : loading ? 'Waiting...' : '▶ Run inference'}
          </button>

          <StreamingText text={streamText} isStreaming={isStreaming} />

          {metrics && (
            <div className="flex gap-2 flex-wrap">
              <MetricBadge label="latency" value={`${metrics.latency_ms}ms`} color="yellow" />
              <MetricBadge label="tok/s" value={metrics.tokens_per_second} color="green" />
              <MetricBadge label="output tokens" value={metrics.completion_tokens || metrics.completion_tokens} />
              <MetricBadge label="finish" value={metrics.finish_reason} color="blue" />
            </div>
          )}

          {metrics && (
            <p className="text-xs text-gray-600 font-mono">
              tokens/sec is the model's inference throughput on your hardware.
              Latency = time-to-first-token + (output_tokens / throughput).
            </p>
          )}
        </div>
      </StageCard>
    </div>
  )
}
