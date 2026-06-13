import { useState, useEffect } from 'react'
import { StageCard, MetricBadge, TokenBar, JsonViewer } from '../components/ui'
import { buildPrompt, getTemplates } from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'

export default function PromptInspector() {
  const store = usePipelineStore()
  const [templates, setTemplates] = useState<Record<string, string>>({})
  const [selectedTemplate, setSelectedTemplate] = useState('rag_default')
  const [customSystem, setCustomSystem] = useState('')
  const [useCustom, setUseCustom] = useState(false)

  const [builtPrompt, setBuiltPrompt] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  // Demo context chunks if none in store
  const demoChunks = [
    "Transformer models use self-attention mechanisms to process sequences in parallel, unlike RNNs which process tokens sequentially. This parallelism enables much faster training.",
    "BERT (Bidirectional Encoder Representations from Transformers) is pre-trained on masked language modeling and next sentence prediction. It uses bidirectional context.",
    "GPT models are autoregressive: they predict the next token given all previous tokens. They use a causal (left-to-right) attention mask during training.",
  ]

  const contextChunks = store.retrievedChunks.length > 0 ? store.retrievedChunks : demoChunks
  const query = store.query || 'How do transformer models differ from traditional RNNs?'

  useEffect(() => {
    getTemplates().then(res => setTemplates(res.data.templates)).catch(() => {})
  }, [])

  const handleBuild = async () => {
    setLoading(true)
    try {
      const res = await buildPrompt(
        query,
        contextChunks,
        selectedTemplate,
        useCustom ? customSystem : undefined,
      )
      setBuiltPrompt(res.data)
      store.setBuiltPrompt(res.data.final_prompt, res.data.messages, res.data.total_tokens, res.data.utilization_pct)
    } catch (e: any) {
      alert(e?.response?.data?.detail || e.message)
    } finally { setLoading(false) }
  }

  const sectionColor: Record<string, string> = {
    system: 'border-l-blue-500',
    context: 'border-l-emerald-500',
    query: 'border-l-yellow-500',
  }

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div>
        <h1 className="text-lg font-mono text-gray-100">Prompt Inspector</h1>
        <p className="text-xs text-gray-500 mt-1">
          See exactly what gets sent to the LLM · Edit sections · Track token budget
        </p>
      </div>

      {/* ── Controls ────────────────────────────────────────────────────── */}
      <StageCard title="Configuration" icon="⚙">
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="param-label mb-1">System template</p>
              <select
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-300"
                value={selectedTemplate}
                onChange={e => setSelectedTemplate(e.target.value)}
                disabled={useCustom}
              >
                {Object.keys(templates).map(k => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 text-xs font-mono text-gray-400 cursor-pointer">
                <input type="checkbox" checked={useCustom} onChange={e => setUseCustom(e.target.checked)}
                  className="accent-emerald-400" />
                Use custom system prompt
              </label>
            </div>
          </div>

          {useCustom && (
            <textarea
              className="w-full h-20 bg-gray-950 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-300 resize-none"
              placeholder="Enter custom system prompt..."
              value={customSystem}
              onChange={e => setCustomSystem(e.target.value)}
            />
          )}

          <button className="btn-primary" onClick={handleBuild} disabled={loading}>
            {loading ? 'Building...' : 'Build prompt'}
          </button>
        </div>
      </StageCard>

      {/* ── Three-Panel View ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-4">

        {/* System prompt */}
        <StageCard title="System prompt" icon="🔵">
          <div className="space-y-2">
            <p className="text-xs text-gray-500">Sets the model's behavior and constraints.</p>
            <div className="bg-gray-950 border-l-2 border-l-blue-500 rounded-r p-3 text-xs font-mono text-gray-300 min-h-24">
              {useCustom
                ? (customSystem || <span className="text-gray-600">empty</span>)
                : (templates[selectedTemplate] || <span className="text-gray-600">loading...</span>)
              }
            </div>
            {builtPrompt && (
              <MetricBadge
                label="tokens"
                value={builtPrompt.sections.find((s: any) => s.name === 'system')?.token_count ?? 0}
                color="blue"
              />
            )}
          </div>
        </StageCard>

        {/* Context */}
        <StageCard title="Retrieved context" icon="🟢">
          <div className="space-y-2">
            <p className="text-xs text-gray-500">Chunks injected as grounding evidence.</p>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {contextChunks.map((chunk, i) => (
                <div key={i} className="bg-gray-950 border-l-2 border-l-emerald-500 rounded-r p-2 text-xs font-mono text-gray-400">
                  <span className="text-emerald-600">[Chunk {i + 1}]</span>{' '}
                  {chunk.slice(0, 120)}...
                </div>
              ))}
            </div>
            {builtPrompt && (
              <MetricBadge
                label="tokens"
                value={builtPrompt.sections.find((s: any) => s.name === 'context')?.token_count ?? 0}
                color="green"
              />
            )}
          </div>
        </StageCard>

        {/* Query */}
        <StageCard title="User query" icon="🟡">
          <div className="space-y-2">
            <p className="text-xs text-gray-500">The user's question.</p>
            <div className="bg-gray-950 border-l-2 border-l-yellow-500 rounded-r p-3 text-xs font-mono text-gray-300 min-h-16">
              {query}
            </div>
            {builtPrompt && (
              <MetricBadge
                label="tokens"
                value={builtPrompt.sections.find((s: any) => s.name === 'query')?.token_count ?? 0}
                color="yellow"
              />
            )}
          </div>
        </StageCard>
      </div>

      {/* ── Assembled Prompt ─────────────────────────────────────────────── */}
      {builtPrompt && (
        <StageCard title="Assembled prompt" icon="✎" status="done">
          <div className="space-y-3">
            <TokenBar used={builtPrompt.total_tokens} total={builtPrompt.context_window} />

            <div className="flex gap-2 flex-wrap">
              <MetricBadge label="total tokens" value={builtPrompt.total_tokens} color="yellow" />
              <MetricBadge label="context window" value={builtPrompt.context_window.toLocaleString()} />
              <MetricBadge
                label="utilization"
                value={`${builtPrompt.utilization_pct}%`}
                color={builtPrompt.utilization_pct > 80 ? 'red' : builtPrompt.utilization_pct > 60 ? 'yellow' : 'green'}
              />
              <MetricBadge label="chunks" value={builtPrompt.n_context_chunks} color="blue" />
            </div>

            {/* Section breakdown */}
            <div className="space-y-1">
              {builtPrompt.sections.map((s: any) => (
                <div key={s.name} className={`border-l-2 pl-3 py-1 ${sectionColor[s.name] || 'border-l-gray-600'}`}>
                  <div className="flex justify-between text-xs font-mono text-gray-500">
                    <span className="uppercase tracking-widest">{s.name}</span>
                    <span>{s.token_count} tokens · {s.char_count} chars</span>
                  </div>
                </div>
              ))}
            </div>

            <JsonViewer data={builtPrompt.messages} title="Raw messages array (sent to LLM API)" collapsed />

            <div>
              <p className="param-label mb-1">Final prompt string</p>
              <div className="code-block max-h-64 overflow-y-auto text-xs whitespace-pre-wrap">
                {builtPrompt.final_prompt}
              </div>
            </div>

            <p className="text-xs text-gray-600 font-mono">
              Context ordering: chunks are placed before the query so the model reads evidence before the question.
              This improves answer grounding compared to query-first ordering.
            </p>
          </div>
        </StageCard>
      )}
    </div>
  )
}
