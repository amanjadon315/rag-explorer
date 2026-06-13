import { useState } from 'react'
import Plot from 'react-plotly.js'
import { StageCard, MetricBadge, ScoreBar } from '../components/ui'
import { fullEvaluation } from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'

const DEMO = {
  query: 'How do transformer models use attention?',
  chunks: [
    'Transformer models use self-attention mechanisms to weigh the importance of each token relative to all others in a sequence.',
    'The attention mechanism computes a weighted sum of values, where weights are determined by the compatibility between queries and keys.',
    'Multi-head attention allows the model to attend to different representation subspaces simultaneously.',
  ],
  response: 'Transformer models use self-attention to process all tokens in parallel. Each token attends to every other token via learned query, key, and value matrices. This differs from RNNs which process tokens sequentially. The model invented the internet in 2004.',
  reference: 'Transformers use self-attention mechanisms where each token attends to all others using query, key, and value projections. Multi-head attention runs this process in parallel across multiple subspaces.',
}

export default function EvaluationPanel() {
  const store = usePipelineStore()

  const [query, setQuery] = useState(DEMO.query)
  const [chunksText, setChunksText] = useState(DEMO.chunks.join('\n\n---\n\n'))
  const [response, setResponse] = useState(DEMO.response)
  const [reference, setReference] = useState(DEMO.reference)

  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  // Populate from pipeline store if available
  const populateFromPipeline = () => {
    if (store.query) setQuery(store.query)
    if (store.retrievedChunks.length) setChunksText(store.retrievedChunks.join('\n\n---\n\n'))
    if (store.response) setResponse(store.response)
  }

  const handleEvaluate = async () => {
    setLoading(true)
    try {
      const chunks = chunksText.split(/\n+---\n+/).map(c => c.trim()).filter(Boolean)
      const res = await fullEvaluation(query, chunks, response, reference || undefined)
      setResults(res.data)
    } catch (e: any) {
      alert(e?.response?.data?.detail || e.message)
    } finally { setLoading(false) }
  }

  const verdictColor = (v: string) => v === 'grounded' ? 'green' : v === 'partial' ? 'yellow' : 'red'

  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-lg font-mono text-gray-100">Evaluation Panel</h1>
        <p className="text-xs text-gray-500 mt-1">
          Precision@K · Recall@K · Hallucination detection · ROUGE
        </p>
      </div>

      {/* ── Inputs ──────────────────────────────────────────────────────── */}
      <StageCard title="Evaluation inputs" icon="📋">
        <div className="space-y-3">
          {store.query && (
            <button className="btn-secondary text-xs" onClick={populateFromPipeline}>
              ↑ Pull from pipeline run
            </button>
          )}
          <div>
            <p className="param-label mb-1">Query</p>
            <input className="w-full bg-gray-950 border border-gray-700 rounded px-3 py-1.5 text-xs font-mono text-gray-300"
              value={query} onChange={e => setQuery(e.target.value)} />
          </div>
          <div>
            <p className="param-label mb-1">Retrieved chunks (separate with --- on its own line)</p>
            <textarea className="w-full h-32 bg-gray-950 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-300 resize-none"
              value={chunksText} onChange={e => setChunksText(e.target.value)} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="param-label mb-1">Model response</p>
              <textarea className="w-full h-24 bg-gray-950 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-300 resize-none"
                value={response} onChange={e => setResponse(e.target.value)} />
            </div>
            <div>
              <p className="param-label mb-1">Reference answer (optional — for ROUGE)</p>
              <textarea className="w-full h-24 bg-gray-950 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-400 resize-none"
                value={reference} onChange={e => setReference(e.target.value)} />
            </div>
          </div>
          <button className="btn-primary" onClick={handleEvaluate} disabled={loading}>
            {loading ? 'Evaluating...' : 'Run full evaluation'}
          </button>
        </div>
      </StageCard>

      {results && (
        <>
          {/* ── Retrieval Metrics ─────────────────────────────────────────── */}
          <StageCard title="Retrieval quality" icon="⌖" status="done">
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="param-label mb-1">Precision@K</p>
                  <p className="text-3xl font-mono text-emerald-400">{results.retrieval.precision_at_k}</p>
                  <p className="text-xs text-gray-600 mt-1">relevant / retrieved</p>
                </div>
                <div className="text-center">
                  <p className="param-label mb-1">Recall@K</p>
                  <p className="text-3xl font-mono text-blue-400">{results.retrieval.recall_at_k}</p>
                  <p className="text-xs text-gray-600 mt-1">retrieved / total relevant</p>
                </div>
                <div className="text-center">
                  <p className="param-label mb-1">K</p>
                  <p className="text-3xl font-mono text-gray-300">{results.retrieval.k}</p>
                  <p className="text-xs text-gray-600 mt-1">chunks retrieved</p>
                </div>
              </div>
              <div className="space-y-1">
                <ScoreBar label="Precision@K" score={results.retrieval.precision_at_k} />
                <ScoreBar label="Recall@K" score={results.retrieval.recall_at_k} />
              </div>
              <p className="text-xs text-gray-600 font-mono">
                Precision = fraction of retrieved chunks containing query-relevant content.
                Recall = fraction of all relevant content that was retrieved.
              </p>
            </div>
          </StageCard>

          {/* ── Hallucination Analysis ────────────────────────────────────── */}
          <StageCard title="Hallucination analysis" icon="🔬" status="done">
            <div className="space-y-3">
              <div className="flex gap-3 items-center">
                <MetricBadge
                  label="verdict"
                  value={results.hallucination.verdict}
                  color={verdictColor(results.hallucination.verdict)}
                />
                <MetricBadge
                  label="grounding score"
                  value={results.hallucination.overall_score}
                  color={results.hallucination.overall_score > 0.8 ? 'green' : results.hallucination.overall_score > 0.5 ? 'yellow' : 'red'}
                />
              </div>

              <ScoreBar label="Grounding score" score={results.hallucination.overall_score} />

              <div className="space-y-1">
                <p className="param-label">Sentence-level grounding</p>
                {results.hallucination.sentences.map((s: any, i: number) => (
                  <div key={i} className={`flex gap-3 items-start rounded px-3 py-2 text-xs font-mono
                    ${s.supported ? 'bg-emerald-950/30 border border-emerald-900' : 'bg-red-950/30 border border-red-900'}`}>
                    <span className={s.supported ? 'text-emerald-400' : 'text-red-400'}>{s.supported ? '✓' : '✗'}</span>
                    <span className="text-gray-300 flex-1">{s.sentence}</span>
                    <span className={`shrink-0 ${s.supported ? 'text-emerald-500' : 'text-red-500'}`}>
                      {(s.overlap_score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-600 font-mono">
                Each sentence is scored by token overlap with retrieved context. Low overlap = likely hallucination.
                Production systems use NLI models (e.g. cross-encoder/nli-deberta-v3-base) for higher accuracy.
              </p>
            </div>
          </StageCard>

          {/* ── ROUGE Scores ──────────────────────────────────────────────── */}
          {results.rouge && (
            <StageCard title="ROUGE scores" icon="📐" status="done">
              <div className="space-y-3">
                <div className="space-y-1">
                  <ScoreBar label="ROUGE-1 (unigram overlap)" score={results.rouge.rouge_1_f} />
                  <ScoreBar label="ROUGE-2 (bigram overlap)" score={results.rouge.rouge_2_f} />
                  <ScoreBar label="ROUGE-L (LCS)" score={results.rouge.rouge_l_f} />
                </div>
                <Plot
                  data={[{
                    type: 'bar',
                    x: ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                    y: [results.rouge.rouge_1_f, results.rouge.rouge_2_f, results.rouge.rouge_l_f],
                    marker: {
                      color: ['#34d399', '#3b82f6', '#f59e0b'],
                      opacity: 0.85,
                    },
                    text: [results.rouge.rouge_1_f, results.rouge.rouge_2_f, results.rouge.rouge_l_f].map(v => v.toFixed(3)),
                    textposition: 'outside',
                  }]}
                  layout={{
                    paper_bgcolor: '#030712',
                    plot_bgcolor: '#030712',
                    font: { color: '#9ca3af', family: 'JetBrains Mono', size: 10 },
                    xaxis: { gridcolor: '#1f2937' },
                    yaxis: { gridcolor: '#1f2937', range: [0, 1], title: 'F1 score' },
                    margin: { l: 40, r: 20, t: 20, b: 40 },
                    height: 200,
                    showlegend: false,
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: '100%' }}
                />
                <p className="text-xs text-gray-600 font-mono">
                  ROUGE-L uses the Longest Common Subsequence — best for evaluating fluency and coverage.
                  Requires a reference answer. Higher = closer to reference.
                </p>
              </div>
            </StageCard>
          )}

          {/* ── Summary card ─────────────────────────────────────────────── */}
          <StageCard title="Evaluation summary" icon="◎">
            <div className="grid grid-cols-2 gap-4 text-xs font-mono">
              <div className="space-y-1">
                <p className="text-gray-500 uppercase tracking-widest text-xs">Retrieval</p>
                <p className="text-gray-300">Precision@{results.retrieval.k}: <span className="text-emerald-400">{results.retrieval.precision_at_k}</span></p>
                <p className="text-gray-300">Recall@{results.retrieval.k}: <span className="text-blue-400">{results.retrieval.recall_at_k}</span></p>
              </div>
              <div className="space-y-1">
                <p className="text-gray-500 uppercase tracking-widest text-xs">Grounding</p>
                <p className="text-gray-300">Score: <span className="text-emerald-400">{results.hallucination.overall_score}</span></p>
                <p className="text-gray-300">Verdict: <span className={verdictColor(results.hallucination.verdict) === 'green' ? 'text-emerald-400' : 'text-yellow-400'}>{results.hallucination.verdict}</span></p>
              </div>
              {results.rouge && (
                <div className="space-y-1">
                  <p className="text-gray-500 uppercase tracking-widest text-xs">ROUGE</p>
                  <p className="text-gray-300">ROUGE-L F1: <span className="text-yellow-400">{results.rouge.rouge_l_f}</span></p>
                </div>
              )}
            </div>
          </StageCard>
        </>
      )}
    </div>
  )
}
