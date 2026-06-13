import { useState } from 'react'
import Plot from 'react-plotly.js'
import { StageCard, MetricBadge, ScoreBar, ParamSlider } from '../components/ui'
import { generateEmbeddings, computeSimilarity, reduceEmbeddings } from '../lib/api'

const DEMO_SENTENCES = [
  "The transformer architecture revolutionized natural language processing.",
  "BERT uses bidirectional attention to understand context.",
  "GPT models generate text using autoregressive decoding.",
  "Vector databases store high-dimensional embeddings for fast retrieval.",
  "The Eiffel Tower is located in Paris, France.",
  "Python is a popular programming language for machine learning.",
  "Neural networks are inspired by the human brain.",
  "Cosine similarity measures the angle between two vectors.",
]

export default function EmbeddingExplorer() {
  const [texts, setTexts] = useState<string[]>(DEMO_SENTENCES)
  const [newText, setNewText] = useState('')
  const [method, setMethod] = useState<'pca' | 'tsne'>('pca')
  const [perplexity, setPerplexity] = useState(5)

  const [scatterData, setScatterData] = useState<any>(null)
  const [simTextA, setSimTextA] = useState(DEMO_SENTENCES[0])
  const [simTextB, setSimTextB] = useState(DEMO_SENTENCES[1])
  const [simResult, setSimResult] = useState<any>(null)
  const [pairMatrix, setPairMatrix] = useState<any>(null)

  const [loading, setLoading] = useState<string | null>(null)

  // ── 2D Scatter ─────────────────────────────────────────────────────────────
  const handleReduce = async () => {
    setLoading('reduce')
    try {
      const labels = texts.map((t, i) => `S${i + 1}: ${t.slice(0, 30)}...`)
      const res = await reduceEmbeddings(texts, labels, method)
      setScatterData(res.data)
    } catch (e: any) {
      alert(e?.response?.data?.detail || e.message)
    } finally { setLoading(null) }
  }

  // ── Cosine Similarity ──────────────────────────────────────────────────────
  const handleSimilarity = async () => {
    setLoading('sim')
    try {
      const res = await computeSimilarity(simTextA, simTextB)
      setSimResult(res.data)
    } catch (e: any) {
      alert(e?.response?.data?.detail || e.message)
    } finally { setLoading(null) }
  }

  // ── Pairwise Matrix ────────────────────────────────────────────────────────
  const handleMatrix = async () => {
    setLoading('matrix')
    try {
      const short = texts.slice(0, 6)
      const embedRes = await generateEmbeddings(short)
      const vecs: number[][] = embedRes.data.vectors

      // Compute cosine similarity for every pair (vectors are normalized, so dot = cosine)
      const n = vecs.length
      const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const dot = vecs[i].reduce((s, v, k) => s + v * vecs[j][k], 0)
          matrix[i][j] = parseFloat(dot.toFixed(3))
        }
      }
      const labels = short.map((t, i) => `S${i + 1}`)
      setPairMatrix({ matrix, labels, texts: short })
    } catch (e: any) {
      alert(e?.response?.data?.detail || e.message)
    } finally { setLoading(null) }
  }

  const plotPoints = scatterData?.points ?? []

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div>
        <h1 className="text-lg font-mono text-gray-100">Embedding Explorer</h1>
        <p className="text-xs text-gray-500 mt-1">
          Visualize semantic space · Compare models · Inspect cosine similarity
        </p>
      </div>

      {/* ── Sentence Manager ──────────────────────────────────────────────── */}
      <StageCard title="Sentence corpus" icon="📝">
        <div className="space-y-2">
          <div className="flex gap-2">
            <input
              className="flex-1 bg-gray-950 border border-gray-700 rounded px-3 py-1.5 text-xs font-mono text-gray-300 placeholder-gray-600"
              placeholder="Add a sentence to visualize..."
              value={newText}
              onChange={e => setNewText(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && newText.trim()) { setTexts(t => [...t, newText.trim()]); setNewText('') } }}
            />
            <button className="btn-primary" onClick={() => { if (newText.trim()) { setTexts(t => [...t, newText.trim()]); setNewText('') } }}>
              Add
            </button>
            <button className="btn-secondary" onClick={() => setTexts(DEMO_SENTENCES)}>Reset</button>
          </div>
          <div className="grid grid-cols-2 gap-1 max-h-40 overflow-y-auto">
            {texts.map((t, i) => (
              <div key={i} className="flex items-center gap-2 bg-gray-950 rounded px-2 py-1 text-xs font-mono">
                <span className="text-emerald-500 shrink-0">S{i + 1}</span>
                <span className="text-gray-400 truncate">{t}</span>
                <button className="ml-auto text-gray-600 hover:text-red-400" onClick={() => setTexts(texts.filter((_, j) => j !== i))}>✕</button>
              </div>
            ))}
          </div>
          <MetricBadge label="sentences" value={texts.length} color="green" />
        </div>
      </StageCard>

      {/* ── 2D Visualization ──────────────────────────────────────────────── */}
      <StageCard title="2D Embedding Space" icon="⊕">
        <div className="space-y-3">
          <div className="flex gap-4 items-end flex-wrap">
            <div>
              <p className="param-label mb-1">Method</p>
              <select
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-300"
                value={method}
                onChange={e => setMethod(e.target.value as 'pca' | 'tsne')}
              >
                <option value="pca">PCA (fast, linear)</option>
                <option value="tsne">t-SNE (cluster-aware)</option>
              </select>
            </div>
            {method === 'tsne' && (
              <div className="w-40">
                <ParamSlider label="Perplexity" value={perplexity} min={2} max={30} step={1}
                  onChange={setPerplexity} />
              </div>
            )}
            <button className="btn-primary" onClick={handleReduce} disabled={loading === 'reduce'}>
              {loading === 'reduce' ? 'Reducing...' : 'Visualize'}
            </button>
          </div>

          {scatterData && (
            <div className="space-y-2">
              <div className="flex gap-2">
                <MetricBadge label="method" value={scatterData.method} color="blue" />
                <MetricBadge label="input dims" value={scatterData.n_input_dimensions} />
                {scatterData.explained_variance_pct && (
                  <MetricBadge label="variance explained" value={`${scatterData.explained_variance_pct}%`} color="green" />
                )}
              </div>
              <Plot
                data={[{
                  x: plotPoints.map((p: any) => p.x),
                  y: plotPoints.map((p: any) => p.y),
                  mode: 'markers+text',
                  type: 'scatter',
                  text: plotPoints.map((p: any) => p.label.slice(0, 20)),
                  textposition: 'top center',
                  textfont: { size: 9, color: '#9ca3af' },
                  marker: { size: 10, color: '#34d399', opacity: 0.85, line: { color: '#065f46', width: 1 } },
                  hovertext: plotPoints.map((p: any) => p.label),
                  hoverinfo: 'text',
                }]}
                layout={{
                  paper_bgcolor: '#030712',
                  plot_bgcolor: '#030712',
                  font: { color: '#9ca3af', family: 'JetBrains Mono', size: 10 },
                  xaxis: { gridcolor: '#1f2937', zerolinecolor: '#374151', title: 'Component 1' },
                  yaxis: { gridcolor: '#1f2937', zerolinecolor: '#374151', title: 'Component 2' },
                  margin: { l: 40, r: 20, t: 20, b: 40 },
                  height: 360,
                  showlegend: false,
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
              <p className="text-xs text-gray-600 font-mono">
                {method === 'pca'
                  ? 'PCA: axes represent directions of maximum variance. Proximity = semantic similarity.'
                  : 't-SNE: preserves local neighborhoods. Clusters = semantically related sentences.'}
              </p>
            </div>
          )}
        </div>
      </StageCard>

      {/* ── Cosine Similarity Inspector ───────────────────────────────────── */}
      <StageCard title="Cosine Similarity Inspector" icon="∠">
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="param-label mb-1">Text A</p>
              <textarea
                className="w-full h-16 bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-300 resize-none"
                value={simTextA}
                onChange={e => setSimTextA(e.target.value)}
              />
            </div>
            <div>
              <p className="param-label mb-1">Text B</p>
              <textarea
                className="w-full h-16 bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-300 resize-none"
                value={simTextB}
                onChange={e => setSimTextB(e.target.value)}
              />
            </div>
          </div>
          <button className="btn-primary" onClick={handleSimilarity} disabled={loading === 'sim'}>
            {loading === 'sim' ? 'Computing...' : 'Compute similarity'}
          </button>
          {simResult && (
            <div className="space-y-2">
              <ScoreBar label="Cosine similarity" score={simResult.cosine_similarity} maxScore={1} />
              <div className="flex gap-2 flex-wrap">
                <MetricBadge label="cosine" value={simResult.cosine_similarity} color={simResult.cosine_similarity > 0.7 ? 'green' : simResult.cosine_similarity > 0.4 ? 'yellow' : 'red'} />
                <MetricBadge label="euclidean dist" value={simResult.euclidean_distance} />
                <MetricBadge label="verdict" value={simResult.interpretation} color="blue" />
              </div>
              <p className="text-xs text-gray-600 font-mono">
                cosine = dot(v_a, v_b) since vectors are L2-normalized.
                Range: –1 (opposite) → 0 (orthogonal) → 1 (identical).
              </p>
            </div>
          )}
        </div>
      </StageCard>

      {/* ── Pairwise Similarity Heatmap ────────────────────────────────────── */}
      <StageCard title="Pairwise Similarity Matrix" icon="▦">
        <div className="space-y-3">
          <p className="text-xs text-gray-500">
            Computes cosine similarity between all pairs in the first 6 sentences.
          </p>
          <button className="btn-primary" onClick={handleMatrix} disabled={loading === 'matrix'}>
            {loading === 'matrix' ? 'Computing...' : 'Build matrix'}
          </button>
          {pairMatrix && (
            <>
              <Plot
                data={[{
                  z: pairMatrix.matrix,
                  x: pairMatrix.labels,
                  y: pairMatrix.labels,
                  type: 'heatmap',
                  colorscale: [
                    [0, '#030712'], [0.3, '#064e3b'], [0.6, '#059669'], [1, '#34d399']
                  ],
                  zmin: 0, zmax: 1,
                  text: pairMatrix.matrix.map((row: number[]) => row.map(v => v.toFixed(2))),
                  texttemplate: '%{text}',
                  textfont: { size: 9 },
                }]}
                layout={{
                  paper_bgcolor: '#030712',
                  plot_bgcolor: '#030712',
                  font: { color: '#9ca3af', family: 'JetBrains Mono', size: 10 },
                  margin: { l: 40, r: 20, t: 20, b: 40 },
                  height: 300,
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
              <div className="space-y-1">
                {pairMatrix.texts.map((t: string, i: number) => (
                  <p key={i} className="text-xs font-mono text-gray-500">
                    <span className="text-emerald-500">S{i + 1}</span>: {t}
                  </p>
                ))}
              </div>
            </>
          )}
        </div>
      </StageCard>
    </div>
  )
}
