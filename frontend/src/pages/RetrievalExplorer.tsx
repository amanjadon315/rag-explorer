import { useState } from 'react'
import Plot from 'react-plotly.js'
import { StageCard, MetricBadge, ParamSlider, ScoreBar } from '../components/ui'
import { vectorSearch, bm25Search, hybridSearch, indexChunks, chunkText } from '../lib/api'

const DEMO_DOC = `Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised.

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.

Transformer models have revolutionized NLP by introducing attention mechanisms that allow models to weigh the importance of different words when processing text. The BERT model uses bidirectional transformers for pre-training.

Vector databases are specialized database systems designed to store, manage, and query vector embeddings efficiently. They enable semantic search by finding vectors that are geometrically close in high-dimensional space.

Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation. It first retrieves relevant documents from a knowledge base, then uses those documents as context for generating answers.`

export default function RetrievalExplorer() {
  const [query, setQuery] = useState('How do transformer models process text?')
  const [topK, setTopK] = useState(5)
  const [docText, setDocText] = useState(DEMO_DOC)
  const [indexed, setIndexed] = useState(false)

  const [vectorResults, setVectorResults] = useState<any>(null)
  const [bm25Results, setBm25Results] = useState<any>(null)
  const [hybridResults, setHybridResults] = useState<any>(null)

  const [loading, setLoading] = useState<string | null>(null)

  const handleIndex = async () => {
    setLoading('index')
    try {
      const chunkRes = await chunkText(docText, 'recursive', 300, 50)
      const items = chunkRes.data.chunks.map((c: any, i: number) => ({
        chunk_id: `ret_chunk_${i}`,
        text: c.text,
      }))
      await indexChunks(items)
      setIndexed(true)
    } catch (e: any) {
      alert(e?.response?.data?.detail || e.message)
    } finally { setLoading(null) }
  }

  const handleSearch = async () => {
    setLoading('search')
    try {
      const [vRes, bRes, hRes] = await Promise.all([
        vectorSearch(query, topK),
        bm25Search(query, topK),
        hybridSearch(query, topK),
      ])
      setVectorResults(vRes.data)
      setBm25Results(bRes.data)
      setHybridResults(hRes.data)
    } catch (e: any) {
      alert(e?.response?.data?.detail || e.message)
    } finally { setLoading(null) }
  }

  // Build bar chart data for score comparison
  const barData = () => {
    if (!hybridResults) return null
    const labels = hybridResults.results.map((r: any) => `Chunk ${r.chunk_id.split('_').pop()}`)
    return {
      labels,
      rrf: hybridResults.results.map((r: any) => r.rrf_score),
      vec: hybridResults.results.map((r: any) => r.vector_score ?? 0),
      bm25: hybridResults.results.map((r: any) => r.bm25_score ?? 0),
    }
  }
  const bd = barData()

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      <div>
        <h1 className="text-lg font-mono text-gray-100">Retrieval Explorer</h1>
        <p className="text-xs text-gray-500 mt-1">
          Compare vector search · BM25 · hybrid RRF — side by side
        </p>
      </div>

      {/* ── Document + Index ─────────────────────────────────────────────── */}
      <StageCard title="Knowledge base" icon="📚" status={indexed ? 'done' : 'idle'}>
        <div className="space-y-3">
          <textarea
            className="w-full h-32 bg-gray-950 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-300 resize-none"
            value={docText}
            onChange={e => { setDocText(e.target.value); setIndexed(false) }}
          />
          <button className="btn-primary" onClick={handleIndex} disabled={loading === 'index'}>
            {loading === 'index' ? 'Indexing...' : 'Chunk + embed + index'}
          </button>
          {indexed && <MetricBadge label="status" value="indexed ✓" color="green" />}
        </div>
      </StageCard>

      {/* ── Query Controls ────────────────────────────────────────────────── */}
      <StageCard title="Query" icon="🔍">
        <div className="space-y-3">
          <div className="flex gap-3 items-end">
            <input
              className="flex-1 bg-gray-950 border border-gray-700 rounded px-3 py-1.5 text-xs font-mono text-gray-300 placeholder-gray-600"
              placeholder="Enter search query..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
            />
            <div className="w-32">
              <ParamSlider label="top-k" value={topK} min={1} max={10} step={1} onChange={setTopK} />
            </div>
            <button className="btn-primary" onClick={handleSearch} disabled={!indexed || loading === 'search'}>
              {loading === 'search' ? 'Searching...' : 'Compare all 3'}
            </button>
          </div>
          <p className="text-xs text-gray-600 font-mono">
            tip: try keyword queries ("transformer attention") vs semantic ones ("how do models understand context")
          </p>
        </div>
      </StageCard>

      {/* ── Score Distribution Chart ──────────────────────────────────────── */}
      {bd && (
        <StageCard title="Score distribution" icon="📊">
          <Plot
            data={[
              {
                name: 'Vector (cosine)',
                x: bd.labels,
                y: bd.vec,
                type: 'bar',
                marker: { color: '#3b82f6', opacity: 0.8 },
              },
              {
                name: 'BM25',
                x: bd.labels,
                y: bd.bm25.map((v: number) => v / Math.max(...bd.bm25)),  // normalize to 0-1
                type: 'bar',
                marker: { color: '#f59e0b', opacity: 0.8 },
              },
              {
                name: 'RRF (hybrid)',
                x: bd.labels,
                y: bd.rrf.map((v: number) => v * 200),  // scale up RRF for visibility
                type: 'bar',
                marker: { color: '#34d399', opacity: 0.8 },
              },
            ]}
            layout={{
              paper_bgcolor: '#030712',
              plot_bgcolor: '#030712',
              font: { color: '#9ca3af', family: 'JetBrains Mono', size: 10 },
              xaxis: { gridcolor: '#1f2937' },
              yaxis: { gridcolor: '#1f2937', title: 'Score (normalized)' },
              margin: { l: 40, r: 20, t: 20, b: 60 },
              height: 240,
              barmode: 'group',
              legend: { font: { size: 10 }, bgcolor: 'transparent' },
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </StageCard>
      )}

      {/* ── Side-by-side Results ──────────────────────────────────────────── */}
      {vectorResults && bm25Results && hybridResults && (
        <div className="grid grid-cols-3 gap-4">

          {/* Vector */}
          <StageCard title="Vector search" icon="⊕" status="done">
            <div className="space-y-1">
              <div className="flex gap-2 mb-2">
                <MetricBadge label="index" value={vectorResults.index_type} color="blue" />
                <MetricBadge label="latency" value={`${vectorResults.latency_ms}ms`} />
              </div>
              {vectorResults.results.map((r: any) => (
                <div key={r.chunk_id} className="bg-gray-950 border border-gray-800 rounded p-2 space-y-1">
                  <div className="flex justify-between">
                    <span className="text-emerald-400 text-xs font-mono">#{r.rank}</span>
                    <span className="text-blue-400 text-xs font-mono">cos={r.score.toFixed(4)}</span>
                  </div>
                  <ScoreBar label="" score={r.score} maxScore={1} />
                  <p className="text-xs font-mono text-gray-400 line-clamp-3">{r.text}</p>
                </div>
              ))}
            </div>
          </StageCard>

          {/* BM25 */}
          <StageCard title="BM25 search" icon="⌖" status="done">
            <div className="space-y-1">
              <div className="flex gap-2 mb-2">
                <MetricBadge label="corpus" value={bm25Results.corpus_size} />
                <MetricBadge label="latency" value={`${bm25Results.latency_ms}ms`} />
              </div>
              {bm25Results.results.length === 0 && (
                <p className="text-xs text-gray-600 font-mono">No BM25 results — try a keyword query.</p>
              )}
              {bm25Results.results.map((r: any) => (
                <div key={r.chunk_id} className="bg-gray-950 border border-gray-800 rounded p-2 space-y-1">
                  <div className="flex justify-between">
                    <span className="text-emerald-400 text-xs font-mono">#{r.rank}</span>
                    <span className="text-yellow-400 text-xs font-mono">bm25={r.score.toFixed(3)}</span>
                  </div>
                  <ScoreBar label="" score={r.score} maxScore={Math.max(...bm25Results.results.map((x: any) => x.score))} />
                  <p className="text-xs font-mono text-gray-400 line-clamp-3">{r.text}</p>
                </div>
              ))}
            </div>
          </StageCard>

          {/* Hybrid RRF */}
          <StageCard title="Hybrid (RRF)" icon="⬡" status="done">
            <div className="space-y-1">
              <div className="flex gap-2 mb-2">
                <MetricBadge label="RRF k" value={hybridResults.rrf_k} color="green" />
              </div>
              {hybridResults.results.map((r: any) => (
                <div key={r.chunk_id} className="bg-gray-950 border border-emerald-900 rounded p-2 space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-emerald-400 text-xs font-mono">#{r.final_rank}</span>
                    <div className="flex gap-1 text-xs font-mono">
                      {r.vector_rank && <span className="text-blue-400">v:{r.vector_rank}</span>}
                      {r.bm25_rank && <span className="text-yellow-400">b:{r.bm25_rank}</span>}
                    </div>
                  </div>
                  <ScoreBar label="" score={r.rrf_score * 200} maxScore={1} />
                  <p className="text-xs font-mono text-gray-400 line-clamp-3">{r.text}</p>
                </div>
              ))}
            </div>
          </StageCard>
        </div>
      )}
    </div>
  )
}
