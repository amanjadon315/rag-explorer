import { useState } from 'react'
import { StageCard, MetricBadge, ParamSlider, JsonViewer, StreamingText, TokenBar } from '../components/ui'
import {
  uploadFile, pasteText, cleanText, chunkText, indexChunks,
  hybridSearch, buildPrompt, streamCompletion, fullEvaluation
} from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'

type Stage = 'upload' | 'clean' | 'chunk' | 'index' | 'retrieve' | 'prompt' | 'infer' | 'eval'

export default function PipelineView() {
  const store = usePipelineStore()
  const [loading, setLoading] = useState<Stage | null>(null)
  const [pasteMode, setPasteMode] = useState(false)
  const [pastedText, setPastedText] = useState('')
  const [error, setError] = useState<string | null>(null)

  // ── Stage outputs (local — supplements the store) ──────────────────────────
  const [cleanOutput, setCleanOutput] = useState<any>(null)
  const [chunkOutput, setChunkOutput] = useState<any>(null)
  const [indexOutput, setIndexOutput] = useState<any>(null)
  const [retrievalOutput, setRetrievalOutput] = useState<any>(null)
  const [promptOutput, setPromptOutput] = useState<any>(null)
  const [evalOutput, setEvalOutput] = useState<any>(null)

  const run = async (stage: Stage, fn: () => Promise<void>) => {
    setLoading(stage)
    setError(null)
    try { await fn() }
    catch (e: any) { setError(`${stage}: ${e?.response?.data?.detail || e.message}`) }
    finally { setLoading(null) }
  }

  // ── 1. Upload / Paste ────────────────────────────────────────────────────
  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    run('upload', async () => {
      const res = await uploadFile(file)
      store.setRawText(res.data.text, res.data.filename)
    })
  }

  const handlePaste = () => run('upload', async () => {
    const res = await pasteText(pastedText)
    store.setRawText(res.data.text, res.data.filename)
    setPasteMode(false)
  })

  // ── 2. Clean ─────────────────────────────────────────────────────────────
  const handleClean = () => run('clean', async () => {
    const res = await cleanText(store.rawText)
    store.setCleanedText(res.data.cleaned)
    setCleanOutput(res.data)
  })

  // ── 3. Chunk ─────────────────────────────────────────────────────────────
  const handleChunk = () => run('chunk', async () => {
    const res = await chunkText(store.cleanedText || store.rawText, store.chunkStrategy, store.chunkSize, store.chunkOverlap)
    store.setChunks(res.data.chunks, res.data.strategy, res.data.chunk_size, res.data.overlap)
    setChunkOutput(res.data)
  })

  // ── 4. Index (embed + store) ──────────────────────────────────────────────
  const handleIndex = () => run('index', async () => {
    const items = store.chunks.map((c, i) => ({ chunk_id: `chunk_${i}`, text: c.text }))
    const res = await indexChunks(items)
    setIndexOutput(res.data)
  })

  // ── 5. Retrieve ───────────────────────────────────────────────────────────
  const handleRetrieve = () => run('retrieve', async () => {
    const res = await hybridSearch(store.query, store.topK)
    const chunks = res.data.results.map((r: any) => r.text)
    const scores = res.data.results.map((r: any) => r.rrf_score)
    store.setRetrievedChunks(chunks, scores)
    setRetrievalOutput(res.data)
  })

  // ── 6. Build Prompt ───────────────────────────────────────────────────────
  const handleBuildPrompt = () => run('prompt', async () => {
    const res = await buildPrompt(store.query, store.retrievedChunks, store.systemTemplate)
    store.setBuiltPrompt(res.data.final_prompt, res.data.messages, res.data.total_tokens, res.data.utilization_pct)
    setPromptOutput(res.data)
  })

  // ── 7. Infer ──────────────────────────────────────────────────────────────
  const handleInfer = () => run('infer', async () => {
  store.setIsStreaming(true)
  store.setResponse('', 0, 0)
  const t0 = Date.now()
  let tokenCount = 0
  let fullText = ''

  await streamCompletion(
    store.promptMessages as any,
    (token) => {
      fullText += token
      store.appendStreamToken(token)
      tokenCount++
    },
    () => {
      const elapsed = (Date.now() - t0) / 1000
      store.setResponse(fullText, Date.now() - t0, tokenCount / elapsed)
      store.setIsStreaming(false)
    }
  )
})

  // ── 8. Evaluate ───────────────────────────────────────────────────────────
  const handleEval = () => run('eval', async () => {
    const res = await fullEvaluation(store.query, store.retrievedChunks, store.response || store.streamingResponse)
    store.setEvaluation(res.data)
    setEvalOutput(res.data)
  })

  const stageStatus = (stage: Stage) => loading === stage ? 'running' : 'idle'

  return (
    <div className="p-6 space-y-4 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-lg font-mono text-gray-100">Pipeline View</h1>
        <p className="text-xs text-gray-500 mt-1">
          Walk through all 11 stages. Each stage shows its input, output, and key metrics.
        </p>
      </div>

      {error && (
        <div className="bg-red-950 border border-red-800 rounded-lg px-4 py-2 text-xs text-red-300 font-mono">
          ⚠ {error}
        </div>
      )}

      {/* ── Stage 1: Document Input ──────────────────────────────────────── */}
      <StageCard title="1 · Document Input" icon="📄" status={store.rawText ? 'done' : stageStatus('upload')}>
        <div className="space-y-3">
          <div className="flex gap-2">
            <label className="btn-primary cursor-pointer">
              Upload file
              <input type="file" className="hidden" accept=".txt,.md,.pdf,.docx,.html" onChange={handleFile} />
            </label>
            <button className="btn-secondary" onClick={() => setPasteMode(v => !v)}>Paste text</button>
          </div>
          {pasteMode && (
            <div className="space-y-2">
              <textarea
                className="w-full h-32 bg-gray-950 border border-gray-700 rounded p-2 text-xs font-mono text-gray-300 resize-none"
                placeholder="Paste any text here..."
                value={pastedText}
                onChange={e => setPastedText(e.target.value)}
              />
              <button className="btn-primary" onClick={handlePaste}>Use this text</button>
            </div>
          )}
          {store.rawText && (
            <div className="space-y-2">
              <div className="flex gap-2 flex-wrap">
                <MetricBadge label="file" value={store.filename || 'pasted'} />
                <MetricBadge label="chars" value={store.rawText.length.toLocaleString()} color="green" />
                <MetricBadge label="words" value={store.rawText.split(/\s+/).length.toLocaleString()} color="blue" />
              </div>
              <div className="code-block max-h-24 overflow-y-auto">{store.rawText.slice(0, 400)}...</div>
            </div>
          )}
        </div>
      </StageCard>

      {/* connector */}
      <div className="flex justify-center"><div className="w-px h-6 bg-gray-800" /></div>

      {/* ── Stage 2: Text Cleaning ───────────────────────────────────────── */}
      <StageCard title="2 · Text Cleaning" icon="🧹" status={cleanOutput ? 'done' : stageStatus('clean')}>
        <div className="space-y-3">
          <button className="btn-primary" onClick={handleClean} disabled={!store.rawText || loading === 'clean'}>
            {loading === 'clean' ? 'Cleaning...' : 'Run cleaner'}
          </button>
          {cleanOutput && (
            <div className="space-y-2">
              <div className="flex gap-2 flex-wrap">
                <MetricBadge label="reduction" value={`${cleanOutput.reduction_pct}%`} color="yellow" />
                <MetricBadge label="before" value={cleanOutput.original_length.toLocaleString()} />
                <MetricBadge label="after" value={cleanOutput.cleaned_length.toLocaleString()} color="green" />
              </div>
              <div className="text-xs text-gray-500 font-mono">
                Changes: {cleanOutput.changes.join(' · ')}
              </div>
            </div>
          )}
        </div>
      </StageCard>

      <div className="flex justify-center"><div className="w-px h-6 bg-gray-800" /></div>

      {/* ── Stage 3: Chunking ────────────────────────────────────────────── */}
      <StageCard title="3 · Chunking" icon="✂" status={store.chunks.length ? 'done' : stageStatus('chunk')}>
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="param-label mb-1">Strategy</p>
              <select
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-300"
                value={store.chunkStrategy}
                onChange={e => usePipelineStore.setState({ chunkStrategy: e.target.value })}
              >
                <option value="recursive">Recursive</option>
                <option value="fixed">Fixed</option>
                <option value="sentence">Sentence</option>
              </select>
            </div>
            <ParamSlider label="Chunk size" value={store.chunkSize} min={128} max={1024} step={64}
              onChange={v => usePipelineStore.setState({ chunkSize: v })} unit=" chars" />
            <ParamSlider label="Overlap" value={store.chunkOverlap} min={0} max={256} step={16}
              onChange={v => usePipelineStore.setState({ chunkOverlap: v })} unit=" chars" />
          </div>
          <button className="btn-primary" onClick={handleChunk} disabled={!store.rawText || loading === 'chunk'}>
            {loading === 'chunk' ? 'Chunking...' : 'Chunk document'}
          </button>
          {chunkOutput && (
            <div className="space-y-2">
              <div className="flex gap-2 flex-wrap">
                <MetricBadge label="chunks" value={chunkOutput.total_chunks} color="green" />
                <MetricBadge label="avg length" value={`${chunkOutput.avg_chunk_length} ch`} />
                <MetricBadge label="avg tokens" value={`~${chunkOutput.avg_token_estimate}`} color="blue" />
              </div>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {chunkOutput.chunks.slice(0, 4).map((c: any) => (
                  <div key={c.index} className="text-xs font-mono bg-gray-950 rounded px-2 py-1.5 border-l-2 border-emerald-800">
                    <span className="text-gray-500">[{c.index}] {c.token_estimate} tokens · </span>
                    {c.text.slice(0, 120)}...
                  </div>
                ))}
                {chunkOutput.total_chunks > 4 && (
                  <p className="text-xs text-gray-600 pl-2">...and {chunkOutput.total_chunks - 4} more chunks</p>
                )}
              </div>
            </div>
          )}
        </div>
      </StageCard>

      <div className="flex justify-center"><div className="w-px h-6 bg-gray-800" /></div>

      {/* ── Stage 4: Embedding + Indexing ────────────────────────────────── */}
      <StageCard title="4 · Embedding Generation + Vector Storage" icon="⊕" status={indexOutput ? 'done' : stageStatus('index')}>
        <div className="space-y-3">
          <p className="text-xs text-gray-500">
            Embeds each chunk with <span className="text-emerald-400">sentence-transformers</span> and stores
            vectors in FAISS HNSW + BM25 sparse index.
          </p>
          <button className="btn-primary" onClick={handleIndex} disabled={!store.chunks.length || loading === 'index'}>
            {loading === 'index' ? 'Embedding...' : `Embed & index ${store.chunks.length} chunks`}
          </button>
          {indexOutput && (
            <div className="flex gap-2 flex-wrap">
              <MetricBadge label="indexed" value={indexOutput.indexed} color="green" />
              <MetricBadge label="model" value={indexOutput.embedding_model} color="blue" />
              <MetricBadge label="embed time" value={`${indexOutput.latency_ms}ms`} color="yellow" />
              <MetricBadge label="faiss size" value={indexOutput.vector_index_size} />
            </div>
          )}
        </div>
      </StageCard>

      <div className="flex justify-center"><div className="w-px h-6 bg-gray-800" /></div>

      {/* ── Stage 5: Similarity Search ────────────────────────────────────── */}
      <StageCard title="5 · Similarity Search + Retrieval" icon="⌖" status={retrievalOutput ? 'done' : stageStatus('retrieve')}>
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              className="flex-1 bg-gray-950 border border-gray-700 rounded px-3 py-1.5 text-xs font-mono text-gray-300 placeholder-gray-600"
              placeholder="Enter your query..."
              value={store.query}
              onChange={e => store.setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleRetrieve()}
            />
            <ParamSlider label="top-k" value={store.topK} min={1} max={10} step={1}
              onChange={store.setTopK} />
          </div>
          <button className="btn-primary" onClick={handleRetrieve} disabled={!store.query || loading === 'retrieve'}>
            {loading === 'retrieve' ? 'Searching...' : 'Hybrid search'}
          </button>
          {retrievalOutput && (
            <div className="space-y-1">
              <div className="flex gap-2 flex-wrap mb-2">
                <MetricBadge label="vector hits" value={retrievalOutput.vector_results_count} color="blue" />
                <MetricBadge label="bm25 hits" value={retrievalOutput.bm25_results_count} />
                <MetricBadge label="RRF k" value={retrievalOutput.rrf_k} />
              </div>
              {retrievalOutput.results.map((r: any) => (
                <div key={r.chunk_id} className="bg-gray-950 border border-gray-800 rounded p-2 text-xs font-mono space-y-1">
                  <div className="flex gap-2">
                    <span className="text-emerald-400">#{r.final_rank}</span>
                    <span className="text-gray-500">RRF {r.rrf_score.toFixed(5)}</span>
                    <span className="text-gray-600">vec:{r.vector_rank ?? '–'} bm25:{r.bm25_rank ?? '–'}</span>
                  </div>
                  <p className="text-gray-300">{r.text.slice(0, 150)}...</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </StageCard>

      <div className="flex justify-center"><div className="w-px h-6 bg-gray-800" /></div>

      {/* ── Stage 6: Prompt Construction ─────────────────────────────────── */}
      <StageCard title="6 · Prompt Construction" icon="✎" status={promptOutput ? 'done' : stageStatus('prompt')}>
        <div className="space-y-3">
          <div className="flex gap-2 items-center">
            <p className="param-label">Template:</p>
            <select
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-300"
              value={store.systemTemplate}
              onChange={e => usePipelineStore.setState({ systemTemplate: e.target.value })}
            >
              <option value="rag_default">RAG Default</option>
              <option value="rag_analytical">Analytical</option>
              <option value="rag_concise">Concise</option>
              <option value="open_ended">Open Ended</option>
            </select>
          </div>
          <button className="btn-primary" onClick={handleBuildPrompt} disabled={!store.retrievedChunks.length || loading === 'prompt'}>
            {loading === 'prompt' ? 'Building...' : 'Build prompt'}
          </button>
          {promptOutput && (
            <div className="space-y-2">
              <TokenBar used={promptOutput.total_tokens} total={promptOutput.context_window} />
              <div className="flex gap-2 flex-wrap">
                {promptOutput.sections.map((s: any) => (
                  <MetricBadge key={s.name} label={s.name} value={`${s.token_count} tok`}
                    color={s.name === 'system' ? 'blue' : s.name === 'context' ? 'green' : 'yellow'} />
                ))}
              </div>
              <JsonViewer data={promptOutput.messages} title="Messages sent to LLM" collapsed />
            </div>
          )}
        </div>
      </StageCard>

      <div className="flex justify-center"><div className="w-px h-6 bg-gray-800" /></div>

      {/* ── Stage 7: LLM Inference ────────────────────────────────────────── */}
      <StageCard title="7 · LLM Inference" icon="▶" status={store.response ? 'done' : stageStatus('infer')}>
        <div className="space-y-3">
          <button className="btn-primary" onClick={handleInfer}
            disabled={!store.promptMessages.length || loading === 'infer' || store.isStreaming}>
            {store.isStreaming ? 'Generating...' : 'Run inference'}
          </button>
          {(store.isStreaming || store.streamingResponse || store.response) && (
            <div className="space-y-2">
              <StreamingText text={store.streamingResponse || store.response} isStreaming={store.isStreaming} />
              {store.latencyMs > 0 && (
                <div className="flex gap-2 flex-wrap">
                  <MetricBadge label="latency" value={`${store.latencyMs}ms`} color="yellow" />
                  <MetricBadge label="tok/s" value={store.tokensPerSecond.toFixed(1)} color="green" />
                </div>
              )}
            </div>
          )}
        </div>
      </StageCard>

      <div className="flex justify-center"><div className="w-px h-6 bg-gray-800" /></div>

      {/* ── Stage 8: Evaluation ───────────────────────────────────────────── */}
      <StageCard title="8 · Evaluation" icon="◎" status={evalOutput ? 'done' : stageStatus('eval')}>
        <div className="space-y-3">
          <button className="btn-primary" onClick={handleEval}
            disabled={!store.response && !store.streamingResponse || loading === 'eval'}>
            {loading === 'eval' ? 'Evaluating...' : 'Run evaluation'}
          </button>
          {evalOutput && (
            <div className="space-y-3">
              <div className="flex gap-2 flex-wrap">
                <MetricBadge label="Precision@K" value={evalOutput.retrieval.precision_at_k} color="green" />
                <MetricBadge label="Recall@K" value={evalOutput.retrieval.recall_at_k} color="blue" />
                <MetricBadge label="grounding" value={evalOutput.hallucination.overall_score} color={evalOutput.hallucination.verdict === 'grounded' ? 'green' : 'yellow'} />
                <MetricBadge label="verdict" value={evalOutput.hallucination.verdict} color={evalOutput.hallucination.verdict === 'hallucinated' ? 'red' : 'green'} />
              </div>
              <div className="text-xs text-gray-500 font-mono space-y-1">
                {evalOutput.hallucination.sentences.map((s: any, i: number) => (
                  <div key={i} className={`flex gap-2 items-start ${s.supported ? 'text-gray-400' : 'text-red-400'}`}>
                    <span>{s.supported ? '✓' : '✗'}</span>
                    <span>{s.sentence.slice(0, 100)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </StageCard>
    </div>
  )
}
