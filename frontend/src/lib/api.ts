import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
})

// ── Ingestion ────────────────────────────────────────────────────────────────
export const uploadFile = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/api/v1/ingestion/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const pasteText = (text: string) =>
  api.post('/api/v1/ingestion/paste', { text })

export const cleanText = (text: string) =>
  api.post('/api/v1/ingestion/clean', { text })

export const chunkText = (text: string, strategy: string, chunkSize: number, overlap: number) =>
  api.post('/api/v1/ingestion/chunk', { text, strategy, chunk_size: chunkSize, overlap })

export const tokenizeText = (text: string) =>
  api.post('/api/v1/ingestion/tokenize', { text })

// ── Embeddings ───────────────────────────────────────────────────────────────
export const generateEmbeddings = (texts: string[]) =>
  api.post('/api/v1/embeddings/generate', { texts })

export const computeSimilarity = (textA: string, textB: string) =>
  api.post('/api/v1/embeddings/similarity', { text_a: textA, text_b: textB })

export const reduceEmbeddings = (texts: string[], labels: string[], method: 'pca' | 'tsne', groups?: string[]) =>
  api.post('/api/v1/embeddings/reduce', { texts, labels, method, groups })

// ── Retrieval ─────────────────────────────────────────────────────────────────
export const indexChunks = (chunks: Array<{ chunk_id: string; text: string }>) =>
  api.post('/api/v1/retrieval/index', { chunks })

export const vectorSearch = (query: string, topK = 5) =>
  api.post('/api/v1/retrieval/vector', { query, top_k: topK })

export const bm25Search = (query: string, topK = 5) =>
  api.post('/api/v1/retrieval/bm25', { query, top_k: topK })

export const hybridSearch = (query: string, topK = 5) =>
  api.post('/api/v1/retrieval/hybrid', { query, top_k: topK })

// ── Prompt ────────────────────────────────────────────────────────────────────
export const buildPrompt = (query: string, contextChunks: string[], template = 'rag_default') =>
  api.post('/api/v1/prompt/build', { query, context_chunks: contextChunks, system_template: template })

export const getTemplates = () => api.get('/api/v1/prompt/templates')

// ── Inference ─────────────────────────────────────────────────────────────────
export const complete = (messages: object[], temperature = 0.7, maxTokens = 1024) =>
  api.post('/api/v1/inference/complete', { messages, temperature, max_tokens: maxTokens })

export const getModels = () => api.get('/api/v1/inference/models')
export const ollamaHealth = () => api.get('/api/v1/inference/health')

// Streaming: returns EventSource-compatible URL
export const streamCompletion = async (
  messages: object[],
  onToken: (token: string) => void,
  onDone: () => void,
) => {
  const resp = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/inference/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, temperature: 0.7, max_tokens: 1024 }),
  })
  const reader = resp.body!.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    const text = decoder.decode(value)
    const lines = text.split('\n').filter(l => l.startsWith('data: '))
    for (const line of lines) {
      const token = line.slice(6)
      if (token === '[DONE]') { onDone(); return }
      if (!token.startsWith('[ERROR]')) onToken(token)
    }
  }
  onDone()
}

// ── Evaluation ────────────────────────────────────────────────────────────────
export const fullEvaluation = (query: string, retrievedChunks: string[], response: string, reference?: string) =>
  api.post('/api/v1/evaluation/full', {
    query,
    retrieved_chunks: retrievedChunks,
    response,
    reference_answer: reference,
  })

export default api
