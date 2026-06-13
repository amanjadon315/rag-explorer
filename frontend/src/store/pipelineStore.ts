import { create } from 'zustand'

export type StageStatus = 'idle' | 'running' | 'done' | 'error'

export interface PipelineStore {
  // Document
  rawText: string
  cleanedText: string
  filename: string

  // Chunks
  chunks: Array<{ index: number; text: string; token_estimate: number }>
  chunkStrategy: string
  chunkSize: number
  chunkOverlap: number

  // Retrieval
  query: string
  topK: number
  retrievalMode: 'vector' | 'bm25' | 'hybrid'
  retrievedChunks: string[]
  retrievalScores: number[]

  // Prompt
  builtPrompt: string
  promptMessages: object[]
  totalTokens: number
  contextWindow: number
  utilizationPct: number
  systemTemplate: string

  // Inference
  response: string
  streamingResponse: string
  isStreaming: boolean
  latencyMs: number
  tokensPerSecond: number

  // Evaluation
  evaluation: object | null

  // Stage statuses
  stageStatus: Record<string, StageStatus>

  // Actions
  setRawText: (t: string, filename?: string) => void
  setCleanedText: (t: string) => void
  setChunks: (chunks: any[], strategy: string, size: number, overlap: number) => void
  setQuery: (q: string) => void
  setTopK: (k: number) => void
  setRetrievalMode: (m: 'vector' | 'bm25' | 'hybrid') => void
  setRetrievedChunks: (chunks: string[], scores: number[]) => void
  setBuiltPrompt: (prompt: string, messages: object[], tokens: number, utilization: number) => void
  setSystemTemplate: (t: string) => void
  appendStreamToken: (token: string) => void
  setResponse: (r: string, latency: number, tps: number) => void
  setIsStreaming: (v: boolean) => void
  setEvaluation: (e: object) => void
  setStageStatus: (stage: string, status: StageStatus) => void
  reset: () => void
}

const initial = {
  rawText: '',
  cleanedText: '',
  filename: '',
  chunks: [],
  chunkStrategy: 'recursive',
  chunkSize: 512,
  chunkOverlap: 64,
  query: '',
  topK: 5,
  retrievalMode: 'hybrid' as const,
  retrievedChunks: [],
  retrievalScores: [],
  builtPrompt: '',
  promptMessages: [],
  totalTokens: 0,
  contextWindow: 8192,
  utilizationPct: 0,
  systemTemplate: 'rag_default',
  response: '',
  streamingResponse: '',
  isStreaming: false,
  latencyMs: 0,
  tokensPerSecond: 0,
  evaluation: null,
  stageStatus: {},
}

export const usePipelineStore = create<PipelineStore>((set) => ({
  ...initial,

  setRawText: (t, filename = '') => set({ rawText: t, filename }),
  setCleanedText: (t) => set({ cleanedText: t }),
  setChunks: (chunks, strategy, size, overlap) =>
    set({ chunks, chunkStrategy: strategy, chunkSize: size, chunkOverlap: overlap }),
  setQuery: (q) => set({ query: q }),
  setTopK: (k) => set({ topK: k }),
  setRetrievalMode: (m) => set({ retrievalMode: m }),
  setRetrievedChunks: (chunks, scores) => set({ retrievedChunks: chunks, retrievalScores: scores }),
  setBuiltPrompt: (prompt, messages, tokens, utilization) =>
    set({ builtPrompt: prompt, promptMessages: messages, totalTokens: tokens, utilizationPct: utilization }),
  setSystemTemplate: (t) => set({ systemTemplate: t }),
  appendStreamToken: (token) =>
    set((s) => ({ streamingResponse: s.streamingResponse + token })),
  setResponse: (r, latency, tps) =>
    set({ response: r, streamingResponse: '', latencyMs: latency, tokensPerSecond: tps }),
  setIsStreaming: (v) => set({ isStreaming: v }),
  setEvaluation: (e) => set({ evaluation: e }),
  setStageStatus: (stage, status) =>
    set((s) => ({ stageStatus: { ...s.stageStatus, [stage]: status } })),
  reset: () => set(initial),
}))
