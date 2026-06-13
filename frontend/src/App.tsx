import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import PipelineView from './pages/PipelineView'
import EmbeddingExplorer from './pages/EmbeddingExplorer'
import RetrievalExplorer from './pages/RetrievalExplorer'
import PromptInspector from './pages/PromptInspector'
import InferenceDashboard from './pages/InferenceDashboard'
import EvaluationPanel from './pages/EvaluationPanel'
import './index.css'

const NAV = [
  { to: '/',            label: 'Pipeline',   icon: '⬡' },
  { to: '/embeddings',  label: 'Embeddings', icon: '⊕' },
  { to: '/retrieval',   label: 'Retrieval',  icon: '⌖' },
  { to: '/prompt',      label: 'Prompt',     icon: '✎' },
  { to: '/inference',   label: 'Inference',  icon: '▶' },
  { to: '/evaluation',  label: 'Evaluation', icon: '◎' },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-950 text-gray-100 overflow-hidden">
        {/* Sidebar */}
        <nav className="w-48 shrink-0 border-r border-gray-800 flex flex-col py-6 gap-1">
          <div className="px-4 mb-6">
            <p className="text-xs text-gray-500 font-mono uppercase tracking-widest">RAG Explorer</p>
            <p className="text-xs text-gray-600 mt-1">LLM Pipeline Inspector</p>
          </div>
          {NAV.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2 text-sm font-mono transition-colors
                 ${isActive
                   ? 'text-emerald-400 bg-emerald-400/10 border-r-2 border-emerald-400'
                   : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                 }`
              }
            >
              <span className="text-base">{icon}</span>
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/"           element={<PipelineView />} />
            <Route path="/embeddings" element={<EmbeddingExplorer />} />
            <Route path="/retrieval"  element={<RetrievalExplorer />} />
            <Route path="/prompt"     element={<PromptInspector />} />
            <Route path="/inference"  element={<InferenceDashboard />} />
            <Route path="/evaluation" element={<EvaluationPanel />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
