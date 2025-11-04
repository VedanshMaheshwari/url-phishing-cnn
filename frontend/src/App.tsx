import { useMemo, useRef, useState } from 'react'
import { predict } from './api'

export default function App() {
  const [url, setUrl] = useState('http://paypal.com.account-verify.co/enter')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [prob, setProb] = useState<number | null>(null)
  const [label, setLabel] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const riskText = useMemo(() => {
    if (prob == null) return '-'
    const pct = (prob * 100).toFixed(2)
    return `${pct}% ${label ? 'Phishing' : 'Legit'}`
  }, [prob, label])

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setProb(null)
    setLabel(null)
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    try {
      setLoading(true)
      const res = await predict(url, ac.signal)
      setProb(res.phish_probability)
      setLabel(res.predicted_label)
    } catch (err: any) {
      setError(err?.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Phishing URL Detector</h1>
        <p>Character-level CNN (PyTorch) served by FastAPI</p>
      </header>

      <form onSubmit={onSubmit} className="card">
        <label htmlFor="url">URL</label>
        <input
          id="url"
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/login"
          autoFocus
        />
        <button type="submit" disabled={loading || !url.trim()}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      <section className="results">
        <div className="result-card">
          <h3>Result</h3>
          <div className={`badge ${label === 1 ? 'bad' : label === 0 ? 'good' : ''}`}>{riskText}</div>
          {prob != null && (
            <p className="mono">prob={prob.toFixed(6)} label={label}</p>
          )}
          {error && <p className="error">{error}</p>}
        </div>
      </section>

      <footer>
        <small>API: {import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}</small>
      </footer>
    </div>
  )
}
