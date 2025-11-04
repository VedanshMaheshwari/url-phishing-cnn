const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export type PredictResponse = {
  phish_probability: number
  predicted_label: number
}

export async function predict(url: string, signal?: AbortSignal): Promise<PredictResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
    signal,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Request failed ${res.status}: ${text}`)
  }
  return res.json()
}
