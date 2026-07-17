// sabrewing gateway client — chat SSE + expert-brain polling. no deps.

const BASE = localStorage.getItem("sabrewing.base") || ""

export function setBase(url) {
  localStorage.setItem("sabrewing.base", url)
}
export function getBase() {
  return BASE
}

// Stream a chat completion. onToken(text) per content delta; onThink(text) per
// reasoning delta. Returns the final usage/stats object. Aborts via signal.
export async function streamChat(messages, { model, temperature, signal, onToken, onThink }) {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, temperature, stream: true, max_tokens: 1024 }),
    signal,
  })
  if (!res.ok) throw new Error(`gateway ${res.status}: ${(await res.text()).slice(0, 200)}`)

  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let nl
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim()
      buf = buf.slice(nl + 1)
      if (!line.startsWith("data:")) continue
      const payload = line.slice(5).trim()
      if (payload === "[DONE]") return
      let evt
      try { evt = JSON.parse(payload) } catch { continue }
      const d = evt.choices?.[0]?.delta || {}
      if (d.content) onToken?.(d.content)
      if (d.reasoning_content) onThink?.(d.reasoning_content)
    }
  }
}

export async function fetchModel() {
  try {
    const r = await fetch(`${BASE}/v1/models`)
    const j = await r.json()
    return j.data?.[0]?.id || "inkling-colibri"
  } catch { return "inkling-colibri" }
}

// /experts -> { rows, cols, map (hex, 1 byte/expert: tier<<6 | heat) }
export async function fetchExperts() {
  try {
    const r = await fetch(`${BASE}/experts`)
    return await r.json()
  } catch { return null }
}

export async function fetchHealth() {
  try {
    const r = await fetch(`${BASE}/health`)
    return await r.json()
  } catch { return null }
}
