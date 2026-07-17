// sabrewing gateway client — chat SSE + brain/profile polling. no deps.
import { settings, authHeaders } from "./settings.js"

const base = () => settings.baseUrl || ""

// Stream a chat completion. onToken(text) per content delta; onThink(text) per
// reasoning delta. Aborts via signal. Sampling comes from shared settings.
const CTRL = /<\|[^|]*\|>/g   // any Inkling control marker, belt-and-suspenders

export async function streamChat(messages, { model, reasoning, signal, onToken, onThink }) {
  const body = {
    model, messages, stream: true,
    temperature: settings.temperature,
    max_tokens: settings.maxTokens,
  }
  if (reasoning) body.reasoning_effort = "high"
  const res = await fetch(`${base()}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
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
      if (d.content) onToken?.(d.content.replace(CTRL, ""))
      if (d.reasoning_content) onThink?.(d.reasoning_content.replace(CTRL, ""))
    }
  }
}

export async function fetchModel() {
  try {
    const r = await fetch(`${base()}/v1/models`, { headers: authHeaders() })
    const j = await r.json()
    return j.data?.[0]?.id || "inkling-colibri"
  } catch { return "inkling-colibri" }
}

// /experts -> { rows, cols, map (hex, 1 byte/expert: tier<<6 | heat) }
export async function fetchExperts() {
  try {
    const r = await fetch(`${base()}/experts`, { headers: authHeaders() })
    return await r.json()
  } catch { return null }
}

export async function fetchHealth() {
  try {
    const r = await fetch(`${base()}/health`, { headers: authHeaders() })
    return await r.json()
  } catch { return null }
}

// /profile -> { seq, turns: [{wall_s, prompt_tokens, completion_tokens,
//   expert_disk_s, expert_wait_s, expert_matmul_s, attention_s, forwards}] }
export async function fetchProfile() {
  try {
    const r = await fetch(`${base()}/profile`, { headers: authHeaders() })
    return await r.json()
  } catch { return null }
}
