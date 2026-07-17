import { reactive, watch } from "vue"

// Shared, persisted settings. Both the UI and the API client read this object,
// so a change in the panel takes effect on the next request with no plumbing.
const DEFAULTS = {
  baseUrl: "",       // "" = same origin the UI is served from
  apiKey: "",        // sent as Authorization: Bearer <key> when set
  temperature: 0.7,
  maxTokens: 1024,
}

function load() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("sabrewing.settings") || "{}") }
  } catch {
    return { ...DEFAULTS }
  }
}

export const settings = reactive(load())

watch(settings, (s) => localStorage.setItem("sabrewing.settings", JSON.stringify(s)), { deep: true })

export function resetSettings() {
  Object.assign(settings, DEFAULTS)
}

// Authorization header when an API key is configured.
export function authHeaders() {
  return settings.apiKey ? { Authorization: `Bearer ${settings.apiKey}` } : {}
}
