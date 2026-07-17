<script setup>
import { ref, nextTick, onMounted } from "vue"
import { streamChat, fetchModel } from "./lib/api.js"
import ExpertBrain from "./components/ExpertBrain.vue"

const model = ref("inkling-colibri")
const messages = ref([])          // {role, content, think, phase}
const draft = ref("")
const busy = ref(false)
const reasoning = ref(localStorage.getItem("sabrewing.reasoning") === "1")
const thread = ref(null)
let controller = null

const PROMPTS = [
  "Explain how a hummingbird hovers.",
  "Write a haiku about dusk.",
  "What makes an MoE model efficient?",
  "Translate 'the night is quiet' into three languages.",
]

onMounted(async () => { model.value = await fetchModel() })

function toggleReasoning() {
  reasoning.value = !reasoning.value
  localStorage.setItem("sabrewing.reasoning", reasoning.value ? "1" : "0")
}

function autosize(e) {
  e.target.style.height = "auto"
  e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px"
}

async function scrollDown() {
  await nextTick()
  thread.value?.scrollTo({ top: thread.value.scrollHeight, behavior: "smooth" })
}

async function send(text) {
  const content = (text ?? draft.value).trim()
  if (!content || busy.value) return
  draft.value = ""
  messages.value.push({ role: "user", content, think: "", phase: "done" })
  // phase: reasoning models open in "thinking" and flip to "answering" on first
  // content token; non-reasoning turns go straight to "answering".
  const turn = { role: "assistant", content: "", think: "", phase: reasoning.value ? "thinking" : "answering" }
  messages.value.push(turn)
  busy.value = true
  controller = new AbortController()
  scrollDown()

  const history = messages.value.slice(0, -1).map((m) => ({ role: m.role, content: m.content }))

  try {
    await streamChat(history, {
      model: model.value,
      temperature: 0.7,
      reasoning: reasoning.value,
      signal: controller.signal,
      // ignore lone-"." keepalive pings (fired during slow cold prefill) — only
      // real reasoning tokens open the box, so reasoning-off never shows it
      onThink: (t) => { if (t === ".") return; turn.phase = "thinking"; turn.think += t; scrollDown() },
      onToken: (t) => { turn.phase = "answering"; turn.content += t; scrollDown() },
    })
  } catch (err) {
    if (err.name !== "AbortError") turn.content += `${turn.content ? "\n\n" : ""}⚠ ${err.message}`
  } finally {
    turn.phase = "done"
    busy.value = false
    controller = null
  }
}

function stop() { controller?.abort() }
function reset() { messages.value = [] }
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <img class="glyph" src="/sabrewing.svg" alt="" />
        <div>
          <h1>sabrewing</h1>
          <div class="sub">{{ model }}</div>
        </div>
      </div>
      <div class="spacer" />
      <button class="toggle" :class="{ on: reasoning }" @click="toggleReasoning" :title="reasoning ? 'Reasoning on' : 'Reasoning off'">
        <span class="knob" /> Reasoning
      </button>
      <div class="stat"><span class="dot" :class="{ cold: !busy }" /> {{ busy ? "generating" : "ready" }}</div>
      <button v-if="messages.length" class="ghost-btn" @click="reset">Clear</button>
    </header>

    <main class="stage with-aside">
      <section class="chat">
        <div v-if="!messages.length" class="empty">
          <img class="big" src="/sabrewing.svg" alt="" />
          <h2>A trillion parameters, on your hardware.</h2>
          <p>Streaming straight from the machine in front of you — nothing leaves this endpoint.</p>
          <div class="chips">
            <button v-for="p in PROMPTS" :key="p" class="chip" @click="send(p)">{{ p }}</button>
          </div>
        </div>

        <div v-else class="thread" ref="thread">
          <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
            <div class="who">{{ m.role === "user" ? "You" : "◈" }}</div>
            <div class="body">
              <div class="name">{{ m.role === "user" ? "You" : "sabrewing" }}</div>

              <!-- reasoning: opens immediately while thinking, collapsible once answering -->
              <details v-if="m.think || m.phase === 'thinking'" class="think" :open="m.phase === 'thinking'">
                <summary>
                  <span v-if="m.phase === 'thinking'" class="think-live">thinking<span class="typing"><i /><i /><i /></span></span>
                  <span v-else>reasoning</span>
                </summary>
                <div class="think-body">{{ m.think || "…" }}</div>
              </details>

              <div v-if="m.content || m.phase !== 'thinking'" class="bubble">
                <template v-if="m.content">{{ m.content }}</template>
                <span v-else-if="busy && i === messages.length - 1" class="typing" aria-label="Generating"><i /><i /><i /></span>
              </div>
            </div>
          </div>
        </div>

        <div class="panel composer">
          <textarea
            v-model="draft"
            :placeholder="busy ? 'generating…' : 'Message sabrewing…'"
            rows="1"
            @input="autosize"
            @keydown.enter.exact.prevent="send()"
          />
          <button v-if="busy" class="send" @click="stop" title="Stop">
            <svg viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="2" /></svg>
          </button>
          <button v-else class="send" :disabled="!draft.trim()" @click="send()" title="Send">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
          </button>
        </div>
      </section>

      <ExpertBrain />
    </main>
  </div>
</template>
