<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue"
import { fetchProfile } from "../lib/api.js"

const turn = ref(null)
let timer = null

// phases the engine reports, in wall-time order, with sabrewing labels
const PHASES = [
  { key: "expert_disk_s",   label: "Expert I/O",   hue: 292 },  // fill (streaming)
  { key: "expert_matmul_s", label: "Expert matmul", hue: 330 }, // routed experts
  { key: "expert_wait_s",   label: "Shared experts", hue: 20 },
  { key: "attention_s",     label: "Attention",    hue: 45 },
]

const phases = computed(() => {
  const t = turn.value
  if (!t) return []
  const measured = PHASES.reduce((a, p) => a + (t[p.key] || 0), 0)
  const other = Math.max(0, t.wall_s - measured)
  const rows = PHASES.map((p) => ({ label: p.label, s: t[p.key] || 0, hue: p.hue }))
  rows.push({ label: "Other", s: other, hue: 275 })
  const total = rows.reduce((a, r) => a + r.s, 0) || 1
  return rows.map((r) => ({ ...r, pct: 100 * r.s / total })).filter((r) => r.s > 0.0005)
})

const tps = computed(() => {
  const t = turn.value
  return t && t.wall_s > 0 ? (t.completion_tokens / t.wall_s).toFixed(2) : "—"
})

async function poll() {
  const p = await fetchProfile()
  if (p?.turns?.length) turn.value = p.turns[p.turns.length - 1]
}
onMounted(() => { poll(); timer = setInterval(poll, 2000) })
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div class="panel card" v-if="turn">
    <h3>Last turn · {{ tps }} tok/s</h3>
    <div class="phase-bar">
      <span v-for="p in phases" :key="p.label"
            :style="{ width: p.pct + '%', background: `oklch(62% 0.15 ${p.hue})` }" :title="`${p.label} ${p.s.toFixed(2)}s`" />
    </div>
    <div class="phase-legend">
      <div v-for="p in phases" :key="p.label" class="phase-row">
        <span class="swatch" :style="{ background: `oklch(62% 0.15 ${p.hue})` }" />
        <span class="phase-name">{{ p.label }}</span>
        <b>{{ p.s.toFixed(2) }}s</b>
      </div>
    </div>
  </div>
</template>
