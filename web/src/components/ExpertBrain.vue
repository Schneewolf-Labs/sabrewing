<script setup>
import { ref, onMounted, onUnmounted, computed } from "vue"
import { fetchExperts, fetchHealth } from "../lib/api.js"

const experts = ref(null)
const health = ref(null)
let timer = null

// downsample the full expert map (up to ~16k cells) to a legible grid; each
// display cell takes the peak heat of its bucket, colored violet->gorget by heat.
const cells = computed(() => {
  const e = experts.value
  if (!e?.map || !e.rows) return []
  const bytes = []
  for (let i = 0; i < e.map.length; i += 2) bytes.push(parseInt(e.map.slice(i, i + 2), 16))
  const MAX = 384
  const stride = Math.max(1, Math.ceil(bytes.length / MAX))
  const out = []
  for (let i = 0; i < bytes.length; i += stride) {
    let tier = 0, heat = 0
    for (let j = i; j < Math.min(i + stride, bytes.length); j++) {
      tier = Math.max(tier, bytes[j] >> 6)
      heat = Math.max(heat, bytes[j] & 63)
    }
    out.push({ tier, heat })
  }
  return out
})

function cellStyle(c) {
  if (c.tier === 0 && c.heat === 0) return { "--cell": "var(--night-2)" }
  const t = Math.min(1, c.heat / 16)          // heat 0..~16 bits -> 0..1
  const hue = (292 + t * 113) % 360            // violet(292) -> magenta -> ember(45), never green
  const light = 40 + t * 34
  const chroma = 0.09 + t * 0.11
  return { "--cell": `oklch(${light}% ${chroma} ${hue})` }
}

const residentGb = computed(() => health.value?.tiers?.ram_gb?.toFixed(1))
const vram = computed(() => health.value?.hwinfo?.vram_total_gb)
const cpu = computed(() => health.value?.hwinfo?.cpu)

async function poll() {
  experts.value = await fetchExperts()
  health.value = await fetchHealth()
}
onMounted(() => { poll(); timer = setInterval(poll, 2500) })
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <aside class="aside">
    <div class="panel card">
      <h3>Expert activity</h3>
      <div v-if="cells.length" class="brain-grid">
        <i v-for="(c, i) in cells" :key="i" :style="cellStyle(c)" />
      </div>
      <p v-else class="stat" style="color: var(--mist-faint)">warming up — send a message</p>
    </div>

    <div class="panel card" v-if="health">
      <h3>Machine</h3>
      <div class="meter">
        <div class="row" v-if="cpu"><span>CPU</span><b style="font-size: var(--step--1)">{{ cpu.split(" ").slice(0, 3).join(" ") }}</b></div>
        <div class="row" v-if="health.hwinfo?.ram_total_gb"><span>RAM</span><b>{{ Math.round(health.hwinfo.ram_total_gb) }} GB</b></div>
        <div class="row" v-if="vram"><span>VRAM</span><b>{{ Math.round(vram) }} GB</b></div>
        <div class="row" v-if="residentGb"><span>Experts cached</span><b>{{ residentGb }} GB</b></div>
      </div>
    </div>
  </aside>
</template>
