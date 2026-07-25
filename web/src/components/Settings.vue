<script setup>
import { settings, resetSettings } from "../lib/settings.js"
defineProps({ open: Boolean })
defineEmits(["close"])
</script>

<template>
  <div v-if="open" class="modal-scrim" @click.self="$emit('close')">
    <div class="modal panel" role="dialog" aria-label="Settings">
      <div class="modal-head">
        <h3>Settings</h3>
        <button class="ghost-btn" @click="$emit('close')">Done</button>
      </div>

      <label class="field">
        <span>Temperature <b>{{ settings.temperature.toFixed(2) }}</b></span>
        <input type="range" min="0" max="1.5" step="0.05" v-model.number="settings.temperature" />
        <small>0 is deterministic (greedy); higher is more varied.</small>
      </label>

      <label class="field">
        <span>Top-p <b>{{ settings.topP.toFixed(2) }}</b></span>
        <input type="range" min="0.1" max="1" step="0.01" v-model.number="settings.topP" />
        <small>Nucleus mass kept when sampling (1.0 = off).</small>
      </label>

      <label class="field">
        <span>Max tokens <b>{{ settings.maxTokens }}</b></span>
        <input type="range" min="32" max="4096" step="32" v-model.number="settings.maxTokens" />
        <small>Longest reply before the engine stops.</small>
      </label>

      <label class="field">
        <span>Reasoning effort</span>
        <select v-model="settings.reasoningEffort">
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
        </select>
        <small>Depth of the reasoning pass (when the Reasoning toggle is on).</small>
      </label>

      <label class="field">
        <span>System prompt</span>
        <textarea rows="3" v-model="settings.systemPrompt" placeholder="Optional — steer the model's role/behavior"></textarea>
        <small>Prepended to every conversation. Leave blank for the model's default.</small>
      </label>

      <label class="field">
        <span>Endpoint</span>
        <input type="text" v-model.trim="settings.baseUrl" placeholder="(same origin) — or http://host:8000" />
        <small>Point at a sabrewing server on another machine.</small>
      </label>

      <label class="field">
        <span>API key</span>
        <input type="password" v-model="settings.apiKey" placeholder="only if the server requires one" autocomplete="off" />
        <small>Sent as a bearer token; stays in this browser.</small>
      </label>

      <button class="ghost-btn reset" @click="resetSettings">Reset to defaults</button>
    </div>
  </div>
</template>
