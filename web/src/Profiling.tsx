import { useEffect, useState } from "react"
import { Activity, Gauge, HardDrive, Timer } from "lucide-react"

import { getProfile, type ProfileTurn } from "@/lib/api"

/* Wall-time phases stacked per turn. The order is the palette's CVD-safe slot
 * order (validated as a set on this surface) — identity never leans on colour
 * alone: segments keep 2px gaps, the legend is always shown and the table
 * carries the exact numbers. Disk *service* time is reported separately: it
 * runs on I/O threads overlapped with compute, so only the stall the compute
 * thread actually felt (I/O wait) belongs inside the wall-time stack. */
const PHASES = [
  { key: "expert_wait_s", name: "I/O wait", color: "#3987e5" },
  { key: "expert_matmul_s", name: "Expert matmul", color: "#199e70" },
  { key: "attention_s", name: "Attention", color: "#c98500" },
  { key: "lm_head_s", name: "LM head", color: "#008300" },
  { key: "other_s", name: "Other", color: "#9085e9" },
] as const

interface Turn extends ProfileTurn { other_s: number; toks: number }

const derive = (turn: ProfileTurn): Turn => ({
  ...turn,
  other_s: Math.max(0, turn.wall_s - turn.expert_wait_s - turn.expert_matmul_s - turn.attention_s - turn.lm_head_s),
  toks: turn.wall_s > 0 ? turn.completion_tokens / turn.wall_s : 0,
})

const seconds = (value: number) => (value >= 10 ? value.toFixed(1) : value.toFixed(2)) + "s"

function ShareBar({ label, turns }: { label: string; turns: Turn[] }) {
  const total = turns.reduce((sum, turn) => sum + turn.wall_s, 0)
  const parts = PHASES.map((phase) => ({ ...phase, value: turns.reduce((sum, turn) => sum + turn[phase.key], 0) }))
  return (
    <div className="prof-share">
      <div className="prof-share-head"><span>{label}</span><code>{seconds(total)}</code></div>
      <div className="prof-share-bar" role="img" aria-label={parts.map((part) => `${part.name} ${seconds(part.value)}`).join(", ")}>
        {parts.map((part) => {
          const share = total > 0 ? part.value / total : 0
          return share > 0.001 ? (
            <span key={part.key} style={{ width: `${100 * share}%`, background: part.color }} title={`${part.name} — ${seconds(part.value)} (${(100 * share).toFixed(1)}%)`}>
              {share >= 0.09 ? `${Math.round(100 * share)}%` : ""}
            </span>
          ) : null
        })}
      </div>
    </div>
  )
}

/* Column chart over the recent turns; oldest on the left. Stacked mode draws the
 * wall-time composition, plain mode a single series (no legend — the title names it). */
function TurnColumns({ turns, stacked, height, format }: { turns: Turn[]; stacked: boolean; height: number; format: (turn: Turn) => string }) {
  const [hover, setHover] = useState<number | null>(null)
  const peak = Math.max(...turns.map((turn) => (stacked ? turn.wall_s : turn.toks)), 1e-9)
  const gap = 2
  const width = Math.max(1, (100 - gap * (turns.length - 1)) / turns.length)
  return (
    <div className="prof-plot" onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" aria-hidden="true">
        {[0.25, 0.5, 0.75].map((line) => <line key={line} x1="0" x2="100" y1={height * line} y2={height * line} className="prof-grid" />)}
        {turns.map((turn, index) => {
          const x = index * (width + gap)
          if (!stacked) {
            const h = (height * turn.toks) / peak
            return <rect key={index} x={x} y={height - h} width={width} height={h} rx="1" fill="var(--primary)" opacity={hover === null || hover === index ? 1 : 0.45} />
          }
          let y = height
          return PHASES.map((phase) => {
            const h = (height * turn[phase.key]) / peak
            y -= h
            return h > 0.1 ? <rect key={`${index}-${phase.key}`} x={x} y={y + 0.35} width={width} height={Math.max(h - 0.7, 0.35)} fill={phase.color} opacity={hover === null || hover === index ? 1 : 0.45} /> : null
          })
        })}
        {/* hit targets bigger than the marks */}
        {turns.map((_, index) => <rect key={index} x={index * (width + gap) - gap / 2} y="0" width={width + gap} height={height} fill="transparent" onMouseEnter={() => setHover(index)} />)}
      </svg>
      <div className="prof-plot-foot">
        <span>{turns.length > 1 ? `${turns.length} turns · oldest → newest` : "1 turn"}</span>
        <code>{hover !== null && turns[hover] ? format(turns[hover]) : `peak ${stacked ? seconds(peak) : peak.toFixed(1) + " tok/s"}`}</code>
      </div>
    </div>
  )
}

export function Profiling({ baseUrl, apiKey, connected }: { baseUrl: string; apiKey: string; connected: boolean }) {
  const [turns, setTurns] = useState<Turn[]>([])

  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getProfile(baseUrl, apiKey)
        if (!disposed) setTurns(result.turns.map(derive))
      } catch { /* engine busy or restarting — keep the last snapshot */ }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 2000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [baseUrl, apiKey, connected])

  const latest = turns[turns.length - 1]
  const recent = turns.slice(-40)
  const diskService = turns.reduce((sum, turn) => sum + turn.expert_disk_s, 0)

  return (
    <div className="prof-page">
      <div className="prof-head">
        <div className="section-title"><Gauge className="size-4" /> Profiling — where the engine spends each turn</div>
        <div className="prof-legend">
          {PHASES.map((phase) => <span key={phase.key}><i style={{ background: phase.color }} />{phase.name}</span>)}
        </div>
      </div>

      {!latest ? (
        <p className="runtime-unavailable">{connected ? "No profiled turns yet — send a chat message and the breakdown appears here." : "Connect to the engine to collect per-turn timings."}</p>
      ) : (
        <>
          <div className="prof-tiles">
            <div><span><Gauge className="size-3" /> Last turn</span><strong>{latest.toks.toFixed(1)}</strong><small>tok/s</small></div>
            <div><span><Timer className="size-3" /> Wall time</span><strong>{seconds(latest.wall_s)}</strong><small>{latest.prompt_tokens} → {latest.completion_tokens} tokens</small></div>
            <div><span><Activity className="size-3" /> Batching</span><strong>{latest.forwards > 0 ? (latest.completion_tokens / latest.forwards).toFixed(2) : "—"}</strong><small>tokens / forward</small></div>
            <div><span><HardDrive className="size-3" /> Disk service</span><strong>{seconds(latest.expert_disk_s)}</strong><small>overlapped with compute</small></div>
          </div>

          <div className="prof-shares">
            <ShareBar label="Last turn" turns={[latest]} />
            {turns.length > 1 ? <ShareBar label={`Window · last ${turns.length} turns`} turns={turns} /> : null}
          </div>

          <div className="prof-charts">
            <div className="prof-chart">
              <div className="prof-chart-title">Throughput per turn (tok/s)</div>
              <TurnColumns turns={recent} stacked={false} height={36} format={(turn) => `${turn.toks.toFixed(1)} tok/s · ${turn.completion_tokens} tokens`} />
            </div>
            <div className="prof-chart">
              <div className="prof-chart-title">Turn wall time by phase (s)</div>
              <TurnColumns turns={recent} stacked height={36} format={(turn) => `${seconds(turn.wall_s)} · ${PHASES.map((phase) => `${phase.name} ${seconds(turn[phase.key])}`).join(" · ")}`} />
            </div>
          </div>

          <div className="prof-table-wrap">
            <table className="prof-table">
              <thead><tr><th>Turn</th><th>Tokens</th><th>tok/s</th><th>Wall</th>{PHASES.map((phase) => <th key={phase.key}><i style={{ background: phase.color }} />{phase.name}</th>)}<th>Disk service</th></tr></thead>
              <tbody>
                {recent.slice().reverse().map((turn, index) => (
                  <tr key={turns.length - index}>
                    <td>{turns.length - index}</td>
                    <td>{turn.prompt_tokens} → {turn.completion_tokens}</td>
                    <td>{turn.toks.toFixed(1)}</td>
                    <td>{seconds(turn.wall_s)}</td>
                    {PHASES.map((phase) => <td key={phase.key}>{seconds(turn[phase.key])}</td>)}
                    <td>{seconds(turn.expert_disk_s)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {diskService > 0 ? <p className="prof-note">Disk service is time spent reading experts on I/O threads; it overlaps with compute, so only the <em>I/O wait</em> the compute thread felt counts inside the wall-time stack. With multiple KV sessions the shares describe the whole engine over the turn's window.</p> : null}
          </div>
        </>
      )}
    </div>
  )
}
