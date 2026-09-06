import { useState } from "react"

function ScoreBar({ score }) {
  const pct = Math.round(score * 100)
  let color = "var(--accent-green)"
  if (pct < 60) color = "var(--accent-orange)"
  if (pct < 40) color = "var(--accent-red)"
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, background: "var(--bg-tertiary)", borderRadius: 6, height: 8, overflow: "hidden", border: "1px solid var(--border)" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 6, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 700, color, minWidth: 32 }}>{pct}%</span>
    </div>
  )
}

const STATUS_COLORS = {
  "BANNED": { bg: "rgba(248,81,73,0.15)", border: "rgba(248,81,73,0.4)", text: "var(--accent-red)" },
  "RESTRICTED": { bg: "rgba(210,153,34,0.15)", border: "rgba(210,153,34,0.4)", text: "var(--accent-orange)" },
  "APPROVED": { bg: "rgba(63,185,80,0.12)", border: "rgba(63,185,80,0.4)", text: "var(--accent-green)" },
  "CAUTION": { bg: "rgba(210,153,34,0.12)", border: "rgba(210,153,34,0.3)", text: "var(--accent-orange)" },
  "SAFE": { bg: "rgba(63,185,80,0.12)", border: "rgba(63,185,80,0.4)", text: "var(--accent-green)" },
}

function statusStyle(label = "") {
  const key = Object.keys(STATUS_COLORS).find(k => label?.toUpperCase().includes(k)) || "CAUTION"
  return STATUS_COLORS[key]
}

function PesticideCard({ p }) {
  const s = statusStyle(p.india_label)
  return (
    <div style={{
      background: "var(--bg-card)",
      border: `1px solid ${s.border}`,
      borderRadius: "var(--radius-sm)",
      padding: "14px 16px",
      borderLeft: `4px solid ${s.text}`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)" }}>
          {p.common_names?.[0] || p.pesticide}
        </span>
        <span style={{
          padding: "2px 8px",
          borderRadius: 12,
          fontSize: 11,
          fontWeight: 600,
          background: s.bg,
          color: s.text,
          border: `1px solid ${s.border}`,
        }}>
          {p.verdict}
        </span>
      </div>

      <ScoreBar score={p.compliance_score} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: "6px 20px", marginTop: 12, fontSize: 12 }}>
        {[
          ["🇮🇳 India Status", p.india_label],
          ["🌍 WHO Class", p.who_label],
          ["🇪🇺 EU Risk", p.eu_label],
          ["🌿 Organic", p.organic_label],
          ["⏰ PHI Days", p.pre_harvest_interval ? `${p.pre_harvest_interval} days` : "—"],
          ["📊 Codex MRL", p.codex_mrl ? `${p.codex_mrl} mg/kg` : "—"],
        ].map(([l, v], i) => (
          <div key={i}>
            <span style={{ color: "var(--text-muted)" }}>{l}: </span>
            <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{v || "—"}</span>
          </div>
        ))}
      </div>

      {p.organic_alternative && p.organic_alternative !== "No alternative mapped" && (
        <div style={{
          marginTop: 10,
          padding: "6px 10px",
          background: "rgba(63,185,80,0.08)",
          borderRadius: 6,
          border: "1px solid rgba(63,185,80,0.25)",
          fontSize: 12,
          color: "var(--accent-green)",
        }}>
          🌿 Organic Alternative: {p.organic_alternative}
        </div>
      )}

      {p.notes && (
        <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-muted)", fontStyle: "italic" }}>
          {p.notes}
        </div>
      )}
    </div>
  )
}

export default function CompliancePanel({ compliance }) {
  const [open, setOpen] = useState(true)
  const { pesticides_found, scored_pesticides, overall_compliance, any_banned, any_eu_banned, warning_needed } = compliance

  const bannerColor = any_banned
    ? { bg: "rgba(248,81,73,0.15)", border: "rgba(248,81,73,0.4)", icon: "🚫" }
    : { bg: "rgba(210,153,34,0.15)", border: "rgba(210,153,34,0.4)", icon: "⚠️" }

  return (
    <div style={{
      border: `1px solid ${bannerColor.border}`,
      borderRadius: "var(--radius)",
      overflow: "hidden",
      background: bannerColor.bg,
    }}>
      {/* Banner */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          padding: "14px 20px",
          display: "flex",
          alignItems: "center",
          gap: 10,
          cursor: "pointer",
        }}
      >
        <span style={{ fontSize: 20 }}>{bannerColor.icon}</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: any_banned ? "var(--accent-red)" : "var(--accent-orange)" }}>
            Regulatory Compliance Alert
          </div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {pesticides_found?.length} pesticide(s) detected
            {any_banned && " · 🚫 India banned"}
            {any_eu_banned && " · 🇪🇺 EU banned"}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            Overall: {Math.round((overall_compliance || 1) * 100)}%
          </span>
          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{open ? "▲" : "▼"}</span>
        </div>
      </div>

      {open && (
        <div style={{ padding: "0 16px 16px", background: "var(--bg-secondary)" }}>
          {/* Overall score bar */}
          <div style={{ marginBottom: 16, padding: "12px 14px", background: "var(--bg-card)", borderRadius: 8, border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>Overall Legal-Efficacy Score</div>
            <ScoreBar score={overall_compliance || 1} />
          </div>

          {/* Pesticide cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {scored_pesticides?.map((p, i) => <PesticideCard key={i} p={p} />)}
          </div>

          {/* Raw warning box */}
          {compliance.warning_box && (
            <pre style={{
              marginTop: 12,
              padding: "10px 14px",
              background: "var(--bg-tertiary)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
              color: "var(--text-secondary)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontFamily: "monospace",
            }}>
              {compliance.warning_box}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
