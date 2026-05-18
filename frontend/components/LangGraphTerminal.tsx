"use client";

import { memo, useMemo } from "react";
import type { OptimizeResponse } from "@/lib/api";

type Line = {
  label: string;
  value: string;
  valueColor?: string;
  highlight?: string;
  indent?: boolean;
};

const MONO: React.CSSProperties = { fontFamily: "var(--font-geist-mono)" };
const LABEL_W = "88px";

function modelColor(model: string): string {
  const m = model.toLowerCase();
  if (m.includes("opus")) return "#a78bfa";
  if (m.includes("sonnet")) return "#00A0AE";
  if (m.includes("haiku")) return "#4ade80";
  return "#ffffff";
}

function modelLabel(model: string): string {
  const m = model.toLowerCase();
  if (m.includes("opus")) return "Opus 4.6";
  if (m.includes("sonnet")) return "Sonnet 4.6";
  if (m.includes("haiku")) return "Haiku 4.5";
  return model.replace("claude-", "");
}

function buildLines(result: OptimizeResponse): Line[] {
  const lines: Line[] = [];
  const k = result.complexity;
  const sigma = result.specificity;
  const ambiguity = Math.max(0, Math.min(1, 1 - sigma));
  const intent = result.task_type.toUpperCase();
  const mLabel = modelLabel(result.model);
  const mColor = modelColor(result.model);
  const isOpus = result.model.toLowerCase().includes("opus");

  lines.push({ label: "cache_lookup", value: "miss" });
  lines.push({
    label: "fast_intent",
    value: `${intent} · conf ${(0.6 + sigma * 0.4).toFixed(2)}`,
    highlight: intent,
  });
  lines.push({
    label: "classify",
    value: `complexity ${(k / 10).toFixed(2)} · specificity ${sigma.toFixed(2)} · ambiguity ${ambiguity.toFixed(2)}`,
  });

  if (result.requires_clarification && result.clarification_questions.length > 0) {
    lines.push({
      label: "clarify",
      value: `ambiguity ${ambiguity.toFixed(2)} ≥ 0.60 — generating questions`,
    });
    for (const q of result.clarification_questions) {
      lines.push({ label: "?", value: q.question, indent: true });
    }
  } else {
    lines.push({
      label: "clarify",
      value: `skip — ambiguity ${ambiguity.toFixed(2)} < 0.60`,
    });
  }

  const tokens = Math.max(120, Math.round(result.master_prompt.length / 3.5));
  lines.push({
    label: "synthesize",
    value: `${tokens.toLocaleString()} tokens · preamble cached`,
  });

  const routeReason =
    k >= 7 ? `complexity ${(k / 10).toFixed(2)} ≥ 0.70`
    : k >= 4 ? `complexity in [0.30, 0.70)`
    : `complexity ${(k / 10).toFixed(2)} < 0.30`;
  lines.push({
    label: "route",
    value: `${mLabel} · ${routeReason}`,
    highlight: mLabel,
    valueColor: mColor,
  });

  if (isOpus) {
    lines.push({ label: "thinking", value: "extended reasoning enabled", valueColor: "#f59e0b" });
  }

  if (result.n_agents > 1 && result.workflow_plan.length > 1) {
    lines.push({
      label: "subagents",
      value: `n_agents=${result.n_agents} → decomposing into ${result.workflow_plan.length} agents`,
    });
    for (const role of result.workflow_plan) {
      lines.push({ label: "↳", value: role, indent: true });
    }
  }

  lines.push({ label: "dispatch", value: "ready · dispatch via Claude Code" });
  return lines;
}

export const LangGraphTerminal = memo(function LangGraphTerminal({
  result,
}: {
  result: OptimizeResponse;
}) {
  const lines = useMemo(() => buildLines(result), [result]);

  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.09)",
        borderRadius: "10px",
        overflow: "hidden",
      }}
    >
      {/* Title bar */}
      <div
        style={{
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          padding: "0.5rem 1rem",
          display: "flex",
          alignItems: "center",
          gap: "0.4rem",
        }}
      >
        {["#ff5f57", "#febc2e", "#28c840"].map((c) => (
          <div
            key={c}
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: c,
              opacity: 0.7,
            }}
          />
        ))}
        <span
          style={{
            marginLeft: "0.5rem",
            ...MONO,
            fontSize: "0.58rem",
            letterSpacing: "0.12em",
            color: "rgba(255,255,255,0.2)",
            textTransform: "uppercase",
          }}
        >
          optivia · langgraph engine
        </span>
      </div>

      {/* Body */}
      <div
        style={{
          padding: "1rem 1.25rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.45rem",
        }}
      >
        {/* Prompt */}
        <div
          style={{
            display: "flex",
            gap: "0.75rem",
            alignItems: "flex-start",
            marginBottom: "0.3rem",
          }}
        >
          <span
            style={{
              ...MONO,
              fontSize: "0.65rem",
              color: "#00A0AE",
              flexShrink: 0,
              paddingTop: "2px",
              width: LABEL_W,
              textAlign: "right",
            }}
          >
            ›
          </span>
          <span
            style={{
              ...MONO,
              fontSize: "0.88rem",
              color: "rgba(255,255,255,0.88)",
              lineHeight: 1.5,
            }}
          >
            {result.master_prompt.replace(/\n+/g, " ").slice(0, 140)}
            {result.master_prompt.length > 140 ? "…" : ""}
          </span>
        </div>

        {/* Output lines */}
        {lines.map((line, i) => {
          const isIndent = line.indent || line.label === "↳" || line.label === "?";
          return (
            <div
              key={i}
              style={{
                display: "flex",
                gap: "0.75rem",
                alignItems: "flex-start",
                paddingLeft: isIndent ? "1.5rem" : 0,
                animation: "fadeSlideIn 0.2s ease",
              }}
            >
              <span
                style={{
                  ...MONO,
                  fontSize: "0.65rem",
                  flexShrink: 0,
                  paddingTop: "2px",
                  width: isIndent ? "auto" : LABEL_W,
                  textAlign: isIndent ? "left" : "right",
                  color:
                    line.label === "?"
                      ? "#f59e0b"
                      : line.label === "↳"
                      ? "#a78bfa"
                      : "rgba(255,255,255,0.28)",
                }}
              >
                {isIndent ? (line.label === "↳" ? "↳" : "?") : line.label}
              </span>
              <span
                style={{
                  ...MONO,
                  fontSize: "0.82rem",
                  lineHeight: 1.5,
                  color: line.valueColor ?? "rgba(255,255,255,0.52)",
                }}
              >
                {line.highlight ? (
                  <>
                    <span
                      style={{
                        color: line.valueColor ?? "#fff",
                        fontWeight: 600,
                      }}
                    >
                      {line.highlight}
                    </span>
                    {line.value.slice(line.highlight.length)}
                  </>
                ) : (
                  line.value
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});
