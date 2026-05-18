import { useEffect, useMemo, useRef, useState } from "react";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";
import { invoke } from "@tauri-apps/api/core";
import { Shell } from "./Shell";

type ClarificationQ = { dimension: string; question: string };
type DispatchState = "idle" | "queued" | "error";

type FleetNode = {
  Name: string;
  "Model Tier": string;
  "System Prompt": string;
  "Estimated Tokens": number;
  "Estimated Duration"?: number;
  "On Critical Path"?: boolean;
  Bottleneck?: boolean;
  ES?: number;
  EF?: number;
  Slack?: number;
};

type FleetEdge = { From: string; To: string };

type OptimizeResponse = {
  // v15 fleet DAG fields (required by backend)
  Task_Type: string;
  Complexity_Score: number;
  Environment_Target: string;
  Nodes: FleetNode[];
  Edges: FleetEdge[];
  Critical_Path: string;
  // Legacy fields
  request_id: string;
  trace_id: string;
  master_prompt: string;
  model: string;
  n_agents: number;
  slash_commands: string[];
  workflow_plan: string[];
  complexity: number;
  specificity: number;
  task_type: string;
  requires_clarification: boolean;
  clarification_questions: ClarificationQ[];
};

type Line = {
  label: string;
  value: string;
  valueColor?: string;
  highlight?: string;
  indent?: boolean;
  pulse?: boolean;
};

const LABEL_W = "92px";

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

function buildLines(r: OptimizeResponse, dispatch: DispatchState): Line[] {
  const lines: Line[] = [];
  const k = r.complexity;
  const sigma = r.specificity;
  const ambiguity = Math.max(0, Math.min(1, 1 - sigma));
  const intent = (r.task_type || r.Task_Type || "unknown").toUpperCase();
  const mLabel = modelLabel(r.model);
  const mColor = modelColor(r.model);
  const isOpus = r.model.toLowerCase().includes("opus");
  const nodes = r.Nodes ?? [];
  const criticalPath = r.Critical_Path ?? "";
  const cpSet = new Set(criticalPath.split(" -> ").filter(Boolean));

  lines.push({ label: "cache_lookup", value: "miss" });
  lines.push({
    label: "fast_intent",
    value: `${intent} · conf ${(0.6 + sigma * 0.4).toFixed(2)}`,
    highlight: intent,
  });
  lines.push({
    label: "classify",
    value: `κ=${k} · specificity ${sigma.toFixed(2)} · ambiguity ${ambiguity.toFixed(2)}`,
  });

  if (r.requires_clarification && r.clarification_questions.length > 0) {
    lines.push({
      label: "clarify",
      value: `ambiguity ${ambiguity.toFixed(2)} ≥ 0.60 — generating questions`,
    });
    for (const q of r.clarification_questions) {
      lines.push({ label: "?", value: q.question, indent: true });
    }
  } else {
    lines.push({
      label: "clarify",
      value: `skip — ambiguity ${ambiguity.toFixed(2)} < 0.60`,
    });
  }

  const tokens = Math.max(120, Math.round(r.master_prompt.length / 3.5));
  lines.push({
    label: "synthesize",
    value: `${tokens.toLocaleString()} tokens · preamble cached`,
  });

  const routeReason =
    k >= 7
      ? `κ=${k} ≥ 7 — strong model`
      : k >= 4
      ? `κ=${k} — balanced`
      : `κ=${k} — fast`;
  lines.push({
    label: "route",
    value: `${mLabel} · ${routeReason}`,
    highlight: mLabel,
    valueColor: mColor,
  });

  if (isOpus) {
    lines.push({ label: "thinking", value: "extended reasoning enabled", valueColor: "#f59e0b" });
  }

  // Fleet + CPM section
  if (nodes.length > 1) {
    lines.push({
      label: "fleet",
      value: `${nodes.length} agents · critical path: ${criticalPath || "—"}`,
      valueColor: "#a78bfa",
    });
    for (const node of nodes) {
      const onCP = node["On Critical Path"] || cpSet.has(node.Name);
      const isBottleneck = node.Bottleneck;
      const cpTag = onCP ? " ⚡" : "";
      const bnTag = isBottleneck ? " 🔴" : "";
      const slack = node.Slack !== undefined ? ` slack=${node.Slack}s` : "";
      lines.push({
        label: "↳",
        value: `${node.Name}${cpTag}${bnTag}${slack}`,
        indent: true,
        valueColor: onCP ? "#a78bfa" : undefined,
      });
    }
  } else if (nodes.length === 1) {
    lines.push({
      label: "fleet",
      value: `single executor · ${nodes[0].Name}`,
    });
  } else if (r.n_agents > 1 && r.workflow_plan.length > 1) {
    // Fallback for older response shape
    lines.push({
      label: "fleet",
      value: `${r.n_agents} agents`,
    });
    for (const role of r.workflow_plan) {
      lines.push({ label: "↳", value: role, indent: true });
    }
  }

  if (dispatch === "queued") {
    lines.push({
      label: "dispatch",
      value: "queued · run `claude` in your terminal",
      valueColor: "#4ade80",
      highlight: "queued",
    });
  } else if (dispatch === "error") {
    lines.push({
      label: "dispatch",
      value: "queue write failed — check ~/.optivia/queue permissions",
      valueColor: "#f87171",
    });
  } else {
    lines.push({ label: "dispatch", value: "ready · type `claude` in the shell below" });
  }
  return lines;
}

function Terminal({
  prompt,
  busy,
  result,
  dispatch,
}: {
  prompt: string;
  busy: boolean;
  result: OptimizeResponse | null;
  dispatch: DispatchState;
}) {
  const lines = useMemo(() => (result ? buildLines(result, dispatch) : []), [result, dispatch]);
  const promptText = (result?.master_prompt?.split("\n")[0] ?? prompt).slice(0, 240);

  return (
    <div
      style={{
        flex: "1 1 0",
        minHeight: 0,
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.09)",
        borderRadius: 10,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
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
          flexShrink: 0,
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
          overflowY: "auto",
        }}
      >
        {prompt && (
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
                fontSize: "0.65rem",
                color: "#00A0AE",
                flexShrink: 0,
                paddingTop: 2,
                width: LABEL_W,
                textAlign: "right",
              }}
            >
              ›
            </span>
            <span
              style={{
                fontSize: "0.88rem",
                color: "rgba(255,255,255,0.88)",
                lineHeight: 1.5,
              }}
            >
              {promptText}
              {busy && (
                <span
                  style={{
                    display: "inline-block",
                    width: 2,
                    height: "1em",
                    background: "rgba(255,255,255,0.6)",
                    marginLeft: 2,
                    verticalAlign: "text-bottom",
                    animation: "blink 0.75s step-end infinite",
                  }}
                />
              )}
            </span>
          </div>
        )}

        {busy && !result && (
          <div
            style={{
              display: "flex",
              gap: "0.75rem",
              alignItems: "flex-start",
              animation: "fadeSlideIn 0.2s ease",
            }}
          >
            <span
              style={{
                fontSize: "0.65rem",
                color: "rgba(255,255,255,0.28)",
                flexShrink: 0,
                paddingTop: 2,
                width: LABEL_W,
                textAlign: "right",
              }}
            >
              pipeline
            </span>
            <span
              style={{
                fontSize: "0.82rem",
                color: "rgba(255,255,255,0.52)",
                animation: "pulse 1s ease-in-out infinite",
              }}
            >
              classify · score · synthesize · route ···
            </span>
          </div>
        )}

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
                  fontSize: "0.65rem",
                  flexShrink: 0,
                  paddingTop: 2,
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
                  fontSize: "0.82rem",
                  lineHeight: 1.5,
                  color: line.valueColor ?? "rgba(255,255,255,0.52)",
                }}
              >
                {line.highlight ? (
                  <>
                    <span style={{ color: line.valueColor ?? "#fff", fontWeight: 600 }}>
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
}

function SettingsSheet({
  open,
  onClose,
  apiBase,
  setApiBase,
}: {
  open: boolean;
  onClose: () => void;
  apiBase: string;
  setApiBase: (v: string) => void;
}) {
  const [shimInstalled, setShimInstalled] = useState<boolean | null>(null);
  const [shimMsg, setShimMsg] = useState<string>("");
  const [shimBusy, setShimBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    void invoke<boolean>("shim_status")
      .then(setShimInstalled)
      .catch(() => setShimInstalled(false));
  }, [open]);

  async function toggleShim() {
    setShimBusy(true);
    setShimMsg("");
    try {
      const cmd = shimInstalled ? "uninstall_shim" : "install_shim";
      const res = await invoke<string[]>(cmd);
      setShimMsg(res.join("  ·  "));
      const status = await invoke<boolean>("shim_status");
      setShimInstalled(status);
    } catch (e) {
      setShimMsg(typeof e === "string" ? e : "shim operation failed");
    } finally {
      setShimBusy(false);
    }
  }

  if (!open) return null;
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 50,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#0a0a0a",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 10,
          width: "min(620px, 92vw)",
          padding: "1.25rem 1.5rem",
          display: "flex",
          flexDirection: "column",
          gap: "1.4rem",
        }}
      >
        {/* Backend section */}
        <section>
          <div
            style={{
              fontSize: "0.6rem",
              textTransform: "uppercase",
              letterSpacing: "0.14em",
              color: "rgba(255,255,255,0.35)",
              marginBottom: "0.75rem",
            }}
          >
            backend
          </div>
          <label
            style={{
              display: "block",
              fontSize: "0.7rem",
              color: "rgba(255,255,255,0.6)",
              marginBottom: "0.4rem",
            }}
          >
            Optivia API base URL
          </label>
          <input
            type="url"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="https://your-optivia-host.example.com"
            style={{
              width: "100%",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 6,
              padding: "0.55rem 0.7rem",
              fontSize: "0.8rem",
              color: "#fff",
            }}
          />
          <p
            style={{
              fontSize: "0.65rem",
              color: "rgba(255,255,255,0.35)",
              lineHeight: 1.55,
              marginTop: "0.6rem",
            }}
          >
            The desktop app POSTs prompts to <code>{`{base}/optimize`}</code>. Your backend
            handles the API key — nothing is stored locally except this URL.
          </p>
        </section>

        {/* Dispatch shim section */}
        <section>
          <div
            style={{
              fontSize: "0.6rem",
              textTransform: "uppercase",
              letterSpacing: "0.14em",
              color: "rgba(255,255,255,0.35)",
              marginBottom: "0.75rem",
            }}
          >
            claude code dispatch shim
          </div>
          <p
            style={{
              fontSize: "0.7rem",
              color: "rgba(255,255,255,0.6)",
              lineHeight: 1.6,
              margin: 0,
            }}
          >
            Installs a tiny shell function in your <code>~/.zshrc</code> and{" "}
            <code>~/.bashrc</code>. When you run <code>claude</code> with no arguments, it
            picks up the most recent optimised prompt from <code>~/.optivia/queue</code> and
            starts Claude Code with it as the first message. Plain <code>claude &lt;args&gt;</code>{" "}
            calls are unaffected.
          </p>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.7rem",
              marginTop: "0.8rem",
            }}
          >
            <button
              onClick={toggleShim}
              disabled={shimBusy || shimInstalled === null}
              style={{
                background:
                  shimInstalled === true
                    ? "rgba(248,113,113,0.1)"
                    : "rgba(74,222,128,0.1)",
                border: `1px solid ${
                  shimInstalled === true
                    ? "rgba(248,113,113,0.4)"
                    : "rgba(74,222,128,0.4)"
                }`,
                color: shimInstalled === true ? "#f87171" : "#4ade80",
                borderRadius: 6,
                padding: "0.45rem 0.9rem",
                fontSize: "0.7rem",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                cursor: shimBusy ? "default" : "pointer",
              }}
            >
              {shimBusy
                ? "..."
                : shimInstalled === null
                ? "checking"
                : shimInstalled
                ? "uninstall"
                : "install shim"}
            </button>
            <span
              style={{
                fontSize: "0.65rem",
                color:
                  shimInstalled === true
                    ? "rgba(74,222,128,0.7)"
                    : "rgba(255,255,255,0.4)",
              }}
            >
              {shimInstalled === true
                ? "active — open a new terminal session to start using it"
                : shimInstalled === false
                ? "not installed"
                : ""}
            </span>
          </div>
          {shimMsg && (
            <div
              style={{
                marginTop: "0.6rem",
                fontSize: "0.6rem",
                color: "rgba(255,255,255,0.4)",
                fontFamily: "inherit",
              }}
            >
              {shimMsg}
            </div>
          )}
        </section>

        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
          }}
        >
          <button
            onClick={onClose}
            style={{
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.14)",
              color: "#fff",
              borderRadius: 6,
              padding: "0.45rem 1rem",
              fontSize: "0.75rem",
            }}
          >
            done
          </button>
        </div>
      </div>
    </div>
  );
}

const STORAGE_KEY = "optivia.apiBase";

export default function App() {
  const [apiBase, setApiBaseState] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem(STORAGE_KEY) || "http://localhost:8000";
  });
  const setApiBase = (v: string) => {
    setApiBaseState(v);
    try {
      localStorage.setItem(STORAGE_KEY, v);
    } catch { /* noop */ }
  };

  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [dispatch, setDispatch] = useState<DispatchState>("idle");
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    taRef.current?.focus();
  }, []);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    setDispatch("idle");
    setSubmittedPrompt(prompt.trim());
    try {
      const res = await tauriFetch(`${apiBase.replace(/\/$/, "")}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim() }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
      }
      const data: OptimizeResponse = await res.json();
      setResult(data);
      // Auto-dispatch: write the master prompt to ~/.optivia/current.json so the
      // embedded shell's claude() shim picks it up the next time `claude` is typed.
      try {
        await invoke("dispatch_to_claude", {
          payload: {
            prompt: data.master_prompt,
            model: data.model,
            n_agents: data.n_agents,
            task_type: data.task_type,
            trace_id: data.trace_id,
            proxy_base: apiBase.replace(/\/$/, ""),
          },
        });
        setDispatch("queued");
      } catch {
        setDispatch("error");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey || !e.shiftKey)) {
      e.preventDefault();
      void submit();
    }
  }

  function reset() {
    setResult(null);
    setError("");
    setPrompt("");
    setSubmittedPrompt("");
    setDispatch("idle");
    taRef.current?.focus();
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        padding: "1.25rem",
        gap: "0.9rem",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
        }}
      >
        <span
          style={{
            fontSize: "0.62rem",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "rgba(255,255,255,0.7)",
          }}
        >
          optivia
        </span>
        <span
          style={{
            fontSize: "0.55rem",
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            color: "rgba(255,255,255,0.25)",
          }}
        >
          pre-execution layer
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: "0.6rem" }}>
          {(result || error) && (
            <button
              onClick={reset}
              style={{
                background: "transparent",
                border: "1px solid rgba(255,255,255,0.14)",
                color: "rgba(255,255,255,0.7)",
                borderRadius: 6,
                padding: "0.3rem 0.7rem",
                fontSize: "0.62rem",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
              }}
            >
              new prompt
            </button>
          )}
          <button
            onClick={() => setSettingsOpen(true)}
            style={{
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.14)",
              color: "rgba(255,255,255,0.7)",
              borderRadius: 6,
              padding: "0.3rem 0.7rem",
              fontSize: "0.62rem",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
            }}
          >
            settings
          </button>
        </div>
      </div>

      {/* Two-pane main: LangGraph engine on top, real shell below */}
      <div
        style={{
          flex: "1 1 0",
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          gap: "0.9rem",
        }}
      >
        <div style={{ flex: "1 1 60%", minHeight: 0, display: "flex" }}>
          <Terminal
            prompt={submittedPrompt || prompt}
            busy={busy}
            result={result}
            dispatch={dispatch}
          />
        </div>
        <div style={{ flex: "1 1 40%", minHeight: 200, display: "flex" }}>
          <div style={{ flex: "1 1 0", display: "flex", minHeight: 0 }}>
            <Shell />
          </div>
        </div>
      </div>

      {/* Error row */}
      {error && (
        <div
          style={{
            border: "1px solid rgba(239,68,68,0.4)",
            background: "rgba(239,68,68,0.08)",
            color: "#fca5a5",
            borderRadius: 6,
            padding: "0.55rem 0.8rem",
            fontSize: "0.72rem",
          }}
        >
          {error}
        </div>
      )}

      {/* Prompt input */}
      <form onSubmit={submit} style={{ display: "flex", gap: "0.6rem", alignItems: "stretch" }}>
        <textarea
          ref={taRef}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="describe a coding task — press enter to optimise"
          rows={1}
          style={{
            flex: 1,
            resize: "none",
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            padding: "0.7rem 0.9rem",
            fontSize: "0.85rem",
            lineHeight: 1.45,
            color: "#fff",
            minHeight: "2.6rem",
            maxHeight: "9rem",
          }}
        />
        <button
          type="submit"
          disabled={!prompt.trim() || busy}
          style={{
            background:
              !prompt.trim() || busy
                ? "rgba(255,255,255,0.04)"
                : "rgba(255,255,255,0.08)",
            border: "1px solid rgba(255,255,255,0.16)",
            color: !prompt.trim() || busy ? "rgba(255,255,255,0.3)" : "#fff",
            borderRadius: 8,
            padding: "0 1rem",
            fontSize: "0.7rem",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            cursor: !prompt.trim() || busy ? "default" : "pointer",
            minWidth: 110,
          }}
        >
          {busy ? "running" : "enter ↵"}
        </button>
      </form>

      <SettingsSheet
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        apiBase={apiBase}
        setApiBase={setApiBase}
      />
    </div>
  );
}
