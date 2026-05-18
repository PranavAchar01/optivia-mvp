"use client";

import { useCallback, useEffect, useState } from "react";
import { Zap, ArrowLeft, RefreshCw } from "lucide-react";
import Link from "next/link";
import { cn, complexityColor, modelBadgeColor, taskTypeLabel } from "@/lib/utils";

interface Trace {
  id: string;
  created_at: string;
  raw_prompt: string;
  classification: { task_type?: string } | null;
  scores: { complexity?: number; specificity?: number } | null;
  routing_decision: { chosen_model?: string; n_agents?: number } | null;
  tokens_in: number | null;
  tokens_out: number | null;
  wall_ms: number | null;
  cost_usd: number | null;
}

const WORKSPACE_ID = "00000000-0000-0000-0000-000000000000";

export default function TracesPage() {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`/api/traces/${WORKSPACE_ID}?limit=50`);
      if (res.ok) {
        setTraces(await res.json());
      } else {
        setError("Failed to load traces");
      }
    } catch {
      setError("Could not reach server");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <main className="min-h-screen flex flex-col">
      <header className="border-b border-border px-6 py-3 flex items-center gap-3">
        <Link href="/" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-primary" />
          <span className="font-semibold tracking-tight">Optivia</span>
        </div>
        <span className="text-xs text-muted-foreground">trace history</span>
        <button
          onClick={load}
          className="ml-auto text-muted-foreground hover:text-foreground transition-colors"
        >
          <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
        </button>
      </header>

      <div className="flex-1 p-6">
        {loading ? (
          <div className="text-center text-muted-foreground py-20 text-sm">Loading traces…</div>
        ) : error ? (
          <div className="text-center text-destructive py-20 text-sm">{error}</div>
        ) : traces.length === 0 ? (
          <div className="text-center text-muted-foreground py-20 text-sm">
            No traces yet — run an optimisation to see results here.
          </div>
        ) : (
          <div className="space-y-2">
            {traces.map((t) => {
              const kappa = t.scores?.complexity;
              const model = t.routing_decision?.chosen_model ?? "";
              const taskType = t.classification?.task_type ?? "unknown";

              return (
                <div
                  key={t.id}
                  className="flex items-start gap-4 px-4 py-3 rounded-xl bg-card border border-border hover:border-primary/30 transition-all"
                >
                  {/* Complexity badge */}
                  <div className={cn("text-lg font-bold font-mono tabular-nums w-8 shrink-0 mt-0.5", complexityColor(kappa ?? 5))}>
                    {kappa ?? "?"}
                  </div>

                  {/* Prompt + metadata */}
                  <div className="flex-1 min-w-0 space-y-1">
                    <p className="text-sm text-foreground truncate">{t.raw_prompt}</p>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs text-muted-foreground bg-accent px-2 py-0.5 rounded font-mono">
                        {taskTypeLabel(taskType)}
                      </span>
                      {model && (
                        <span className={cn("text-xs px-2 py-0.5 rounded border font-mono", modelBadgeColor(model))}>
                          {model.replace("claude-", "")}
                        </span>
                      )}
                      {t.wall_ms && (
                        <span className="text-xs text-muted-foreground">{(t.wall_ms / 1000).toFixed(1)}s</span>
                      )}
                      {t.cost_usd != null && t.cost_usd > 0 && (
                        <span className="text-xs text-muted-foreground">${t.cost_usd.toFixed(4)}</span>
                      )}
                    </div>
                  </div>

                  {/* Timestamp */}
                  <div className="text-xs text-muted-foreground shrink-0">
                    {new Date(t.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
