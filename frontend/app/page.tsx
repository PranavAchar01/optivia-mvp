"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { Zap, Loader2, AlertCircle, History } from "lucide-react";
import { cn } from "@/lib/utils";
import { optimize, continueClarification, submitFeedback, type OptimizeResponse } from "@/lib/api";
import { ResultPanel } from "@/components/ResultPanel";
import { ClarificationModal } from "@/components/ClarificationModal";

type State = "idle" | "loading" | "clarifying" | "done" | "error";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [state, setState] = useState<State>("idle");
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [error, setError] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!prompt.trim() || state === "loading") return;

    setState("loading");
    setError("");
    setResult(null);

    try {
      const res = await optimize({ prompt });
      if (res.requires_clarification) {
        setResult(res);
        setState("clarifying");
      } else {
        setResult(res);
        setState("done");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setState("error");
    }
  };

  const handleClarificationSubmit = async (answers: string[]) => {
    if (!result) return;
    setState("loading");
    try {
      const res = await continueClarification(result.request_id, answers);
      setResult(res);
      setState("done");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setState("error");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  const handleFeedback = async (thumbs: number) => {
    if (!result?.trace_id) return;
    await submitFeedback(result.trace_id, thumbs);
  };

  return (
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-border px-6 py-3 flex items-center gap-3">
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-primary" />
          <span className="font-semibold tracking-tight">Optivia</span>
        </div>
        <span className="text-xs text-muted-foreground">pre-execution optimization layer</span>
        <Link
          href="/traces"
          className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <History className="w-3.5 h-3.5" />
          Traces
        </Link>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left: Input */}
        <div className="w-1/2 flex flex-col border-r border-border">
          <div className="flex-1 p-6 flex flex-col gap-4">
            <div>
              <h2 className="text-sm font-medium text-foreground mb-1">Your prompt</h2>
              <p className="text-xs text-muted-foreground">
                Describe the coding task. Optivia will classify, score, and synthesize an optimised
                master prompt before Claude Code runs.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col flex-1 gap-3">
              <textarea
                ref={textareaRef}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. Add a password reset flow to the existing FastAPI auth module"
                className="flex-1 min-h-[200px] bg-accent border border-border rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-muted-foreground leading-relaxed"
                disabled={state === "loading"}
              />

              <button
                type="submit"
                disabled={!prompt.trim() || state === "loading"}
                className={cn(
                  "flex items-center justify-center gap-2 px-6 py-3 rounded-xl text-sm font-medium transition-all",
                  prompt.trim() && state !== "loading"
                    ? "bg-primary text-primary-foreground hover:opacity-90 shadow-lg shadow-primary/20"
                    : "bg-accent text-muted-foreground cursor-not-allowed"
                )}
              >
                {state === "loading" ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Optimising…
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Optimise
                    <span className="text-xs opacity-60 ml-1">⌘↵</span>
                  </>
                )}
              </button>
            </form>

            {state === "error" && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-xs">
                <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Example prompts */}
          {state === "idle" && (
            <div className="px-6 pb-6">
              <p className="text-xs text-muted-foreground mb-2">Try an example:</p>
              <div className="flex flex-col gap-1.5">
                {[
                  "Add a password reset flow to the existing FastAPI auth module",
                  "Refactor the database service to use the repository pattern",
                  "Debug why the WebSocket connection drops after 30 seconds",
                  "Build a rate-limiting middleware with Redis and sliding window",
                ].map((ex) => (
                  <button
                    key={ex}
                    onClick={() => setPrompt(ex)}
                    className="text-left text-xs px-3 py-2 rounded-lg bg-accent border border-border hover:border-primary/50 hover:text-foreground text-muted-foreground transition-all truncate"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Result */}
        <div className="w-1/2 overflow-y-auto p-6">
          {(state === "done" || state === "clarifying") && result ? (
            <ResultPanel result={result} onFeedback={handleFeedback} />
          ) : state === "loading" ? (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-muted-foreground">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
              <div className="text-center space-y-1">
                <p className="text-sm">Running 17-agent pipeline…</p>
                <p className="text-xs">classify → score → synthesize → route</p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
              <Zap className="w-10 h-10 opacity-20" />
              <p className="text-sm">Results will appear here</p>
            </div>
          )}
        </div>
      </div>

      {/* Clarification modal */}
      {state === "clarifying" && result?.requires_clarification && (
        <ClarificationModal
          questions={result.clarification_questions}
          onSubmit={handleClarificationSubmit}
          onDismiss={() => setState("done")}
        />
      )}
    </main>
  );
}
