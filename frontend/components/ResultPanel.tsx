"use client";

import { useState } from "react";
import {
  ThumbsUp,
  ThumbsDown,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Network,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { LangGraphTerminal } from "./LangGraphTerminal";
import { WorkflowGraph } from "./WorkflowGraph";
import type { OptimizeResponse } from "@/lib/api";

interface Props {
  result: OptimizeResponse;
  onFeedback: (thumbs: number) => void;
}

export function ResultPanel({ result, onFeedback }: Props) {
  const [copied, setCopied] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState<number | null>(null);

  const copy = async () => {
    await navigator.clipboard.writeText(result.master_prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const sendFeedback = (thumbs: number) => {
    setFeedbackSent(thumbs);
    onFeedback(thumbs);
  };

  return (
    <div className="space-y-3">
      {/* Primary: terminal-style LangGraph engine view */}
      <LangGraphTerminal result={result} />

      {/* Toggle: workflow graph (n8n-style React Flow) */}
      {result.workflow_plan.length > 1 && <button
        onClick={() => setShowGraph((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-white/10 bg-white/[0.03] text-[11px] uppercase tracking-[0.12em] text-white/40 hover:text-white/70 hover:border-white/20 transition-colors"
        style={{ fontFamily: "var(--font-geist-mono)" }}
      >
        <span className="flex items-center gap-2">
          <Network className="w-3 h-3" />
          {showGraph ? "Hide" : "Show"} workflow graph
        </span>
        {showGraph ? (
          <ChevronUp className="w-3 h-3" />
        ) : (
          <ChevronDown className="w-3 h-3" />
        )}
      </button>}
      {showGraph && result.workflow_plan.length > 1 && (
        <div
          className="rounded-lg border border-white/10 bg-white/[0.03] overflow-hidden"
          style={{ height: 280 }}
        >
          <WorkflowGraph
            steps={result.workflow_plan}
            taskType={result.task_type}
          />
        </div>
      )}

      {/* Master prompt collapsible */}
      <div className="rounded-lg border border-white/10 bg-white/[0.03] overflow-hidden">
        <button
          className="w-full flex items-center justify-between px-3 py-2 text-[11px] uppercase tracking-[0.12em] text-white/40 hover:text-white/70 transition-colors"
          style={{ fontFamily: "var(--font-geist-mono)" }}
          onClick={() => setShowPrompt((v) => !v)}
        >
          <span>Master prompt</span>
          {showPrompt ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
        </button>
        {showPrompt && (
          <div className="px-3 pb-3 space-y-2">
            <pre
              className="text-xs text-white/85 bg-black/40 rounded-md p-3 overflow-auto whitespace-pre-wrap max-h-64 leading-relaxed"
              style={{ fontFamily: "var(--font-geist-mono)" }}
            >
              {result.master_prompt}
            </pre>
            <button
              onClick={copy}
              className="flex items-center gap-1.5 text-xs text-white/40 hover:text-white/80 transition-colors"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              {copied ? "Copied" : "Copy prompt"}
            </button>
          </div>
        )}
      </div>

      {/* Feedback */}
      <div className="flex items-center justify-between pt-1">
        <span
          className="text-[11px] uppercase tracking-[0.12em] text-white/30"
          style={{ fontFamily: "var(--font-geist-mono)" }}
        >
          Was this helpful?
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => sendFeedback(1)}
            className={cn(
              "p-1.5 rounded-md border transition-all",
              feedbackSent === 1
                ? "border-emerald-500 text-emerald-400 bg-emerald-500/10"
                : "border-white/10 text-white/40 hover:text-emerald-400 hover:border-emerald-500/50"
            )}
          >
            <ThumbsUp className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => sendFeedback(-1)}
            className={cn(
              "p-1.5 rounded-md border transition-all",
              feedbackSent === -1
                ? "border-red-500 text-red-400 bg-red-500/10"
                : "border-white/10 text-white/40 hover:text-red-400 hover:border-red-500/50"
            )}
          >
            <ThumbsDown className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
