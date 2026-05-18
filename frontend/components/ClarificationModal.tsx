"use client";

import { useState } from "react";
import { X, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";

interface Question {
  dimension: string;
  question: string;
}

interface Props {
  questions: Question[];
  onSubmit: (answers: string[]) => void;
  onDismiss: () => void;
}

const DIMENSION_COLOR: Record<string, string> = {
  scope: "text-blue-400",
  ambiguity: "text-yellow-400",
  risk: "text-red-400",
  dependency: "text-purple-400",
  context_load: "text-cyan-400",
};

export function ClarificationModal({ questions, onSubmit, onDismiss }: Props) {
  const [answers, setAnswers] = useState<string[]>(questions.map(() => ""));

  const allAnswered = answers.every((a) => a.trim().length > 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg bg-card border border-border rounded-xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-primary" />
            <span className="text-sm font-semibold">Clarification needed</span>
          </div>
          <button onClick={onDismiss} className="text-muted-foreground hover:text-foreground transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Questions */}
        <div className="p-4 space-y-4">
          {questions.map((q, i) => (
            <div key={i} className="space-y-2">
              <div className="flex items-center gap-2">
                <span className={cn("text-xs font-mono uppercase tracking-wider", DIMENSION_COLOR[q.dimension] ?? "text-muted-foreground")}>
                  [{q.dimension}]
                </span>
              </div>
              <p className="text-sm text-foreground">{q.question}</p>
              <textarea
                className="w-full bg-accent border border-border rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-primary text-foreground placeholder:text-muted-foreground"
                rows={2}
                placeholder="Your answer…"
                value={answers[i]}
                onChange={(e) => {
                  const next = [...answers];
                  next[i] = e.target.value;
                  setAnswers(next);
                }}
              />
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 p-4 border-t border-border">
          <button
            onClick={onDismiss}
            className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip
          </button>
          <button
            onClick={() => onSubmit(answers)}
            disabled={!allAnswered}
            className={cn(
              "px-4 py-2 text-sm rounded-lg font-medium transition-all",
              allAnswered
                ? "bg-primary text-primary-foreground hover:opacity-90"
                : "bg-accent text-muted-foreground cursor-not-allowed"
            )}
          >
            Continue →
          </button>
        </div>
      </div>
    </div>
  );
}
