export interface OptimizeRequest {
  prompt: string;
  user_id?: string;
  workspace_id?: string;
  project_context?: { language?: string; framework?: string };
}

export interface OptimizeResponse {
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
  clarification_questions: { dimension: string; question: string }[];
}

export async function optimize(req: OptimizeRequest): Promise<OptimizeResponse> {
  const res = await fetch("/api/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Optimization failed");
  }
  return res.json();
}

export async function continueClarification(
  request_id: string,
  answers: string[],
  workspace_id = "",
  user_id = "",
): Promise<OptimizeResponse> {
  const res = await fetch("/api/optimize/continue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id, answers, workspace_id, user_id }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Continuation failed");
  }
  return res.json();
}

export async function submitFeedback(trace_id: string, thumbs: number): Promise<void> {
  await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trace_id, thumbs }),
  }).catch(() => {});
}
