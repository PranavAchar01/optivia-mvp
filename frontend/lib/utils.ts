import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function complexityColor(k: number): string {
  if (k <= 3) return "text-emerald-400";
  if (k <= 6) return "text-yellow-400";
  if (k <= 8) return "text-orange-400";
  return "text-red-400";
}

export function modelBadgeColor(model: string): string {
  if (model.includes("haiku")) return "bg-emerald-500/15 text-emerald-300 border-emerald-500/30";
  if (model.includes("sonnet")) return "bg-blue-500/15 text-blue-300 border-blue-500/30";
  return "bg-purple-500/15 text-purple-300 border-purple-500/30";
}

export function taskTypeLabel(t: string): string {
  return { new_code: "New Code", debug: "Debug", refactor: "Refactor", review: "Review", explain: "Explain", long: "Long Task", trivial: "Trivial", meta: "Meta" }[t] ?? t;
}
