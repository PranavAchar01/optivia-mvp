"use client";

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  BackgroundVariant,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

interface Props {
  steps: string[];
  taskType: string;
}

const NODE_COLORS: Record<string, string> = {
  new_code: "#3b82f6",
  debug: "#f59e0b",
  refactor: "#8b5cf6",
  review: "#10b981",
  explain: "#06b6d4",
  long: "#f97316",
  trivial: "#6b7280",
  meta: "#ec4899",
};

export function WorkflowGraph({ steps, taskType }: Props) {
  const accent = NODE_COLORS[taskType] ?? "#3b82f6";

  const nodes: Node[] = useMemo(
    () =>
      steps.map((step, i) => ({
        id: `step-${i}`,
        position: { x: 220 * i, y: 100 },
        data: { label: step },
        style: {
          background: i === 0 ? accent : "hsl(224 71% 8%)",
          border: `1px solid ${i === 0 ? accent : "hsl(216 34% 20%)"}`,
          borderRadius: 8,
          color: "hsl(213 31% 91%)",
          fontSize: 12,
          fontFamily: "ui-monospace, monospace",
          padding: "10px 14px",
          maxWidth: 180,
          whiteSpace: "pre-wrap" as const,
          boxShadow: i === 0 ? `0 0 12px ${accent}40` : "none",
        },
      })),
    [steps, accent]
  );

  const edges: Edge[] = useMemo(
    () =>
      steps.slice(0, -1).map((_, i) => ({
        id: `e-${i}`,
        source: `step-${i}`,
        target: `step-${i + 1}`,
        animated: true,
        markerEnd: { type: MarkerType.ArrowClosed, color: accent },
        style: { stroke: accent, strokeWidth: 1.5 },
      })),
    [steps, accent]
  );

  if (steps.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        No workflow plan yet
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.3 }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="hsl(216 34% 20%)" />
      <Controls showInteractive={false} />
      {steps.length >= 4 && (
        <MiniMap
          nodeColor={() => accent}
          nodeStrokeColor="transparent"
          maskColor="rgba(3,7,18,0.75)"
          style={{
            bottom: 12,
            right: 12,
            background: "hsl(224 71% 6%)",
            border: "1px solid hsl(216 34% 17%)",
            borderRadius: 6,
          }}
        />
      )}
    </ReactFlow>
  );
}
