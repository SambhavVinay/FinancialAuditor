"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
  type SyntheticEvent,
} from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  NodeToolbar,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

/** Matches backend SSE step ids and order */
const CREW_BLUEPRINT = [
  {
    stepId: "crew_vision",
    agentRole: "Forensic Vision Inspector",
    phase: "crew" as const,
  },
  {
    stepId: "crew_compliance",
    agentRole: "Policy Compliance Auditor",
    phase: "crew" as const,
  },
  {
    stepId: "crew_actuary",
    agentRole: "Payout Actuary",
    phase: "crew" as const,
  },
  {
    stepId: "crew_communicator",
    agentRole: "Customer Relations Lead",
    phase: "crew" as const,
  },
];

const PRECALL_BLUEPRINT = {
  stepId: "vision_precall",
  agentRole: "Direct multimodal vision (Gemma 4 31B)",
  phase: "precall" as const,
};

export type FlowStepStatus = "pending" | "running" | "done";

export interface PipelineStepState {
  stepId: string;
  agentRole: string;
  phase: "precall" | "crew";
  status: FlowStepStatus;
  progressMessage?: string;
  taskDescription?: string;
  finalOutput?: string;
  rawLog?: string;
  expanded: boolean;
}

export function createInitialPipeline(includePrecall: boolean): PipelineStepState[] {
  const base = (meta: {
    stepId: string;
    agentRole: string;
    phase: "precall" | "crew";
  }): PipelineStepState => ({
    ...meta,
    status: "pending",
    expanded: false,
  });

  if (includePrecall) {
    return [base(PRECALL_BLUEPRINT), ...CREW_BLUEPRINT.map(base)];
  }
  return CREW_BLUEPRINT.map(base);
}

/* ─── n8n-ish tokens ─────────────────────────────────── */
const canvasBg = "#0b0e14";
const nodeBorder = "#3f3f4d";
const nodeBg = "#1a1d26";
const nodeBgRunning = "#1e1a2e";
const nodeBgFocused = "#1f2430";
const accent = "#ff6d5a";
const edgeIdle = "#4b5563";
const edgeActive = "#a78bfa";

const INNER_W = 400;
const NODE_W = 220;
const NODE_H_MIN = 108;
const ZIG_X = 22;
const VERTICAL_GAP = 128;
const TRIGGER_SIZE = 48;
const TOP_PAD = 12;
const GAP_AFTER_TRIGGER = 20;

const centerX = (INNER_W - NODE_W) / 2;

function agentX(index: number): number {
  return centerX + (index % 2 === 0 ? -ZIG_X : ZIG_X);
}

function flowHeightForStepCount(n: number, expanded: boolean): number {
  if (n <= 0) return expanded ? 360 : 280;
  const body =
    TOP_PAD +
    TRIGGER_SIZE +
    GAP_AFTER_TRIGGER +
    n * VERTICAL_GAP +
    NODE_H_MIN +
    40;
  const cap = expanded ? 1180 : 920;
  const floor = expanded ? 520 : 300;
  return Math.min(cap, Math.max(floor, body));
}

/** PowerShell-style block matching backend logs; uses server rawLog when present. */
function formatPsLog(step: PipelineStepState): string {
  if (step.status === "done" && step.rawLog?.trim()) {
    return step.rawLog.trim();
  }
  const agent = step.agentRole;
  const task =
    step.taskDescription?.trim() ||
    (step.status === "pending"
      ? "This step has not started yet. It will run when earlier nodes complete."
      : step.status === "running"
        ? "(Agent is executing — full task text will appear here when this step completes.)"
        : "(No task text.)");
  const answer =
    step.status === "done"
      ? step.finalOutput?.trim() ||
        "(No final answer captured — check server logs.)"
      : step.status === "running"
        ? step.progressMessage?.trim() ||
          "(Waiting for model output…)"
        : "(—)";
  return `# Agent: ${agent}\n## Task:\n${task}\n\n## Final Answer:\n${answer}`;
}

function stopFlowEvent(e: SyntheticEvent) {
  e.stopPropagation();
}

type AgentNodeData = {
  step: PipelineStepState;
  isFocused: boolean;
  onFocusHeader: () => void;
  onShowMore: () => void;
  detailsOpen: boolean;
  detailsText: string;
  onCloseDetails: () => void;
  expandedLayout: boolean;
};

function TriggerNode() {
  return (
    <div
      className="nodrag nopan"
      style={{
        position: "relative",
        width: TRIGGER_SIZE,
        height: TRIGGER_SIZE,
        borderRadius: "50%",
        background: "linear-gradient(145deg, #2a2f3d 0%, #1a1e28 100%)",
        border: `2px solid ${nodeBorder}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "1.25rem",
        boxShadow: "0 2px 12px rgba(0,0,0,0.45)",
      }}
      title="Claim submitted"
    >
      <Handle
        id="source-bottom"
        type="source"
        position={Position.Bottom}
        style={{
          width: 10,
          height: 10,
          background: "#888",
          border: "2px solid #1a1d26",
        }}
      />
      ⚡
    </div>
  );
}

function AgentNode({ data }: NodeProps<Node<AgentNodeData>>) {
  const {
    step,
    isFocused,
    onFocusHeader,
    onShowMore,
    detailsOpen,
    detailsText,
    onCloseDetails,
    expandedLayout,
  } = data;
  const isPending = step.status === "pending";
  const isRunning = step.status === "running";
  const isDone = step.status === "done";

  const borderColor = isRunning
    ? accent
    : isFocused
      ? "#6d5ccd"
      : isDone
        ? "#3d5248"
        : nodeBorder;
  const bg = isRunning ? nodeBgRunning : isFocused ? nodeBgFocused : nodeBg;

  const subtitle =
    step.phase === "precall" ? "PRE-CREW API" : "CREWAI AGENT";

  const subDetail =
    step.progressMessage && isRunning ? ` · ${step.progressMessage}` : "";

  return (
    <>
      <NodeToolbar
        isVisible={detailsOpen}
        position={Position.Right}
        offset={14}
        align="start"
        className="nodrag nopan"
        style={{
          pointerEvents: "all",
        }}
      >
        <div
          style={{
            width: expandedLayout ? "min(520px, 48vw)" : "min(420px, 72vw)",
            maxHeight: expandedLayout ? 440 : 340,
            display: "flex",
            flexDirection: "column",
            borderRadius: 10,
            border: "2px solid #3d5248",
            background: "rgba(26,29,38,0.98)",
            boxShadow:
              "0 12px 40px rgba(0,0,0,0.55), 0 0 0 1px rgba(167,139,250,0.12)",
            overflow: "hidden",
            fontFamily:
              "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 10,
              padding: "8px 10px 8px 12px",
              borderBottom: "1px solid rgba(255,255,255,0.08)",
              background: "rgba(0,0,0,0.35)",
            }}
          >
            <div style={{ minWidth: 0, flex: 1 }}>
              <div
                style={{
                  fontSize: "0.62rem",
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  color: "rgba(167,139,250,0.95)",
                  textTransform: "uppercase",
                }}
              >
                {step.agentRole}
              </div>
              <div
                style={{
                  fontSize: "0.58rem",
                  color: "rgba(148,163,184,0.65)",
                  marginTop: 4,
                }}
              >
                {step.status === "done"
                  ? "Execution log"
                  : step.status === "running"
                    ? "In progress"
                    : "Pending"}
              </div>
            </div>
            <button
              type="button"
              onMouseDown={stopFlowEvent}
              onPointerDown={stopFlowEvent}
              onClick={(e) => {
                e.stopPropagation();
                onCloseDetails();
              }}
              style={{
                flexShrink: 0,
                fontSize: "0.65rem",
                fontWeight: 600,
                padding: "4px 10px",
                borderRadius: 6,
                border: `1px solid ${nodeBorder}`,
                background: "rgba(255,255,255,0.06)",
                color: "rgba(203,213,225,0.9)",
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Close
            </button>
          </div>
          <pre
            style={{
              margin: 0,
              padding: "10px 12px 12px",
              fontSize: "0.72rem",
              lineHeight: 1.55,
              color: "rgba(226,232,240,0.92)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontFamily: "ui-monospace, monospace",
              overflow: "auto",
              maxHeight: expandedLayout ? 360 : 280,
            }}
          >
            {detailsText}
          </pre>
        </div>
      </NodeToolbar>
      <Handle
        id="target-top"
        type="target"
        position={Position.Top}
        style={{
          width: 10,
          height: 10,
          background: "#888",
          border: "2px solid #1a1d26",
        }}
      />
      <Handle
        id="target-left"
        type="target"
        position={Position.Left}
        style={{
          width: 10,
          height: 10,
          background: "#888",
          border: "2px solid #1a1d26",
        }}
      />
      <Handle
        id="target-right"
        type="target"
        position={Position.Right}
        style={{
          width: 10,
          height: 10,
          background: "#888",
          border: "2px solid #1a1d26",
        }}
      />
      <div
        className="nodrag nopan"
        style={{
          width: NODE_W,
          minHeight: NODE_H_MIN,
          borderRadius: 10,
          border: `2px solid ${borderColor}`,
          background: bg,
          boxShadow: isRunning
            ? `0 0 0 1px rgba(255,109,90,0.15), 0 8px 24px rgba(0,0,0,0.35)`
            : "0 4px 16px rgba(0,0,0,0.35)",
          fontFamily:
            "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif",
          transition: "border-color 0.2s, box-shadow 0.2s, background 0.2s",
          overflow: "hidden",
          pointerEvents: "auto",
        }}
      >
        <button
          type="button"
          onMouseDown={stopFlowEvent}
          onPointerDown={stopFlowEvent}
          onClick={(e) => {
            e.stopPropagation();
            onFocusHeader();
          }}
          style={{
            width: "100%",
            padding: "10px 12px 8px 14px",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            textAlign: "left",
            color: "#e2e8f0",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 10,
            }}
          >
            <span
              style={{
                fontSize: "1rem",
                lineHeight: 1.2,
                marginTop: 2,
                opacity: isPending ? 0.35 : 1,
              }}
            >
              {isPending ? "○" : isRunning ? "⏳" : "✓"}
            </span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "#f1f5f9",
                  lineHeight: 1.35,
                }}
              >
                {step.agentRole}
              </div>
              <div
                style={{
                  fontSize: "0.58rem",
                  fontWeight: 500,
                  color: "rgba(148,163,184,0.75)",
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                  marginTop: 6,
                  lineHeight: 1.4,
                }}
              >
                {subtitle}
                {subDetail}
              </div>
            </div>
            <span
              style={{
                color: "rgba(148,163,184,0.45)",
                fontSize: "0.65rem",
                marginTop: 2,
              }}
            >
              {isFocused ? "▼" : "▶"}
            </span>
          </div>
        </button>
        <div
          style={{
            padding: "0 10px 10px",
            display: "flex",
            justifyContent: "center",
          }}
        >
          <button
            type="button"
            onMouseDown={stopFlowEvent}
            onPointerDown={stopFlowEvent}
            onClick={(e) => {
              e.stopPropagation();
              onShowMore();
            }}
            className="nodrag nopan"
            style={{
              fontSize: "0.65rem",
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              padding: "6px 14px",
              borderRadius: 6,
              border: `1px solid ${nodeBorder}`,
              background: "rgba(255,255,255,0.04)",
              color: "rgba(203,213,225,0.85)",
              cursor: "pointer",
              fontFamily: "inherit",
              pointerEvents: "auto",
            }}
          >
            Show more
          </button>
        </div>
      </div>
      <Handle
        id="source-left"
        type="source"
        position={Position.Left}
        style={{
          width: 10,
          height: 10,
          background: "#888",
          border: "2px solid #1a1d26",
        }}
      />
      <Handle
        id="source-right"
        type="source"
        position={Position.Right}
        style={{
          width: 10,
          height: 10,
          background: "#888",
          border: "2px solid #1a1d26",
        }}
      />
    </>
  );
}

const nodeTypes: NodeTypes = {
  agentNode: AgentNode,
  triggerNode: TriggerNode,
};

function buildNodes(
  pipeline: PipelineStepState[],
  focusedIndex: number,
  detailsOpenStepId: string | null,
  onFocusHeader: (index: number) => void,
  onOpenDetails: (index: number) => void,
  onCloseDetails: () => void,
  expandedLayout: boolean,
): Node[] {
  if (pipeline.length === 0) return [];

  const firstAgentX = agentX(0);
  const firstNodeCenterX = firstAgentX + NODE_W / 2;
  const triggerX = firstNodeCenterX - TRIGGER_SIZE / 2;
  const triggerY = TOP_PAD;
  const firstAgentY = triggerY + TRIGGER_SIZE + GAP_AFTER_TRIGGER;

  const nodes: Node[] = [
    {
      id: "__trigger__",
      type: "triggerNode",
      position: { x: triggerX, y: triggerY },
      data: {},
      draggable: false,
      selectable: false,
    },
  ];

  pipeline.forEach((step, i) => {
    nodes.push({
      id: step.stepId,
      type: "agentNode",
      position: { x: agentX(i), y: firstAgentY + i * VERTICAL_GAP },
      data: {
        step,
        isFocused: i === focusedIndex,
        onFocusHeader: () => onFocusHeader(i),
        onShowMore: () => onOpenDetails(i),
        detailsOpen: detailsOpenStepId === step.stepId,
        detailsText: formatPsLog(step),
        onCloseDetails,
        expandedLayout,
      },
      draggable: false,
    });
  });

  return nodes;
}

function buildEdges(pipeline: PipelineStepState[]): Edge[] {
  const edges: Edge[] = [];

  if (pipeline.length === 0) return edges;

  const first = pipeline[0];
  edges.push({
    id: "e-trigger-first",
    source: "__trigger__",
    target: first.stepId,
    sourceHandle: "source-bottom",
    targetHandle: "target-top",
    type: "smoothstep",
    animated: first.status === "running",
    style: {
      stroke: first.status === "running" ? edgeActive : edgeIdle,
      strokeWidth: 2,
    },
  });

  for (let i = 0; i < pipeline.length - 1; i++) {
    const a = pipeline[i];
    const b = pipeline[i + 1];
    const nextRunning = b.status === "running";
    const useRight = i % 2 === 0;
    edges.push({
      id: `e-${a.stepId}-${b.stepId}`,
      source: a.stepId,
      target: b.stepId,
      sourceHandle: useRight ? "source-right" : "source-left",
      targetHandle: useRight ? "target-right" : "target-left",
      type: "smoothstep",
      animated: nextRunning,
      style: {
        stroke: nextRunning ? edgeActive : edgeIdle,
        strokeWidth: 2,
      },
    });
  }

  return edges;
}

function FitViewOnSteps({ stepCount }: { stepCount: number }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    if (stepCount > 0) {
      const t = window.setTimeout(() => {
        fitView({ padding: 0.12, duration: 260, maxZoom: 1 });
      }, 60);
      return () => window.clearTimeout(t);
    }
  }, [stepCount, fitView]);
  return null;
}

export interface ClaimPipelineCanvasProps {
  pipeline: PipelineStepState[];
  /** Optional: sync parent expanded state when user focuses a step */
  onStepSelect?: (stepId: string) => void;
  loading: boolean;
  /** Wider/taller canvas and detail popover (e.g. while claim is streaming). */
  layout?: "default" | "expanded";
}

function ClaimPipelineCanvasInner({
  pipeline,
  onStepSelect,
  loading,
  layout = "default",
}: ClaimPipelineCanvasProps) {
  const [focusedIndex, setFocusedIndex] = useState(0);
  const [followLive, setFollowLive] = useState(true);
  const [detailsOpenStepId, setDetailsOpenStepId] = useState<string | null>(
    null,
  );

  const autoFocusIndex = useMemo(() => {
    if (pipeline.length === 0) return 0;
    const ri = pipeline.findIndex((s) => s.status === "running");
    if (ri !== -1) return ri;
    let lastDone = -1;
    pipeline.forEach((s, i) => {
      if (s.status === "done") lastDone = i;
    });
    return lastDone >= 0 ? lastDone : 0;
  }, [pipeline]);

  const displayIndex = followLive ? autoFocusIndex : focusedIndex;
  const safeIndex = Math.min(
    Math.max(0, displayIndex),
    Math.max(0, pipeline.length - 1),
  );

  const selectStep = useCallback(
    (index: number, userInitiated: boolean) => {
      if (userInitiated) setFollowLive(false);
      const clamped = Math.max(0, Math.min(index, pipeline.length - 1));
      setFocusedIndex(clamped);
      const step = pipeline[clamped];
      if (step && userInitiated) {
        onStepSelect?.(step.stepId);
      }
    },
    [pipeline, onStepSelect],
  );

  const closeDetails = useCallback(() => {
    setDetailsOpenStepId(null);
  }, []);

  const onFocusHeader = useCallback(
    (index: number) => {
      setDetailsOpenStepId(null);
      selectStep(index, true);
    },
    [selectStep],
  );

  const onOpenDetails = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(index, pipeline.length - 1));
      const step = pipeline[clamped];
      if (step) setDetailsOpenStepId(step.stepId);
      selectStep(index, true);
    },
    [pipeline, selectStep],
  );

  const goNext = useCallback(() => {
    const cur = followLive ? autoFocusIndex : focusedIndex;
    setFollowLive(false);
    setDetailsOpenStepId(null);
    const max = Math.max(0, pipeline.length - 1);
    const ni = Math.min(cur + 1, max);
    setFocusedIndex(ni);
    const s = pipeline[ni];
    if (s) onStepSelect?.(s.stepId);
  }, [
    pipeline,
    followLive,
    autoFocusIndex,
    focusedIndex,
    onStepSelect,
  ]);

  const goPrev = useCallback(() => {
    const cur = followLive ? autoFocusIndex : focusedIndex;
    setFollowLive(false);
    setDetailsOpenStepId(null);
    const pi = Math.max(cur - 1, 0);
    setFocusedIndex(pi);
    const s = pipeline[pi];
    if (s) onStepSelect?.(s.stepId);
  }, [
    pipeline,
    followLive,
    autoFocusIndex,
    focusedIndex,
    onStepSelect,
  ]);

  const followLiveClick = useCallback(() => {
    setFollowLive(true);
    setDetailsOpenStepId(null);
  }, []);

  const expandedLayout = layout === "expanded";

  const nextNodes = useMemo(
    () =>
      buildNodes(
        pipeline,
        safeIndex,
        detailsOpenStepId,
        onFocusHeader,
        onOpenDetails,
        closeDetails,
        expandedLayout,
      ),
    [
      pipeline,
      safeIndex,
      detailsOpenStepId,
      onFocusHeader,
      onOpenDetails,
      closeDetails,
      expandedLayout,
    ],
  );
  const nextEdges = useMemo(() => buildEdges(pipeline), [pipeline]);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(nextNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(nextEdges);

  useEffect(() => {
    setNodes(nextNodes);
  }, [nextNodes, setNodes]);

  useEffect(() => {
    setEdges(nextEdges);
  }, [nextEdges, setEdges]);

  const flowH = flowHeightForStepCount(pipeline.length, expandedLayout);

  const navBtn: CSSProperties = {
    fontSize: "0.65rem",
    fontWeight: 600,
    letterSpacing: "0.06em",
    textTransform: "uppercase",
    padding: "5px 10px",
    borderRadius: 6,
    border: `1px solid ${nodeBorder}`,
    background: "rgba(255,255,255,0.05)",
    color: "rgba(203,213,225,0.9)",
    cursor: "pointer",
    fontFamily: "inherit",
  };

  return (
    <div
      style={{
        borderRadius: 12,
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.06)",
        background: canvasBg,
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          padding: "10px 14px",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          background: "rgba(0,0,0,0.25)",
        }}
      >
        <span
          style={{
            fontSize: "0.68rem",
            fontWeight: 700,
            letterSpacing: "0.12em",
            color: "rgba(148,163,184,0.65)",
          }}
        >
          AGENT PIPELINE
        </span>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 6,
          }}
        >
          <button type="button" style={navBtn} onClick={goPrev}>
            Previous
          </button>
          <button type="button" style={navBtn} onClick={goNext}>
            Next
          </button>
          <button
            type="button"
            style={{
              ...navBtn,
              borderColor: followLive ? accent : nodeBorder,
              color: followLive ? "#fecaca" : navBtn.color,
            }}
            onClick={followLiveClick}
          >
            Follow live
          </button>
          <span style={{ fontSize: "0.68rem", color: "rgba(148,163,184,0.4)" }}>
            {pipeline.length} step{pipeline.length === 1 ? "" : "s"}
          </span>
        </div>
      </div>

      <div style={{ width: "100%", height: flowH, position: "relative" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.12, maxZoom: 1 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag={false}
          panOnScroll={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          minZoom={0.85}
          maxZoom={1.05}
          proOptions={{ hideAttribution: true }}
          defaultEdgeOptions={{
            type: "smoothstep",
          }}
        >
          <Background
            id="claim-pipeline-bg"
            variant={BackgroundVariant.Dots}
            gap={18}
            size={1.2}
            color="rgba(100,116,139,0.22)"
          />
          <Controls
            showInteractive={false}
            className="claim-flow-controls"
            style={{
              background: "rgba(26,29,38,0.92)",
              border: `1px solid ${nodeBorder}`,
              borderRadius: 8,
            }}
          />
          <FitViewOnSteps stepCount={pipeline.length} />
        </ReactFlow>
      </div>

      {loading && pipeline.length > 0 && pipeline.every((s) => s.status === "pending") && (
        <div
          style={{
            padding: "8px 14px 10px",
            fontSize: "0.78rem",
            color: "rgba(148,163,184,0.55)",
            borderTop: "1px solid rgba(255,255,255,0.05)",
          }}
        >
          Connecting to claim engine…
        </div>
      )}

      {loading && pipeline.some((s) => s.status === "running") && (
        <p
          style={{
            margin: 0,
            padding: "8px 14px 12px",
            fontSize: "0.7rem",
            color: "rgba(148,163,184,0.45)",
            borderTop: "1px solid rgba(255,255,255,0.05)",
          }}
        >
          Use <strong style={{ color: "rgba(203,213,225,0.75)" }}>Show more</strong>{" "}
          on a step to open execution details beside the node. Next/Previous and
          Follow live still track the run. Typical run: 30–60s.
        </p>
      )}
    </div>
  );
}

export function ClaimPipelineCanvas(props: ClaimPipelineCanvasProps) {
  return (
    <ReactFlowProvider>
      <ClaimPipelineCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
