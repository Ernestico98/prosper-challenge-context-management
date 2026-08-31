import React, { useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MarkerType,
  Position,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";
import NodeInspector from "./NodeInspector.jsx";

const COLUMN = 300;
const ROW = 190;
// Lines and arrowheads are set here rather than in styles.css: React Flow writes
// arrowheads into the SVG <defs> with inline attributes, out of CSS's reach.
const EDGE_COLOR = "#7d8aa3";

/**
 * Lay the graph out by distance from the entry node.
 *
 * This is the FALLBACK, not the layout. An agent JSON is not required to carry
 * coordinates — a Copilot generating one should not have to invent them — so a
 * graph that has never been arranged still opens as something readable. Once a
 * human drags a node, their position is authoritative and gets saved with the
 * agent: breadth-first order is a reasonable guess, not a good layout.
 */
function autoLayout(agent) {
  const byName = Object.fromEntries(agent.nodes.map((n) => [n.name, n]));
  const depth = { [agent.initial_node]: 0 };
  const queue = [agent.initial_node];

  while (queue.length) {
    const current = queue.shift();
    for (const edge of byName[current]?.edges || []) {
      if (depth[edge.target] === undefined) {
        depth[edge.target] = depth[current] + 1;
        queue.push(edge.target);
      }
    }
  }

  const perColumn = {};
  return Object.fromEntries(
    agent.nodes.map((node) => {
      const column = depth[node.name] ?? 0;
      const row = perColumn[column] || 0;
      perColumn[column] = row + 1;
      return [node.name, { x: 60 + column * COLUMN, y: 40 + row * ROW }];
    })
  );
}

function nodeLabel(node, isInitial) {
  const badges = [];
  if (isInitial) badges.push("entry");
  if (node.end) badges.push("ends call");
  if (node.context_strategy === "reset") badges.push("context reset");

  return (
    <div className="flow-node">
      <div className="flow-node-title">{node.name}</div>
      {badges.length > 0 && (
        <div className="flow-node-badges">
          {badges.map((b) => (
            <span key={b} className={`badge ${b === "context reset" ? "reset" : ""}`}>
              {b}
            </span>
          ))}
        </div>
      )}
      {node.tools?.length > 0 && (
        <div className="flow-node-tools">
          {node.tools.map((t) => (
            <span key={t} className="tool-chip">
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function GraphEditor({ agent, tools, onChange }) {
  const [selected, setSelected] = useState(agent.initial_node);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);

  /**
   * Rebuild the flow nodes only when the graph's *content* changes, never on a
   * position change. Regenerating them on every render is what makes a node
   * snap back the instant you let go of it.
   */
  const signature = useMemo(
    () =>
      JSON.stringify([
        agent.initial_node,
        selected,
        agent.nodes.map((n) => [n.name, n.tools, n.end, n.context_strategy]),
      ]),
    [agent, selected]
  );

  useEffect(() => {
    const fallback = autoLayout(agent);
    setNodes((current) => {
      const live = Object.fromEntries(current.map((n) => [n.id, n.position]));
      return agent.nodes.map((node) => ({
        id: node.name,
        // Saved position wins, then wherever it is on screen right now, then
        // the automatic layout for a node that has never been placed.
        position: node.position || live[node.name] || fallback[node.name],
        data: { label: nodeLabel(node, node.name === agent.initial_node) },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        className: node.name === selected ? "node selected" : "node",
      }));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature]);

  const edges = useMemo(
    () =>
      agent.nodes.flatMap((node) =>
        (node.edges || []).map((edge) => ({
          id: `${node.name}-${edge.function}-${edge.target}`,
          source: node.name,
          target: edge.target,
          label: edge.function,
          style: { stroke: EDGE_COLOR, strokeWidth: 1.5 },
          labelBgPadding: [6, 3],
          labelBgBorderRadius: 4,
          markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLOR, width: 18, height: 18 },
        }))
      ),
    [agent]
  );

  const updateNode = useCallback(
    (name, changes) => {
      onChange({
        ...agent,
        nodes: agent.nodes.map((node) =>
          node.name === name ? { ...node, ...changes } : node
        ),
      });
    },
    [agent, onChange]
  );

  // Persist the drop, not every frame of the drag: the position becomes part of
  // the agent JSON, so the arrangement survives a reload once saved.
  const onNodeDragStop = useCallback(
    (_, node) => {
      const position = { x: Math.round(node.position.x), y: Math.round(node.position.y) };
      const stored = agent.nodes.find((n) => n.name === node.id)?.position;
      // A click registers as a zero-distance drag. Writing it would mark the
      // agent unsaved for having looked at it.
      if (stored && stored.x === position.x && stored.y === position.y) return;
      updateNode(node.id, { position });
    },
    [agent, updateNode]
  );

  const relayout = useCallback(() => {
    const fallback = autoLayout(agent);
    onChange({
      ...agent,
      nodes: agent.nodes.map((node) => ({ ...node, position: fallback[node.name] })),
    });
    setNodes((current) =>
      current.map((node) => ({ ...node, position: fallback[node.id] }))
    );
  }, [agent, onChange, setNodes]);

  const selectedNode = agent.nodes.find((n) => n.name === selected);

  return (
    <div className="graph-layout">
      <div className="canvas">
        <div className="canvas-toolbar">
          <button className="ghost" onClick={relayout} title="Re-run the automatic layout">
            Auto-arrange
          </button>
        </div>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onNodeDragStop={onNodeDragStop}
          onNodeClick={(_, node) => setSelected(node.id)}
          fitView
          fitViewOptions={{ padding: 0.18 }}
          minZoom={0.2}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={18} size={1} color="#2b3240" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>

      <NodeInspector
        agent={agent}
        node={selectedNode}
        tools={tools}
        onChange={(changes) => updateNode(selected, changes)}
      />
    </div>
  );
}
