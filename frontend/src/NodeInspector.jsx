import React from "react";

/**
 * Editing surface for one node.
 *
 * The three controls that matter for Phase 2 are here: the prompt, the data
 * tools the node may call, and whether entering it resets the context. Edges
 * are shown but their targets are what the graph is made of, so they are edited
 * as data rather than dragged.
 */
export default function NodeInspector({ agent, node, tools, onChange }) {
  if (!node) return <aside className="inspector empty">Select a node.</aside>;

  const prompt = node.task_messages?.[0]?.content || "";
  const otherNodes = agent.nodes.filter((n) => n.name !== node.name);

  const setPrompt = (content) =>
    onChange({
      task_messages: [{ role: "developer", ...(node.task_messages?.[0] || {}), content }],
    });

  const toggleTool = (name) => {
    const current = node.tools || [];
    onChange({
      tools: current.includes(name)
        ? current.filter((t) => t !== name)
        : [...current, name],
    });
  };

  const setEdge = (index, changes) =>
    onChange({
      edges: node.edges.map((edge, i) => (i === index ? { ...edge, ...changes } : edge)),
    });

  const placeholders = [...prompt.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);

  return (
    <aside className="inspector">
      <h2>{node.name}</h2>

      <label>
        Prompt
        <textarea
          value={prompt}
          rows={10}
          onChange={(event) => setPrompt(event.target.value)}
          spellCheck={false}
        />
      </label>
      {placeholders.length > 0 && (
        <p className="hint">
          Fills from state at build time: {placeholders.map((p) => `{${p}}`).join(", ")}
        </p>
      )}

      <label>
        Context on entering this node
        <select
          value={node.context_strategy || "append"}
          onChange={(event) =>
            onChange({
              context_strategy:
                event.target.value === "append" ? undefined : event.target.value,
            })
          }
        >
          <option value="append">Append — keep the conversation so far</option>
          <option value="reset">Reset — drop it and start from this prompt</option>
        </select>
      </label>
      {node.context_strategy === "reset" && (
        <p className="hint warn">
          Everything said so far is discarded. Anything this node still needs must come
          back through a {"{placeholder}"} above — including the caller's language.
        </p>
      )}

      <fieldset>
        <legend>Data tools — run and stay on this step</legend>
        {tools.map((tool) => (
          <label key={tool.name} className="check" title={tool.description}>
            <input
              type="checkbox"
              checked={(node.tools || []).includes(tool.name)}
              onChange={() => toggleTool(tool.name)}
            />
            <span>{tool.name}</span>
          </label>
        ))}
      </fieldset>

      <fieldset>
        <legend>Edges — transition to another node</legend>
        {(node.edges || []).length === 0 && (
          <p className="hint">
            {node.end ? "Terminal node: the call ends here." : "No transitions out."}
          </p>
        )}
        {(node.edges || []).map((edge, index) => (
          <div className="edge-row" key={edge.function}>
            <code>{edge.function}</code>
            <select
              value={edge.target}
              onChange={(event) => setEdge(index, { target: event.target.value })}
            >
              {otherNodes.map((n) => (
                <option key={n.name} value={n.name}>
                  {n.name}
                </option>
              ))}
            </select>
            <input
              value={edge.description}
              onChange={(event) => setEdge(index, { description: event.target.value })}
              placeholder="When the model should take this edge"
            />
          </div>
        ))}
      </fieldset>
    </aside>
  );
}
