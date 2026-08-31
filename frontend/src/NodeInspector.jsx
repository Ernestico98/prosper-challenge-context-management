import React, { useEffect, useState } from "react";
import EdgeEditor from "./EdgeEditor.jsx";
import { isValidNodeName } from "./graphOps.js";

/**
 * Editing surface for one node.
 *
 * Renaming is committed on blur rather than on every keystroke, because a
 * rename cascades: every edge that targets this node, and the agent's entry
 * node if it is this one, are rewritten with it. Doing that per character
 * would rewrite the graph a dozen times while someone types.
 */
export default function NodeInspector({
  agent,
  node,
  tools,
  warnings,
  onChange,
  onRename,
  onDelete,
  onMakeEntry,
  onEdgeChange,
  onEdgeDelete,
  onEdgeAdd,
}) {
  const [draftName, setDraftName] = useState(node.name);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    setDraftName(node.name);
    setConfirming(false);
  }, [node.name]);

  const prompt = node.task_messages?.[0]?.content || "";
  const isEntry = agent.initial_node === node.name;
  const nameTaken = draftName !== node.name && agent.nodes.some((n) => n.name === draftName);
  const nameOk = isValidNodeName(draftName) && !nameTaken;

  const setPrompt = (content) =>
    onChange({
      task_messages: [{ role: "developer", ...(node.task_messages?.[0] || {}), content }],
    });

  const toggleTool = (name) => {
    const current = node.tools || [];
    onChange({
      tools: current.includes(name) ? current.filter((t) => t !== name) : [...current, name],
    });
  };

  const placeholders = [...prompt.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);

  return (
    <aside className="inspector">
      <label>
        Node name
        <input
          className={nameOk ? "" : "invalid"}
          value={draftName}
          onChange={(event) => setDraftName(event.target.value)}
          onBlur={() => (nameOk ? onRename(draftName) : setDraftName(node.name))}
          onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
        />
      </label>
      {nameTaken && <p className="hint warn">Another node already has that name.</p>}
      {warnings?.length > 0 && (
        <p className="hint warn">⚠ {warnings.join(". ")}.</p>
      )}

      <div className="node-flags">
        <label className="check" title="Where every call starts">
          <input type="checkbox" checked={isEntry} onChange={onMakeEntry} disabled={isEntry} />
          <span>Entry node</span>
        </label>
        <label className="check" title="Hangs up once this node has spoken">
          <input
            type="checkbox"
            checked={!!node.end}
            onChange={(event) => onChange({ end: event.target.checked })}
          />
          <span>Ends the call</span>
        </label>
      </div>

      <label>
        Prompt
        <textarea
          value={prompt}
          rows={9}
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
              context_strategy: event.target.value === "append" ? undefined : event.target.value,
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
        <legend>
          Edges — transition to another node
          <button
            className="ghost tiny"
            onClick={() =>
              onEdgeAdd({
                function: "go_next",
                description: "When the model should take this edge.",
                target: agent.nodes.find((n) => n.name !== node.name)?.name || node.name,
              })
            }
            disabled={agent.nodes.length < 2}
          >
            + Edge
          </button>
        </legend>

        {(node.edges || []).length === 0 && (
          <p className="hint">
            {node.end
              ? "Terminal node: the call ends here."
              : "No transitions out. Drag from this node's right handle to another, or add one here."}
          </p>
        )}

        {(node.edges || []).map((edge, index) => (
          <EdgeEditor
            key={`${edge.function}-${index}`}
            edge={edge}
            index={index}
            nodes={agent.nodes.filter((n) => n.name !== node.name)}
            onChange={(changes) => onEdgeChange(index, changes)}
            onDelete={onEdgeDelete}
          />
        ))}
      </fieldset>

      <fieldset>
        <legend>Danger</legend>
        <button
          className="danger"
          onClick={() => (confirming ? onDelete() : setConfirming(true))}
          onBlur={() => setConfirming(false)}
          disabled={agent.nodes.length < 2}
        >
          {confirming ? "Click again — inbound edges go too" : "Delete this node"}
        </button>
      </fieldset>
    </aside>
  );
}
