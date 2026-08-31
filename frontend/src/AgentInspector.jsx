import React, { useState } from "react";

/**
 * Agent-level settings, shown when no node is selected.
 *
 * These live in the inspector rather than behind another tab because they are
 * the same kind of editing as a node's prompt — just one level up — and
 * clicking empty canvas is a natural way to ask "what is this agent?".
 */
export default function AgentInspector({ agent, onChange, onDelete, isLive }) {
  const [confirming, setConfirming] = useState(false);
  const set = (changes) => onChange({ ...agent, ...changes });

  return (
    <aside className="inspector">
      <h2 className="agent-title">Agent settings</h2>
      <p className="hint">Click a node to edit that step instead.</p>

      <label>
        Name
        <input value={agent.name || ""} onChange={(e) => set({ name: e.target.value })} />
      </label>

      <label>
        Persona — the system instruction every node inherits
        <textarea
          value={agent.persona || ""}
          rows={8}
          onChange={(e) => set({ persona: e.target.value })}
          spellCheck={false}
        />
      </label>

      <label>
        Entry node — where every call starts
        <select
          value={agent.initial_node || ""}
          onChange={(e) => set({ initial_node: e.target.value })}
        >
          {agent.nodes.map((node) => (
            <option key={node.name} value={node.name}>
              {node.name}
            </option>
          ))}
        </select>
      </label>

      <div className="two-up">
        <label>
          Model
          <input value={agent.model || ""} onChange={(e) => set({ model: e.target.value })} />
        </label>
        <label>
          Voice id
          <input
            value={agent.voice_id || ""}
            onChange={(e) => set({ voice_id: e.target.value })}
          />
        </label>
      </div>

      <fieldset>
        <legend>Danger</legend>
        {isLive ? (
          <p className="hint">
            This agent is live. Make another one live before deleting it.
          </p>
        ) : (
          <button
            className="danger"
            onClick={() => (confirming ? onDelete() : setConfirming(true))}
            onBlur={() => setConfirming(false)}
          >
            {confirming ? "Click again to delete for good" : "Delete this agent"}
          </button>
        )}
      </fieldset>
    </aside>
  );
}
