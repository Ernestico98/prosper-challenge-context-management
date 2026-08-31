import React, { useState } from "react";

/**
 * Names the edge you just drew.
 *
 * An edge is a tool the model can call, so it is meaningless without a function
 * name and a description of when to take it — those two strings are what the
 * LLM actually sees. Creating a nameless edge on drop and making you find it
 * later would produce agents with `edge_3` in their tool list.
 */
export default function NewEdgeForm({ source, target, existing, onCreate, onCancel }) {
  const [fn, setFn] = useState("");
  const [description, setDescription] = useState("");

  const clash = existing.includes(fn);
  const valid = /^[a-z_][a-z0-9_]*$/.test(fn) && description.trim() && !clash;

  return (
    <div className="edge-form">
      <h3>
        New transition: <code>{source}</code> → <code>{target}</code>
      </h3>

      <label>
        Function name — what the model calls to take this edge
        <input
          autoFocus
          value={fn}
          placeholder="record_details"
          onChange={(event) => setFn(event.target.value)}
        />
      </label>
      {clash && <p className="hint warn">Another edge already uses that name.</p>}

      <label>
        Description — when the model should call it
        <input
          value={description}
          placeholder="Once the caller has given their name and reason."
          onChange={(event) => setDescription(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && valid) onCreate({ function: fn, description });
            if (event.key === "Escape") onCancel();
          }}
        />
      </label>

      <div className="edge-form-actions">
        <button
          className="primary"
          disabled={!valid}
          onClick={() => onCreate({ function: fn, description })}
        >
          Add transition
        </button>
        <button className="ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}
