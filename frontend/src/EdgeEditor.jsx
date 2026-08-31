import React from "react";

const TYPES = ["string", "boolean", "integer", "number"];

/**
 * One edge, including the slots it collects.
 *
 * The properties table is the part that looks like plumbing and is not: these
 * are the arguments the model fills in when it takes the edge, and they land in
 * `flow_manager.state`. That makes them the only things that survive a context
 * reset — `provider_preference` exists precisely because a preference stated
 * before a reset was otherwise lost.
 */
export default function EdgeEditor({ edge, index, nodes, onChange, onDelete }) {
  const properties = edge.properties || {};
  const required = edge.required || [];

  const setProperty = (name, changes) =>
    onChange({ properties: { ...properties, [name]: { ...properties[name], ...changes } } });

  const renameProperty = (from, to) => {
    if (!to || to === from || properties[to]) return;
    const next = Object.fromEntries(
      Object.entries(properties).map(([key, value]) => [key === from ? to : key, value])
    );
    onChange({
      properties: next,
      required: required.map((name) => (name === from ? to : name)),
    });
  };

  const removeProperty = (name) => {
    const { [name]: _dropped, ...rest } = properties;
    onChange({ properties: rest, required: required.filter((r) => r !== name) });
  };

  const addProperty = () => {
    let name = "slot";
    for (let i = 2; properties[name]; i += 1) name = `slot_${i}`;
    onChange({
      properties: { ...properties, [name]: { type: "string", description: "" } },
    });
  };

  const toggleRequired = (name) =>
    onChange({
      required: required.includes(name)
        ? required.filter((r) => r !== name)
        : [...required, name],
    });

  return (
    <div className="edge-card">
      <div className="edge-card-head">
        <input
          className="edge-fn"
          value={edge.function}
          onChange={(event) => onChange({ function: event.target.value })}
        />
        <button className="danger-link" onClick={() => onDelete(index)} title="Delete edge">
          ✕
        </button>
      </div>

      <label>
        Goes to
        <select value={edge.target} onChange={(event) => onChange({ target: event.target.value })}>
          {nodes.map((node) => (
            <option key={node.name} value={node.name}>
              {node.name}
            </option>
          ))}
        </select>
      </label>

      <label>
        When to take it
        <input
          value={edge.description || ""}
          placeholder="When the model should call it"
          onChange={(event) => onChange({ description: event.target.value })}
        />
      </label>

      <div className="slots">
        <div className="slots-head">
          <span>Slots collected — these survive a context reset</span>
          <button className="ghost tiny" onClick={addProperty}>
            + Slot
          </button>
        </div>

        {Object.keys(properties).length === 0 && (
          <p className="hint">None. The edge just moves the conversation on.</p>
        )}

        {Object.entries(properties).map(([name, spec]) => (
          <div className="slot-row" key={name}>
            <input
              className="slot-name"
              defaultValue={name}
              onBlur={(event) => renameProperty(name, event.target.value.trim())}
            />
            <select
              value={spec.type || "string"}
              onChange={(event) => setProperty(name, { type: event.target.value })}
            >
              {TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
            <label className="check tiny" title="Required">
              <input
                type="checkbox"
                checked={required.includes(name)}
                onChange={() => toggleRequired(name)}
              />
              <span>req</span>
            </label>
            <button className="danger-link" onClick={() => removeProperty(name)}>
              ✕
            </button>
            <input
              className="slot-description"
              value={spec.description || ""}
              placeholder="What this slot means, in the model's words"
              onChange={(event) => setProperty(name, { description: event.target.value })}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
