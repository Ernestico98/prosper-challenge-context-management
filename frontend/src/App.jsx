import React, { useCallback, useEffect, useState } from "react";
import GraphEditor from "./GraphEditor.jsx";
import CatalogBrowser from "./CatalogBrowser.jsx";
import TestCall from "./TestCall.jsx";
import {
  activateAgent,
  agentTemplate,
  createAgent,
  deleteAgent,
  getAgent,
  listAgents,
  listTools,
  saveAgent,
} from "./api.js";

const TABS = [
  { id: "graph", label: "Agent graph" },
  { id: "catalog", label: "Catalog" },
  { id: "call", label: "Test call" },
];

export default function App() {
  const [tab, setTab] = useState("graph");
  const [agents, setAgents] = useState([]);
  const [agentId, setAgentId] = useState(null);
  const [agent, setAgent] = useState(null);
  const [tools, setTools] = useState([]);
  const [status, setStatus] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [newName, setNewName] = useState(null); // null = not creating

  const live = agents.find((a) => a.active);
  const isLive = live?.id === agentId;

  const refreshAgents = useCallback(
    async (select) => {
      const { agents: list } = await listAgents();
      setAgents(list);
      if (select) setAgentId(select);
      else if (!agentId) setAgentId((list.find((a) => a.active) || list[0])?.id);
      return list;
    },
    [agentId]
  );

  useEffect(() => {
    Promise.all([refreshAgents(), listTools()])
      .then(([, t]) => setTools(t.tools))
      .catch((error) => setStatus({ kind: "error", text: error.message }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!agentId) return;
    getAgent(agentId)
      .then((data) => {
        setAgent(data);
        setDirty(false);
      })
      .catch((error) => setStatus({ kind: "error", text: error.message }));
  }, [agentId]);

  const update = useCallback((next) => {
    setAgent(next);
    setDirty(true);
    setStatus(null);
  }, []);

  const run = useCallback(async (work, ok) => {
    try {
      await work();
      setStatus(ok ? { kind: "ok", text: ok } : null);
    } catch (error) {
      setStatus({ kind: "error", text: error.message });
    }
  }, []);

  const save = () =>
    run(async () => {
      await saveAgent(agentId, agent);
      setDirty(false);
      await refreshAgents(agentId);
      // The runner re-reads the agent JSON on every connection, so a save is
      // live on the next call. No restart, no deploy step.
    }, "Saved. Reconnect the test call to run it.");

  const create = () =>
    run(async () => {
      const id = newName.trim();
      const template = await agentTemplate();
      await createAgent(id, { ...template, name: id });
      setNewName(null);
      await refreshAgents(id);
    }, "Agent created. Set it live to call it.");

  const activate = () =>
    run(async () => {
      await activateAgent(agentId);
      await refreshAgents(agentId);
    }, "Live. Reconnect the test call to run it.");

  const remove = () =>
    run(async () => {
      await deleteAgent(agentId);
      setAgentId(null);
      setAgent(null);
      const list = await refreshAgents();
      setAgentId((list.find((a) => a.active) || list[0])?.id);
    }, "Agent deleted.");

  return (
    <div className="app">
      <header>
        <div className="brand">
          <strong>Prosper</strong> agent builder
        </div>

        <select
          value={agentId || ""}
          onChange={(event) => setAgentId(event.target.value)}
          disabled={!agents.length}
        >
          {agents.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name} ({a.nodes} nodes){a.active ? " — live" : ""}
            </option>
          ))}
        </select>

        <button className="ghost" onClick={() => setNewName("")} title="Create a new agent">
          + New
        </button>

        {isLive ? (
          <span className="badge live-badge">live</span>
        ) : (
          <button className="ghost" onClick={activate} disabled={!agentId}>
            Set live
          </button>
        )}

        <nav>
          {TABS.map((t) => (
            <button
              key={t.id}
              className={tab === t.id ? "tab active" : "tab"}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div className="spacer" />
        {status && <span className={`status ${status.kind}`}>{status.text}</span>}
        <button className="primary" onClick={save} disabled={!dirty || !agent}>
          {dirty ? "Save agent" : "Saved"}
        </button>
      </header>

      {newName !== null && (
        // An inline row rather than window.prompt: a browser modal blocks the
        // page, and this can validate as you type.
        <div className="new-agent-bar">
          <label>
            New agent id
            <input
              autoFocus
              value={newName}
              placeholder="clinic_reception"
              onChange={(event) => setNewName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && /^[A-Za-z0-9][\w.-]*$/.test(newName)) create();
                if (event.key === "Escape") setNewName(null);
              }}
            />
          </label>
          <span className="hint">
            Letters, digits, dots, dashes and underscores. Starts with a letter or digit.
          </span>
          <div className="spacer" />
          <button
            className="primary"
            onClick={create}
            disabled={!/^[A-Za-z0-9][\w.-]*$/.test(newName)}
          >
            Create
          </button>
          <button className="ghost" onClick={() => setNewName(null)}>
            Cancel
          </button>
        </div>
      )}

      <main>
        {tab === "graph" && agent && (
          <GraphEditor
            agent={agent}
            tools={tools}
            onChange={update}
            onDeleteAgent={remove}
            isLive={isLive}
          />
        )}
        {tab === "catalog" && <CatalogBrowser />}
        {tab === "call" && <TestCall agent={agent} live={live} />}
        {tab === "graph" && !agent && <div className="empty">Loading agent…</div>}
      </main>
    </div>
  );
}
