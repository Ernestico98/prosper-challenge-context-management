import React, { useCallback, useEffect, useState } from "react";
import GraphEditor from "./GraphEditor.jsx";
import CatalogBrowser from "./CatalogBrowser.jsx";
import TestCall from "./TestCall.jsx";
import { getAgent, listAgents, listTools, saveAgent } from "./api.js";

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

  useEffect(() => {
    Promise.all([listAgents(), listTools()])
      .then(([a, t]) => {
        setAgents(a.agents);
        setTools(t.tools);
        const active = a.agents.find((x) => x.active) || a.agents[0];
        if (active) setAgentId(active.id);
      })
      .catch((error) => setStatus({ kind: "error", text: error.message }));
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

  const save = useCallback(async () => {
    try {
      await saveAgent(agentId, agent);
      setDirty(false);
      // The runner re-reads the agent JSON on every connection, so a save is
      // live on the next call. No restart, no deploy step.
      setStatus({ kind: "ok", text: "Saved. Reconnect the test call to run it." });
    } catch (error) {
      setStatus({ kind: "error", text: error.message });
    }
  }, [agentId, agent]);

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

      <main>
        {tab === "graph" && agent && (
          <GraphEditor agent={agent} tools={tools} onChange={update} />
        )}
        {tab === "catalog" && <CatalogBrowser />}
        {tab === "call" && <TestCall agent={agent} />}
        {tab === "graph" && !agent && <div className="empty">Loading agent…</div>}
      </main>
    </div>
  );
}
