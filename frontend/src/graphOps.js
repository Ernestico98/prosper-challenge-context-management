/**
 * Pure operations on an agent graph.
 *
 * Every one takes an agent and returns a new one. They live apart from the
 * editor because the interesting part is not the React plumbing, it is that
 * node names are identity: edges target them by name and `initial_node` names
 * one. So renaming and deleting have to fix up the whole graph in a single
 * step, or a save can catch it half-updated — and the API compiles before it
 * writes, so a half-updated graph is a 422 rather than a silent corruption.
 */

const NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function isValidNodeName(name) {
  return NAME_PATTERN.test(name || "");
}

/** The name `addNode` will use. Exported so the editor can select the node it
 *  just created without the ops smuggling UI state back inside the agent —
 *  anything added to that object gets written to the agent's JSON on save. */
export function nextNodeName(agent, base = "new_step") {
  const taken = new Set(agent.nodes.map((n) => n.name));
  if (!taken.has(base)) return base;
  for (let i = 2; ; i += 1) {
    if (!taken.has(`${base}_${i}`)) return `${base}_${i}`;
  }
}

export function addNode(agent, position) {
  const name = nextNodeName(agent);
  return {
    ...agent,
    nodes: [
      ...agent.nodes,
      {
        name,
        task_messages: [{ role: "developer", content: "" }],
        edges: [],
        tools: [],
        ...(position ? { position } : {}),
      },
    ],
  };
}

export function deleteNode(agent, name) {
  const remaining = agent.nodes.filter((n) => n.name !== name);
  if (!remaining.length) return agent; // an agent with no nodes will not compile

  return {
    ...agent,
    // Inbound edges go with the node. Leaving them would make the graph
    // uncompilable, and the user did not ask to break their agent.
    nodes: remaining.map((node) => ({
      ...node,
      edges: (node.edges || []).filter((edge) => edge.target !== name),
    })),
    initial_node:
      agent.initial_node === name ? remaining[0].name : agent.initial_node,
  };
}

export function renameNode(agent, from, to) {
  if (from === to) return agent;
  if (!isValidNodeName(to)) return agent;
  if (agent.nodes.some((n) => n.name === to)) return agent; // names are identity

  return {
    ...agent,
    nodes: agent.nodes.map((node) => ({
      ...(node.name === from ? { ...node, name: to } : node),
      edges: (node.edges || []).map((edge) =>
        edge.target === from ? { ...edge, target: to } : edge
      ),
    })),
    initial_node: agent.initial_node === from ? to : agent.initial_node,
  };
}

export function updateNode(agent, name, changes) {
  return {
    ...agent,
    nodes: agent.nodes.map((node) => (node.name === name ? { ...node, ...changes } : node)),
  };
}

export function setEntryNode(agent, name) {
  return { ...agent, initial_node: name };
}

export function addEdge(agent, source, edge) {
  return updateNode(agent, source, {
    edges: [...(agent.nodes.find((n) => n.name === source)?.edges || []), edge],
  });
}

export function updateEdge(agent, source, index, changes) {
  const node = agent.nodes.find((n) => n.name === source);
  return updateNode(agent, source, {
    edges: node.edges.map((edge, i) => (i === index ? { ...edge, ...changes } : edge)),
  });
}

export function deleteEdge(agent, source, index) {
  const node = agent.nodes.find((n) => n.name === source);
  return updateNode(agent, source, {
    edges: node.edges.filter((_, i) => i !== index),
  });
}

/**
 * Problems worth showing but not worth blocking on.
 *
 * You naturally create a node before wiring it, so a half-built graph has to
 * stay saveable. The things that cannot compile at all — duplicate names, edges
 * to nowhere, unknown tools — are rejected by the API instead.
 */
export function warnings(agent) {
  const found = [];
  const reachable = new Set([agent.initial_node]);
  const queue = [agent.initial_node];
  const byName = Object.fromEntries(agent.nodes.map((n) => [n.name, n]));

  while (queue.length) {
    for (const edge of byName[queue.shift()]?.edges || []) {
      if (!reachable.has(edge.target)) {
        reachable.add(edge.target);
        queue.push(edge.target);
      }
    }
  }

  for (const node of agent.nodes) {
    if (!node.end && !(node.edges || []).length) {
      found.push({ node: node.name, text: "no way out: add an edge, or mark it as ending the call" });
    }
    if (!reachable.has(node.name)) {
      found.push({ node: node.name, text: "unreachable from the entry node" });
    }
  }
  if (!agent.nodes.some((n) => n.end)) {
    found.push({ node: null, text: "no node ends the call, so it never hangs up" });
  }
  return found;
}
