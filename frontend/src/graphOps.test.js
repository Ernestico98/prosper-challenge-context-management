//
// Graph operation tests.
//
//     node --test frontend/src/          (or: make test-ui)
//
// These are the operations where a mistake is silent and expensive: node names
// are identity, so renaming or deleting one has to fix up every edge that
// targets it and the agent's entry node. Clicking through the browser proves a
// button is wired; this proves the graph survives.
//

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  addEdge,
  addNode,
  deleteEdge,
  deleteNode,
  isValidNodeName,
  nextNodeName,
  renameNode,
  setEntryNode,
  updateEdge,
  warnings,
} from "./graphOps.js";

const agent = () => ({
  name: "Test",
  initial_node: "greeting",
  nodes: [
    {
      name: "greeting",
      task_messages: [],
      edges: [{ function: "go", description: "d", target: "middle" }],
    },
    {
      name: "middle",
      task_messages: [],
      edges: [{ function: "onward", description: "d", target: "farewell" }],
    },
    { name: "farewell", task_messages: [], edges: [], end: true },
  ],
});

const names = (a) => a.nodes.map((n) => n.name);
const targets = (a) => a.nodes.flatMap((n) => (n.edges || []).map((e) => e.target));

describe("renaming", () => {
  it("rewrites every edge that pointed at the old name", () => {
    const renamed = renameNode(agent(), "middle", "collect");
    assert.deepEqual(names(renamed), ["greeting", "collect", "farewell"]);
    assert.ok(!targets(renamed).includes("middle"));
    assert.ok(targets(renamed).includes("collect"));
  });

  it("follows the entry node when that is the one renamed", () => {
    const renamed = renameNode(agent(), "greeting", "hello");
    assert.equal(renamed.initial_node, "hello");
  });

  it("refuses a name another node already has", () => {
    // Names are identity: a duplicate makes one node unreachable, and the API
    // rejects the save outright.
    const attempted = renameNode(agent(), "middle", "farewell");
    assert.deepEqual(names(attempted), names(agent()));
  });

  it("refuses a name that is not an identifier", () => {
    assert.deepEqual(names(renameNode(agent(), "middle", "2 words")), names(agent()));
    assert.ok(isValidNodeName("collect_details"));
    assert.ok(!isValidNodeName("collect details"));
    assert.ok(!isValidNodeName("9lives"));
  });
});

describe("deleting a node", () => {
  it("takes inbound edges with it", () => {
    // Leaving them behind would make the graph uncompilable, and the user asked
    // to remove a node, not to break their agent.
    const pruned = deleteNode(agent(), "middle");
    assert.deepEqual(names(pruned), ["greeting", "farewell"]);
    assert.ok(!targets(pruned).includes("middle"));
  });

  it("moves the entry node when the entry is deleted", () => {
    const pruned = deleteNode(agent(), "greeting");
    assert.ok(names(pruned).includes(pruned.initial_node));
  });

  it("refuses to remove the last node", () => {
    const single = { ...agent(), nodes: [agent().nodes[0]] };
    assert.deepEqual(names(deleteNode(single, "greeting")), ["greeting"]);
  });
});

describe("adding", () => {
  it("never reuses a name", () => {
    let built = agent();
    for (let i = 0; i < 3; i += 1) built = addNode(built);
    assert.equal(new Set(names(built)).size, names(built).length);
  });

  it("announces the name it will use, so the editor can select it", () => {
    const next = nextNodeName(agent());
    assert.ok(names(addNode(agent())).includes(next));
  });

  it("adds nothing that would be written to the agent's JSON", () => {
    // An earlier version smuggled the selection back inside the agent, which
    // would have been saved to disk.
    const built = addNode(agent());
    assert.deepEqual(Object.keys(built).sort(), Object.keys(agent()).sort());
  });

  it("appends and removes edges on the right node", () => {
    const wired = addEdge(agent(), "farewell", {
      function: "again",
      description: "d",
      target: "greeting",
    });
    assert.equal(wired.nodes.find((n) => n.name === "farewell").edges.length, 1);
    assert.equal(deleteEdge(wired, "farewell", 0).nodes.find((n) => n.name === "farewell").edges.length, 0);
  });

  it("edits an edge in place", () => {
    const edited = updateEdge(agent(), "greeting", 0, { target: "farewell" });
    assert.equal(edited.nodes[0].edges[0].target, "farewell");
    assert.equal(edited.nodes[0].edges[0].function, "go");
  });

  it("moves the entry node", () => {
    assert.equal(setEntryNode(agent(), "middle").initial_node, "middle");
  });
});

describe("warnings", () => {
  it("says nothing about a well-formed graph", () => {
    assert.deepEqual(warnings(agent()), []);
  });

  it("flags a node with no way out", () => {
    const stranded = addNode(agent());
    const found = warnings(stranded).filter((w) => w.node === nextNodeName(agent()));
    assert.ok(found.some((w) => w.text.includes("no way out")));
  });

  it("flags a node nothing can reach", () => {
    const orphan = addNode(agent());
    assert.ok(warnings(orphan).some((w) => w.text.includes("unreachable")));
  });

  it("flags a graph that never hangs up", () => {
    const endless = {
      ...agent(),
      nodes: agent().nodes.map((n) => ({ ...n, end: false })),
    };
    assert.ok(warnings(endless).some((w) => w.text.includes("never hangs up")));
  });
});
