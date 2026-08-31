// Every call goes to the same origin: the Pipecat runner serves the UI, the
// builder API and the WebRTC client, so there is nothing to configure.

async function request(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${path} failed (${response.status})`);
  return body;
}

export const listAgents = () => request("/api/agents");
export const getAgent = (id) => request(`/api/agents/${id}`);
export const saveAgent = (id, agent) =>
  request(`/api/agents/${id}`, { method: "PUT", body: JSON.stringify(agent) });
export const listTools = () => request("/api/tools");
export const catalogSummary = () => request("/api/catalog/summary");
export const catalogLocations = () => request("/api/catalog/locations");
export const catalogProviders = () => request("/api/catalog/providers");
export const catalogAppointmentTypes = () => request("/api/catalog/appointment-types");
