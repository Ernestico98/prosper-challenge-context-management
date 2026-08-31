import React, { useEffect, useState } from "react";
import {
  catalogAppointmentTypes,
  catalogLocations,
  catalogProviders,
  catalogSummary,
} from "./api.js";

const VIEWS = ["providers", "locations", "appointment types"];

export default function CatalogBrowser() {
  const [summary, setSummary] = useState(null);
  const [view, setView] = useState("providers");
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    catalogSummary().then(setSummary).catch(() => {});
  }, []);

  useEffect(() => {
    const load = {
      providers: () => catalogProviders().then((d) => d.providers),
      locations: () => catalogLocations().then((d) => d.locations),
      "appointment types": () =>
        catalogAppointmentTypes().then((d) => d.appointment_types),
    }[view];
    load().then(setRows).catch(() => setRows([]));
  }, [view]);

  const needle = filter.trim().toLowerCase();
  const visible = needle
    ? rows.filter((row) => JSON.stringify(row).toLowerCase().includes(needle))
    : rows;

  return (
    <div className="catalog">
      {summary && (
        <div className="summary">
          <Stat value={summary.locations} label="locations" />
          <Stat value={summary.providers} label="providers" />
          <Stat value={summary.appointment_types} label="appointment types" />
          <Stat
            value={`${summary.bookable_specialties}/${summary.specialties}`}
            label="specialties staffed"
            warn={summary.bookable_specialties < summary.specialties}
          />
          {summary.unstaffed_appointment_types.length > 0 && (
            <div className="notice">
              <strong>{summary.unstaffed_appointment_types.length} advertised but unbookable:</strong>{" "}
              {summary.unstaffed_appointment_types.join(", ")}. No provider offers them, so
              the agent never routes a caller there.
            </div>
          )}
        </div>
      )}

      <div className="toolbar">
        {VIEWS.map((v) => (
          <button key={v} className={view === v ? "tab active" : "tab"} onClick={() => setView(v)}>
            {v}
          </button>
        ))}
        <input
          className="filter"
          placeholder="Filter…"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <span className="count">
          {visible.length} of {rows.length}
        </span>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {Object.keys(visible[0] || rows[0] || {}).map((key) => (
                <th key={key}>{key.replace(/_/g, " ")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((row) => (
              <tr key={row.id}>
                {Object.entries(row).map(([key, value]) => (
                  <td key={key} className={cellClass(key, value)}>
                    {render(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Stat({ value, label, warn }) {
  return (
    <div className={warn ? "stat warn" : "stat"}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function cellClass(key, value) {
  if (key === "duplicate_name" && value) return "flag";
  if (key === "bookable_combinations" && value === 0) return "flag";
  return "";
}

function render(value) {
  if (Array.isArray(value)) return value.join(", ") || "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}
