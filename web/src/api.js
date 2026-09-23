import { useCallback, useEffect, useState } from "react";

// Same-origin in production (FastAPI serves the built assets); proxied by Vite in dev.
const BASE = `${(import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "")}/api/v1`;
const BATCH_CHUNK = 2000;

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function detailToMessage(detail, fallback) {
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) =>
        typeof item === "string"
          ? item
          : `${(item.loc ?? []).filter((p) => p !== "body").join(".")}: ${item.msg}`.replace(/^: /, ""),
      )
      .join("; ");
  }
  return fallback;
}

async function request(path, { method = "GET", body, params } = {}) {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  Object.entries(params ?? {}).forEach(([k, v]) => url.searchParams.set(k, v));
  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError("Cannot reach the CredScore API. Start it with `uvicorn backend.app:app --port 8000`.");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(detailToMessage(payload?.detail, `Request failed (${response.status})`), response.status);
  }
  return payload;
}

export const api = {
  health: () => request("/health"),
  model: () => request("/model"),
  performance: () => request("/model/performance"),
  schema: () => request("/schema"),
  insights: () => request("/insights"),
  score: (profile, topK = 12) => request("/score", { method: "POST", body: profile, params: { top_k: topK } }),

  /** Scores any number of profiles, chunked to respect the API's batch limit. */
  async scoreBatch(rows, reasons = 3) {
    const results = [];
    const summary = { submitted: 0, scored: 0, failed: 0, decisions: { APPROVE: 0, REVIEW: 0, DECLINE: 0 } };
    let pdTotal = 0;
    let scoreTotal = 0;
    for (let start = 0; start < rows.length; start += BATCH_CHUNK) {
      const chunk = rows.slice(start, start + BATCH_CHUNK);
      const response = await request("/score/batch", {
        method: "POST",
        body: { applicants: chunk },
        params: { reasons },
      });
      response.results.forEach((item) => results.push({ ...item, index: item.index + start }));
      const s = response.summary;
      summary.submitted += s.submitted;
      summary.scored += s.scored;
      summary.failed += s.failed;
      Object.entries(s.decisions).forEach(([key, count]) => (summary.decisions[key] += count));
      if (s.scored) {
        pdTotal += s.mean_probability_of_default * s.scored;
        scoreTotal += s.mean_credit_score * s.scored;
      }
    }
    summary.mean_probability_of_default = summary.scored ? pdTotal / summary.scored : null;
    summary.mean_credit_score = summary.scored ? scoreTotal / summary.scored : null;
    return { summary, results };
  },
};

/** Loads data once per mount; returns { data, error, loading, reload }. */
export function useApi(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const run = useCallback(() => {
    let active = true;
    setState((s) => ({ ...s, loading: true }));
    loader()
      .then((data) => active && setState({ data, error: null, loading: false }))
      .catch((error) => active && setState({ data: null, error, loading: false }));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(run, [run]);
  return { ...state, reload: run };
}
