import { useMemo, useRef, useState } from "react";
import { Bar, BarChart, Cell, ReferenceLine } from "recharts";
import { api, useApi } from "../api.js";
import { DECISION, RISKY, STRONG, demoBatch, download, num, parseCsv, pct, toCsv } from "../lib.js";
import { C, Card, ChartCard, ErrorBox, GlassTooltip, Grid, Loading, Stat, Table, X, Y } from "../ui.jsx";

const BIN = 20;

function histogram(scores) {
  const bins = [];
  for (let start = 300; start < 850; start += BIN) {
    bins.push({ start, label: start, count: scores.filter((s) => s >= start && s < start + BIN).length });
  }
  return bins;
}

export default function Batch() {
  const schema = useApi(api.schema, []);
  const model = useApi(api.model, []);
  const [rows, setRows] = useState(null);
  const [source, setSource] = useState("");
  const [response, setResponse] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef(null);

  if (schema.error || model.error) return <ErrorBox error={schema.error ?? model.error} onRetry={schema.reload} />;
  if (!schema.data || !model.data) return <Loading what="the applicant schema" />;

  const fields = Object.keys(schema.data.profile.properties);
  const required = schema.data.profile.required ?? [];

  const load = (loaded, name) => {
    setRows(loaded);
    setSource(name);
    setResponse(null);
    setError(null);
  };

  const onFile = async (file) => {
    if (!file) return;
    try {
      load(parseCsv(await file.text()), file.name);
    } catch (err) {
      setError(new Error(`Could not read ${file.name}: ${err.message}`));
    }
  };

  const unknown = rows ? Object.keys(rows[0] ?? {}).filter((c) => c !== "applicant_id" && !fields.includes(c)) : [];
  const missing = rows ? required.filter((f) => !(f in (rows[0] ?? {}))) : [];
  const clean = rows?.map((row) => Object.fromEntries(Object.entries(row).filter(([k]) => !unknown.includes(k))));

  const score = async () => {
    setBusy(true);
    setError(null);
    try {
      setResponse(await api.scoreBatch(clean));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  const downloadTemplate = () => {
    const examples = [
      { applicant_id: "EXAMPLE-TYPICAL", ...schema.data.defaults },
      { applicant_id: "EXAMPLE-STRONG", ...STRONG },
      { applicant_id: "EXAMPLE-RISKY", ...RISKY },
    ];
    download("credscore_template.csv", toCsv(examples, ["applicant_id", ...fields]), "text/csv");
  };

  return (
    <div className="page">
      <header className="stack" style={{ gap: 6 }}>
        <h1>Batch scoring</h1>
        <p className="muted">Score a whole file of applicants in one go. Invalid rows are flagged individually, not fatal.</p>
      </header>

      <div className="row">
        <button className="btn" onClick={downloadTemplate}>⤓ Download CSV template</button>
        <button className="btn" onClick={() => load(demoBatch(schema.data.defaults), "demo applicants")}>
          ✨ Load 250 demo applicants
        </button>
      </div>

      <div
        className={`dropzone ${dragOver ? "over" : ""}`}
        onClick={() => fileInput.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          onFile(e.dataTransfer.files?.[0]);
        }}
      >
        <div style={{ fontSize: "1.6rem" }}>⇪</div>
        <b className="ink">Drop a CSV here, or click to choose one</b>
        <p className="tiny muted">One row per applicant, columns as in the template. Optional columns may be left out or empty.</p>
        <input ref={fileInput} type="file" accept=".csv,text/csv" hidden onChange={(e) => onFile(e.target.files?.[0])} />
      </div>

      {error && <ErrorBox error={error} />}

      {rows && (
        <Card hover={false} className="stack">
          <div className="spread">
            <div>
              <b className="ink">{num(rows.length)} applicants</b> <span className="muted">loaded from {source}</span>
            </div>
            <button className="btn btn-primary" onClick={score} disabled={busy || missing.length > 0}>
              {busy ? <span className="spinner" /> : "⚡"} {busy ? "Scoring…" : `Score ${num(rows.length)} applicants`}
            </button>
          </div>
          {unknown.length > 0 && <div className="status warn">Ignoring unrecognised columns: {unknown.join(", ")}</div>}
          {missing.length > 0 && <div className="status bad">Missing required columns: {missing.join(", ")}</div>}
        </Card>
      )}

      {response && <Results response={response} rows={clean} bands={model.data.scorecard.bands} />}
    </div>
  );
}

function Results({ response, rows, bands }) {
  const { summary, results } = response;
  const table = useMemo(
    () =>
      results.map((item) => ({
        id: item.index,
        applicant_id: item.applicant_id ?? `row ${item.index + 1}`,
        credit_score: item.result?.credit_score ?? null,
        pd: item.result?.probability_of_default ?? null,
        band: item.result?.risk_band?.code ?? "",
        decision: item.result?.decision ?? "",
        reasons: (item.result?.reasons ?? []).join(" · "),
        error: item.error ?? "",
      })),
    [results],
  );
  const bins = useMemo(() => histogram(table.map((r) => r.credit_score).filter(Boolean)), [table]);
  const decisions = ["APPROVE", "REVIEW", "DECLINE"].map((key) => ({
    key,
    name: `${DECISION[key].icon} ${DECISION[key].label}`,
    count: summary.decisions[key],
  }));
  const scored = summary.scored || 1;

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="grid grid-auto">
        <Stat label="Scored" value={num(summary.scored)} sub={`${num(summary.failed)} rows failed validation`} />
        {decisions.map((d) => (
          <Stat key={d.key} label={DECISION[d.key].label} value={pct(d.count / scored, 0)} sub={`${num(d.count)} applicants`} />
        ))}
        <Stat label="Average PD" value={pct(summary.mean_probability_of_default)} />
        <Stat label="Average score" value={num(summary.mean_credit_score)} />
      </div>

      <div className="split">
        <ChartCard title="Credit score distribution" height={320} note="Vertical rules mark the risk-band boundaries.">
          <BarChart data={bins} margin={{ left: 4, right: 16, top: 8, bottom: 4 }}>
            <Grid />
            <X dataKey="label" tickFormatter={(v) => v} interval={4} />
            <Y allowDecimals={false} domain={[0, "auto"]} />
            <GlassTooltip
              rows={(row) => [
                { label: "Score range", value: `${row.start}–${row.start + BIN}` },
                { label: "Applicants", value: num(row.count), color: C.blue },
              ]}
            />
            {bands
              .filter((b) => b.min_score > 300)
              .map((b) => (
                <ReferenceLine key={b.code} x={Math.round(b.min_score / BIN) * BIN} stroke={C.axis}
                  label={{ value: b.code, fill: C.ink2, fontSize: 11, position: "top" }} />
              ))}
            <Bar dataKey="count" fill={C.blue} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ChartCard>

        <ChartCard title="Decisions" height={320}>
          <BarChart data={decisions} layout="vertical" margin={{ left: 4, right: 40, top: 8, bottom: 4 }}>
            <Grid />
            <X type="number" domain={[0, "auto"]} />
            <Y type="category" dataKey="name" width={130} tick={{ fill: C.ink2, fontSize: 12 }} />
            <GlassTooltip rows={(row) => [{ label: "Applicants", value: num(row.count) }]} />
            <Bar dataKey="count" barSize={26} radius={4} isAnimationActive={false}
                 label={{ position: "right", fill: C.ink2, fontSize: 12, formatter: (v) => num(v) }}>
              {decisions.map((d) => (
                <Cell key={d.key} fill={DECISION[d.key].color} />
              ))}
            </Bar>
          </BarChart>
        </ChartCard>
      </div>

      <Card hover={false} className="stack pad-0">
        <div className="card spread" style={{ paddingBottom: 0 }}>
          <h3>Results</h3>
          <button
            className="btn btn-sm"
            onClick={() =>
              download(
                "credscore_results.csv",
                toCsv(rows.map((row, i) => ({ ...row, ...stripId(table[i]) }))),
                "text/csv",
              )
            }
          >
            ⤓ Download results
          </button>
        </div>
        <Table
          maxHeight={460}
          columns={[
            { key: "applicant_id", label: "Applicant" },
            { key: "credit_score", label: "Score", align: "right", render: (v) => v ?? "—" },
            { key: "pd", label: "PD", align: "right", render: (v) => pct(v, 2) },
            { key: "band", label: "Band" },
            { key: "decision", label: "Decision", render: (v) => (v ? DECISION[v].label : "—") },
            { key: "reasons", label: "Main risk factors" },
            { key: "error", label: "Error" },
          ]}
          rows={table}
        />
      </Card>
    </div>
  );
}

const stripId = ({ id, ...rest }) => rest;
