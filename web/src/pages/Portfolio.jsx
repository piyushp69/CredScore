import { useMemo, useState } from "react";
import { Bar, BarChart, LabelList, ReferenceLine } from "recharts";
import { api, useApi } from "../api.js";
import { num, pct } from "../lib.js";
import { C, Card, ChartCard, Disclosure, ErrorBox, GlassTooltip, Grid, Loading, Stat, Table, X, Y } from "../ui.jsx";

export default function Portfolio() {
  const { data, error, loading, reload } = useApi(api.insights, []);
  const [key, setKey] = useState(null);

  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (loading || !data) return <Loading what="portfolio insights" />;

  const { kpis, segments } = data;
  const segment = segments.find((s) => s.key === key) ?? segments[0];
  const baseRate = kpis.default_rate;

  return (
    <div className="page">
      <header className="stack" style={{ gap: 6 }}>
        <h1>Portfolio insights</h1>
        <p className="muted">
          Observed outcomes for the {num(kpis.applicants)} labelled loan applications the model learned from.
        </p>
      </header>

      <div className="grid grid-auto">
        <Stat label="Applications" value={num(kpis.applicants)} />
        <Stat label="Default rate" value={pct(baseRate, 2)} sub={`${num(kpis.defaults)} with payment difficulties`} />
        <Stat label="Median income" value={num(kpis.median_income)} />
        <Stat label="Median credit" value={num(kpis.median_credit)} />
        <Stat label="Median age" value={`${kpis.median_age} yrs`} />
        <Stat label="Bureau history" value={pct(kpis.share_with_bureau_history, 0)} sub="Share with a bureau record" />
      </div>

      <div className="row">
        <label className="field" style={{ maxWidth: 320, width: "100%" }}>
          <span>Break down by</span>
          <select className="input" value={segment.key} onChange={(e) => setKey(e.target.value)}>
            {segments.map((s) => (
              <option key={s.key} value={s.key}>
                {s.title}
              </option>
            ))}
          </select>
        </label>
        <p className="small muted grow">{segment.description}</p>
      </div>

      <div className="split">
        <SegmentChart segment={segment} baseRate={baseRate} />
        <ChartCard title="Applicant mix" height={Math.max(320, segment.rows.length * 32 + 60)}>
          <BarChart data={segment.rows} layout="vertical" margin={{ left: 4, right: 20, top: 8, bottom: 4 }}>
            <Grid />
            <X type="number" tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
            <Y type="category" dataKey="segment" width={140} tick={{ fill: C.ink2, fontSize: 11 }} />
            <GlassTooltip
              rows={(row) => [
                { label: "Applicants", value: num(row.applicants) },
                { label: "Share", value: pct(row.share) },
              ]}
            />
            <Bar dataKey="share" fill={C.neutral} barSize={16} radius={4} isAnimationActive={false} />
          </BarChart>
        </ChartCard>
      </div>

      <Disclosure label="View as table">
        <Table
          columns={[
            { key: "segment", label: segment.title },
            { key: "applicants", label: "Applicants", align: "right", render: (v) => num(v) },
            { key: "share", label: "Share", align: "right", render: (v) => pct(v) },
            { key: "default_rate", label: "Default rate", align: "right", render: (v) => pct(v, 2) },
          ]}
          rows={segment.rows}
        />
      </Disclosure>

      <Extremes segments={segments} baseRate={baseRate} />
    </div>
  );
}

function SegmentChart({ segment, baseRate }) {
  const rows = segment.ordered ? segment.rows : [...segment.rows].sort((a, b) => b.default_rate - a.default_rate);
  const max = Math.max(...rows.map((r) => r.default_rate));
  const min = Math.min(...rows.map((r) => r.default_rate));
  return (
    <ChartCard
      title={`Default rate by ${segment.title.toLowerCase()}`}
      height={Math.max(320, rows.length * 32 + 60)}
      note={`Dotted line: portfolio average of ${pct(baseRate)}. Extremes are labelled; hover for the rest.`}
    >
      <BarChart data={rows} layout="vertical" margin={{ left: 4, right: 52, top: 8, bottom: 4 }}>
        <Grid />
        <X type="number" tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
        <Y type="category" dataKey="segment" width={140} tick={{ fill: C.ink2, fontSize: 11 }} />
        <GlassTooltip
          rows={(row) => [
            { label: "Default rate", value: pct(row.default_rate, 2), color: C.blue },
            { label: "Applicants", value: num(row.applicants) },
            { label: "Share", value: pct(row.share) },
          ]}
        />
        <ReferenceLine x={baseRate} stroke={C.ink2} strokeDasharray="4 4" />
        <Bar dataKey="default_rate" fill={C.blue} barSize={16} radius={4} isAnimationActive={false}>
          <LabelList
            dataKey="default_rate"
            position="right"
            fill={C.ink2}
            fontSize={11}
            formatter={(v) => (v === max || v === min ? pct(v) : "")}
          />
        </Bar>
      </BarChart>
    </ChartCard>
  );
}

function Extremes({ segments, baseRate }) {
  const rows = useMemo(
    () =>
      segments.flatMap((s) =>
        s.rows
          .filter((r) => r.share >= 0.02)
          .map((r) => ({
            id: `${s.key}-${r.segment}`,
            segment: `${s.title}: ${r.segment}`,
            share: r.share,
            default_rate: r.default_rate,
            ratio: r.default_rate / baseRate,
          })),
      ),
    [segments, baseRate],
  );
  const columns = [
    { key: "segment", label: "Segment" },
    { key: "share", label: "Share", align: "right", render: (v) => pct(v) },
    { key: "default_rate", label: "Default rate", align: "right", render: (v) => pct(v, 2) },
    { key: "ratio", label: "vs portfolio", align: "right", render: (v) => `${v.toFixed(1)}×` },
  ];
  const sorted = [...rows].sort((a, b) => b.default_rate - a.default_rate);
  return (
    <div className="stack">
      <div>
        <h2>Where the risk concentrates</h2>
        <p className="muted small">Segments holding at least 2% of applicants, across every breakdown above.</p>
      </div>
      <div className="grid grid-2">
        <Card hover={false} className="stack pad-0">
          <div className="card" style={{ paddingBottom: 0 }}>
            <h3>Highest default rates</h3>
          </div>
          <Table columns={columns} rows={sorted.slice(0, 8)} />
        </Card>
        <Card hover={false} className="stack pad-0">
          <div className="card" style={{ paddingBottom: 0 }}>
            <h3>Lowest default rates</h3>
          </div>
          <Table columns={columns} rows={sorted.slice(-8).reverse()} />
        </Card>
      </div>
    </div>
  );
}
