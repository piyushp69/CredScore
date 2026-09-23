import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, Cell, Line, LineChart, ReferenceLine } from "recharts";
import { api, useApi } from "../api.js";
import { BAND_COLOR, DECISION, RISKY, STRONG, dec, download, num, pct } from "../lib.js";
import {
  C, Card, ChartCard, Chip, Disclosure, ErrorBox, GlassTooltip, Grid, Loading, NumberField, ScoreScale,
  SelectField, Stat, Table, Tabs, ToggleField, X, Y,
} from "../ui.jsx";

// [key, label, kind, props]. Kinds: num | int | optional | optionalInt | select | optionalSelect | toggle
const SECTIONS = {
  Applicant: [
    ["age_years", "Age (years)", "num", { min: 18, max: 100 }],
    ["family_status", "Family status", "select"],
    ["education", "Education", "select"],
    ["children", "Children", "int", { min: 0, max: 20 }],
    ["family_members", "Household size", "int", { min: 1, max: 25 }],
    ["housing_type", "Housing", "select"],
    ["owns_realty", "Owns real estate", "toggle"],
    ["owns_car", "Owns a car", "toggle"],
    ["car_age_years", "Car age (years)", "optional", { min: 0, max: 100 }],
    ["region_rating", "Region rating (1 = best)", "int", { min: 1, max: 3 }],
  ],
  "Employment & income": [
    ["income_type", "Income type", "select"],
    ["occupation", "Occupation", "optionalSelect"],
    ["organization_type", "Employer type", "optionalSelect"],
    ["annual_income", "Annual income", "num", { min: 1, step: 5000 }],
    ["years_employed", "Years in current job", "optional", { min: 0, max: 60, step: 0.5,
      hint: "Leave empty for pensioners or unemployed applicants." }],
  ],
  "Loan request": [
    ["contract_type", "Contract type", "select"],
    ["credit_amount", "Credit amount", "num", { min: 1, step: 10000 }],
    ["annuity_amount", "Annuity (periodic payment)", "num", { min: 1, step: 500 }],
    ["goods_price", "Goods price", "optional", { min: 1, step: 10000,
      hint: "Price of the goods financed. Leave empty to use the credit amount." }],
  ],
  "External scores": [
    ["ext_source_1", "External score 1", "optional", { min: 0, max: 1, step: 0.01,
      hint: "Normalized external bureau scores (0–1, higher is safer). Leave empty if unavailable." }],
    ["ext_source_2", "External score 2", "optional", { min: 0, max: 1, step: 0.01 }],
    ["ext_source_3", "External score 3", "optional", { min: 0, max: 1, step: 0.01 }],
  ],
  "Credit bureau": [
    ["bureau_loans", "Loans on record (0 = none)", "int", { min: 0, max: 500 }],
    ["bureau_active_loans", "Active loans", "int", { min: 0, max: 500 }],
    ["bureau_total_credit", "Total credit", "num", { min: 0, step: 10000 }],
    ["bureau_total_debt", "Outstanding debt", "num", { min: 0, step: 10000 }],
    ["bureau_overdue_amount", "Largest overdue amount", "num", { min: 0, step: 1000 }],
    ["bureau_days_overdue", "Days currently overdue", "int", { min: 0, max: 3000 }],
    ["bureau_years_since_last_loan", "Years since latest loan", "optional", { min: 0, max: 50, step: 0.5 }],
    ["bureau_enquiries_last_year", "Bureau enquiries (last year)", "optionalInt", { min: 0, max: 100 }],
  ],
  "Home Credit history": [
    ["prev_applications", "Previous applications (0 = new client)", "int", { min: 0, max: 200 }],
    ["prev_approved", "Approved", "int", { min: 0, max: 200 }],
    ["prev_refused", "Refused", "int", { min: 0, max: 200 }],
    ["installments_paid", "Installments on record", "int", { min: 0, max: 2000 }],
    ["late_payment_share", "Share paid late", "num", { min: 0, max: 1, step: 0.01 }],
    ["avg_days_past_due", "Average days past due", "num", { min: 0, max: 3000 }],
    ["max_days_past_due", "Worst days past due", "num", { min: 0, max: 3000 }],
    ["payment_ratio", "Share of each installment paid", "num", { min: 0, max: 5, step: 0.05 }],
  ],
};

const WHAT_IF = {
  ext_source_2: { label: "External score 2", range: () => [0.02, 0.85] },
  ext_source_3: { label: "External score 3", range: () => [0.02, 0.9] },
  annual_income: { label: "Annual income", range: (v) => [Math.max(20000, v * 0.3), v * 3] },
  credit_amount: { label: "Credit amount", range: (v) => [Math.max(20000, v * 0.2), v * 2.5] },
  annuity_amount: { label: "Annuity", range: (v) => [Math.max(1000, v * 0.4), v * 2] },
  age_years: { label: "Age (years)", range: () => [20, 70] },
  years_employed: { label: "Years in current job", range: () => [0, 30] },
  late_payment_share: { label: "Share of installments paid late", range: () => [0, 0.6] },
};

export default function Underwriting() {
  const schema = useApi(api.schema, []);
  const model = useApi(api.model, []);
  const [profile, setProfile] = useState(null);
  const [section, setSection] = useState(Object.keys(SECTIONS)[0]);
  const [scoring, setScoring] = useState(false);
  const [scored, setScored] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (schema.data && !profile) setProfile({ ...schema.data.defaults });
  }, [schema.data, profile]);

  if (schema.error || model.error) {
    const retry = () => {
      if (schema.error) schema.reload();
      if (model.error) model.reload();
    };
    return <ErrorBox error={schema.error ?? model.error} onRetry={retry} />;
  }
  if (!schema.data || !model.data || !profile) return <Loading what="the applicant form" />;

  const { options, defaults } = schema.data;
  const presets = { "Typical applicant": defaults, "Strong applicant": STRONG, "Risky applicant": RISKY };
  const set = (key, value) => setProfile((p) => ({ ...p, [key]: value }));

  const submit = async (event) => {
    event.preventDefault();
    setScoring(true);
    setError(null);
    try {
      const result = await api.score(profile);
      setScored({ profile, result });
      setHistory((h) => [
        {
          id: h.length ? h[0].id + 1 : 0,
          time: new Date().toLocaleTimeString(),
          score: result.credit_score,
          band: result.risk_band.code,
          pd: result.probability_of_default,
          decision: result.decision,
          income: profile.annual_income,
          credit: profile.credit_amount,
        },
        ...h,
      ]);
    } catch (err) {
      setError(err);
    } finally {
      setScoring(false);
    }
  };

  return (
    <div className="page">
      <header className="stack" style={{ gap: 6 }}>
        <h1>Applicant underwriting</h1>
        <p className="muted">Score a loan applicant, see why the model decided what it did, and test what would change it.</p>
      </header>

      <div className="row">
        <span className="small muted">Start from an example</span>
        {Object.entries(presets).map(([name, preset]) => (
          <button key={name} type="button" className="btn btn-sm" onClick={() => setProfile({ ...preset })}>
            {name}
          </button>
        ))}
      </div>

      <form onSubmit={submit} className="glass card stack">
        <Tabs tabs={Object.keys(SECTIONS)} active={section} onChange={setSection} />
        <div className="grid grid-3">
          {SECTIONS[section].map(([key, label, kind, props = {}]) => {
            const { hint, ...inputProps } = props;
            const common = { label, hint, value: profile[key], onChange: (v) => set(key, v) };
            if (kind === "toggle") return <ToggleField key={key} {...common} />;
            if (kind === "select" || kind === "optionalSelect") {
              return (
                <SelectField key={key} {...common} options={options[key] ?? []} optional={kind === "optionalSelect"} />
              );
            }
            const optional = kind === "optional" || kind === "optionalInt";
            const step = kind === "int" || kind === "optionalInt" ? 1 : inputProps.step ?? "any";
            return <NumberField key={key} {...common} {...inputProps} step={step} optional={optional} />;
          })}
        </div>
        <button className="btn btn-primary btn-wide" type="submit" disabled={scoring}>
          {scoring ? <span className="spinner" /> : "⚡"} {scoring ? "Scoring…" : "Score applicant"}
        </button>
      </form>

      {error && <ErrorBox error={error} />}

      {scored ? (
        <Result scored={scored} model={model.data} history={history} />
      ) : (
        <Card hover={false}>
          <p className="muted">Fill in the applicant details (or start from an example) and press <b>Score applicant</b>.</p>
        </Card>
      )}
    </div>
  );
}

function Result({ scored, model, history }) {
  const { profile, result } = scored;
  const band = result.risk_band;
  const decision = DECISION[result.decision];
  const baseRate = model.training?.base_rate;

  const explanations = [...result.explanations]
    .sort((a, b) => Math.abs(a.contribution) - Math.abs(b.contribution))
    .map((e) => ({ ...e, name: `${e.label} = ${e.display_value}` }));

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="grid grid-3">
        <Card>
          <span className="label">Credit score</span>
          <div className="hero mono-num">{result.credit_score}</div>
          <div style={{ marginTop: 10 }}>
            <Chip color={BAND_COLOR[band.status]}>
              Band {band.code} · {band.label}
            </Chip>
          </div>
        </Card>
        <Card>
          <span className="label">Probability of default</span>
          <div className="hero mono-num">{pct(result.probability_of_default)}</div>
          {baseRate && (
            <p className="tiny muted" style={{ marginTop: 8 }}>
              {(result.probability_of_default / baseRate).toFixed(1)}× the portfolio average of {pct(baseRate)}
            </p>
          )}
        </Card>
        <Card>
          <span className="label">Recommended decision</span>
          <div className="row" style={{ gap: 12, marginTop: 6 }}>
            <span className="badge-icon" style={{ background: decision.color }}>{decision.icon}</span>
            <span style={{ color: "var(--ink)", fontSize: "1.5rem", fontWeight: 640 }}>{decision.label}</span>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>{result.decision_reason}</p>
        </Card>
      </div>

      <Card hover={false}>
        <ScoreScale score={result.credit_score} bands={model.scorecard.bands} />
      </Card>

      <div className="grid grid-4">
        {Object.entries(result.key_metrics).map(([key, metric]) => (
          <Stat key={key} label={metric.label} value={metric.display_value} />
        ))}
      </div>

      <div className="split">
        <ChartCard
          title="What drove this score"
          height={Math.max(280, explanations.length * 34 + 40)}
          legend={[
            { label: "Raises risk", color: C.riskUp },
            { label: "Lowers risk", color: C.riskDown },
          ]}
          note="SHAP contributions in log-odds. Bars to the right raise the default risk, to the left lower it."
        >
          <BarChart data={explanations} layout="vertical" margin={{ left: 4, right: 20, top: 4, bottom: 4 }}>
            <Grid />
            <X type="number" tickFormatter={(v) => v.toFixed(1)} />
            <Y type="category" dataKey="name" width={250} tick={{ fill: C.ink2, fontSize: 11 }} />
            <GlassTooltip
              rows={(row) => [
                { label: "Impact", value: `${row.contribution > 0 ? "+" : ""}${row.contribution.toFixed(3)}`,
                  color: row.contribution > 0 ? C.riskUp : C.riskDown },
                { label: "Source", value: row.source },
              ]}
            />
            <ReferenceLine x={0} stroke={C.axis} />
            <Bar dataKey="contribution" barSize={15} radius={4} isAnimationActive={false}>
              {explanations.map((e) => (
                <Cell key={e.feature} fill={e.contribution > 0 ? C.riskUp : C.riskDown} />
              ))}
            </Bar>
          </BarChart>
        </ChartCard>

        <div className="stack">
          <Card hover={false} className="stack">
            <h3>Main risk factors</h3>
            {result.reasons.length ? (
              <ul className="reason-list">
                {result.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            ) : (
              <p className="muted small">No factor materially raises this applicant's risk.</p>
            )}
            {result.explanations.some((e) => e.source === "default") && (
              <p className="tiny muted">
                Not provided, so a typical applicant's value was used:{" "}
                {result.explanations.filter((e) => e.source === "default").map((e) => e.label).join(", ")}.
              </p>
            )}
            <button
              type="button"
              className="btn btn-sm"
              onClick={() =>
                download(
                  `credscore_decision_${Date.now()}.json`,
                  JSON.stringify({ generated_at: new Date().toISOString(), profile, result }, null, 2),
                  "application/json",
                )
              }
            >
              ⤓ Download decision report
            </button>
          </Card>
          <Card hover={false} className="stack">
            <h3>Decision policy</h3>
            <p className="small muted">
              Approve below {pct(model.policy.approve_below)} · decline at {pct(model.policy.decline_at)} or above.
              Cut-offs were set on the validation set to approve about 70% and decline the riskiest 10%.
            </p>
          </Card>
        </div>
      </div>

      <WhatIf profile={profile} result={result} policy={model.policy} />

      {history.length > 1 && (
        <Card hover={false} className="stack pad-0">
          <div className="card" style={{ paddingBottom: 0 }}>
            <h3>This session</h3>
          </div>
          <Table
            maxHeight={280}
            columns={[
              { key: "time", label: "Time" },
              { key: "score", label: "Score", align: "right" },
              { key: "band", label: "Band" },
              { key: "pd", label: "PD", align: "right", render: (v) => pct(v) },
              { key: "decision", label: "Decision", render: (v) => DECISION[v].label },
              { key: "income", label: "Income", align: "right", render: (v) => num(v) },
              { key: "credit", label: "Credit", align: "right", render: (v) => num(v) },
            ]}
            rows={history}
          />
        </Card>
      )}
    </div>
  );
}

function WhatIf({ profile, result, policy }) {
  const choices = Object.keys(WHAT_IF).filter((f) => f !== "late_payment_share" || profile.installments_paid);
  const [selected, setField] = useState(choices[0]);
  // A newly scored profile may no longer offer the selected input.
  const field = choices.includes(selected) ? selected : choices[0];
  const [curve, setCurve] = useState(null);
  const [error, setError] = useState(null);

  const values = useMemo(() => {
    const current = Number(profile[field] ?? 0);
    const [lo, hi] = WHAT_IF[field].range(current);
    return Array.from({ length: 25 }, (_, i) => Math.round((lo + ((hi - lo) * i) / 24) * 1000) / 1000);
  }, [field, profile]);

  useEffect(() => {
    let active = true;
    setCurve(null);
    setError(null);
    api
      .scoreBatch(values.map((v) => ({ ...profile, [field]: v })), 0)
      .then((response) => {
        if (!active) return;
        setCurve(values.map((value, i) => ({ value, pd: response.results[i].result?.probability_of_default ?? null })));
      })
      .catch((err) => active && setError(err));
    return () => {
      active = false;
    };
  }, [field, values, profile]);

  const label = WHAT_IF[field].label;
  const formatX = (v) => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000)}k` : String(v));

  return (
    <div className="stack">
      <div className="spread">
        <div>
          <h2>What-if analysis</h2>
          <p className="muted small">Vary one input, keep everything else fixed, and watch the probability of default respond.</p>
        </div>
        <label className="field" style={{ maxWidth: 260, width: "100%" }}>
          <span>Input to vary</span>
          <select className="input" value={field} onChange={(e) => setField(e.target.value)}>
            {choices.map((key) => (
              <option key={key} value={key}>
                {WHAT_IF[key].label}
              </option>
            ))}
          </select>
        </label>
      </div>
      {error ? (
        <ErrorBox error={error} />
      ) : !curve ? (
        <Card hover={false}>
          <Loading what="the what-if curve" />
        </Card>
      ) : (
        <ChartCard
          title={`Probability of default vs ${label.toLowerCase()}`}
          height={340}
          legend={[
            { label: "Probability of default", color: C.blue },
            { label: "Policy cut-offs", color: C.muted, dash: true },
          ]}
        >
          <LineChart data={curve} margin={{ left: 4, right: 24, top: 8, bottom: 4 }}>
            <Grid />
            <X dataKey="value" tickFormatter={formatX} type="number" domain={["dataMin", "dataMax"]}
               label={{ value: label, position: "insideBottom", offset: -4, fill: C.ink2, fontSize: 12 }} />
            <Y tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
            <GlassTooltip
              rows={(row) => [
                { label, value: formatX(row.value) },
                { label: "PD", value: pct(row.pd), color: C.blue },
              ]}
            />
            <ReferenceLine y={policy.approve_below} stroke={C.muted} strokeDasharray="4 4"
              label={{ value: `Approve below ${pct(policy.approve_below)}`, fill: C.ink2, fontSize: 11, position: "insideBottomRight" }} />
            <ReferenceLine y={policy.decline_at} stroke={C.muted} strokeDasharray="4 4"
              label={{ value: `Decline at ${pct(policy.decline_at)}`, fill: C.ink2, fontSize: 11, position: "insideTopRight" }} />
            <ReferenceLine x={Number(profile[field] ?? 0)} stroke={C.ink} strokeOpacity={0.5}
              label={{ value: "current", fill: C.ink2, fontSize: 11, position: "top" }} />
            <Line type="monotone" dataKey="pd" stroke={C.blue} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ChartCard>
      )}
      {curve && (
        <Disclosure label="View as table">
          <Table
            maxHeight={260}
            columns={[
              { key: "value", label, align: "right", render: (v) => dec(v, 3) },
              { key: "pd", label: "Probability of default", align: "right", render: (v) => pct(v, 2) },
            ]}
            rows={curve.map((row, i) => ({ ...row, id: i }))}
          />
        </Disclosure>
      )}
    </div>
  );
}
