import { useState } from "react";
import { Bar, BarChart, Line, LineChart, ReferenceLine } from "recharts";
import { api, useApi } from "../api.js";
import { DECISION, dec, num, pct } from "../lib.js";
import { C, Card, ChartCard, Disclosure, ErrorBox, GlassTooltip, Grid, Loading, Stat, Table, Tabs, X, Y } from "../ui.jsx";

const TABS = ["Discrimination", "Calibration", "Decision policy", "Explainability", "Fairness"];

/** Linear interpolation of a curve given as ascending `xs` / `ys` at `x`. */
function interpolate(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i += 1) {
    if (x <= xs[i]) {
      const span = xs[i] - xs[i - 1];
      return span > 0 ? ys[i - 1] + ((ys[i] - ys[i - 1]) * (x - xs[i - 1])) / span : ys[i];
    }
  }
  return ys[ys.length - 1];
}

export default function Performance() {
  const model = useApi(api.model, []);
  const report = useApi(api.performance, []);
  const [tab, setTab] = useState(TABS[0]);

  if (model.error || report.error) {
    const retry = () => {
      if (model.error) model.reload();
      if (report.error) report.reload();
    };
    return <ErrorBox error={model.error ?? report.error} onRetry={retry} />;
  }
  if (!model.data || !report.data) return <Loading what="the evaluation report" />;

  const info = model.data;
  const { metrics, training } = info;
  const test = metrics.test;
  const baseline = metrics.baseline_test;
  const aucGain = test.roc_auc - baseline.roc_auc;

  return (
    <div className="page">
      <header className="stack" style={{ gap: 6 }}>
        <h1>Model performance</h1>
        <p className="muted small">
          Model {info.version} · {info.algorithm} · trained on {num(training.n_train)} applications, evaluated on{" "}
          {num(training.n_test)} held-out applications the model never saw.
        </p>
      </header>

      <div className="grid grid-auto">
        <Stat label="ROC AUC" value={dec(test.roc_auc)} delta={`${aucGain >= 0 ? "+" : ""}${aucGain.toFixed(3)} vs baseline`}
              deltaUp={aucGain >= 0} title="Baseline: ranking by the average external score alone." />
        <Stat label="Gini" value={dec(test.gini)} />
        <Stat label="KS statistic" value={dec(test.ks)} title="Max separation between good and bad score distributions." />
        <Stat label="PR AUC" value={dec(test.pr_auc)} sub={`random scores ${dec(test.base_rate)}`} />
        <Stat label="Brier score" value={dec(test.brier, 4)} title="Mean squared error of the predicted PD (lower is better)." />
        <Stat label="Mean PD" value={pct(test.mean_pd, 2)} sub={`actual ${pct(test.base_rate, 2)}`} />
      </div>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {tab === "Discrimination" && <Discrimination report={report.data} metrics={metrics} training={training} />}
      {tab === "Calibration" && <Calibration report={report.data} />}
      {tab === "Decision policy" && <Policy report={report.data} policy={info.policy} />}
      {tab === "Explainability" && <Explainability report={report.data} info={info} />}
      {tab === "Fairness" && <Fairness report={report.data} />}
    </div>
  );
}

function Discrimination({ report, metrics, training }) {
  // The two curves are sampled at different FPRs; read the baseline at the model's.
  const { model, baseline } = report.roc;
  const roc = model.fpr.map((fpr, i) => ({
    fpr,
    model: model.tpr[i],
    baseline: interpolate(baseline.fpr, baseline.tpr, fpr),
  }));
  const hist = report.score_histogram;
  const distribution = hist.bin_start.map((start, i) => ({
    score: start + hist.bin_width / 2,
    repaid: hist.repaid[i],
    defaulted: hist.defaulted[i],
  }));

  return (
    <div className="stack">
      <div className="split">
        <ChartCard
          title="ROC curve"
          height={360}
          legend={[
            { label: `CredScore model (AUC ${dec(metrics.test.roc_auc)})`, color: C.blue },
            { label: `External scores only (AUC ${dec(metrics.baseline_test.roc_auc)})`, color: C.orange },
          ]}
        >
          <LineChart data={roc} margin={{ left: 4, right: 20, top: 8, bottom: 16 }}>
            <Grid />
            <X dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)}
               label={{ value: "False positive rate", position: "insideBottom", offset: -8, fill: C.ink2, fontSize: 12 }} />
            <Y domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} />
            <GlassTooltip
              rows={(row) => [
                { label: "False positive rate", value: pct(row.fpr) },
                { label: "Model TPR", value: pct(row.model), color: C.blue },
                { label: "Baseline TPR", value: pct(row.baseline), color: C.orange },
              ]}
            />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.muted} strokeDasharray="4 4" />
            <Line dataKey="baseline" stroke={C.orange} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="model" stroke={C.blue} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ChartCard>

        <ChartCard
          title="Score distribution by actual outcome"
          height={360}
          legend={[
            { label: "Repaid", color: C.blue },
            { label: "Defaulted", color: C.orange },
          ]}
        >
          <LineChart data={distribution} margin={{ left: 4, right: 20, top: 8, bottom: 16 }}>
            <Grid />
            <X dataKey="score" type="number" domain={[300, 850]}
               label={{ value: "Credit score", position: "insideBottom", offset: -8, fill: C.ink2, fontSize: 12 }} />
            <Y tickFormatter={(v) => pct(v, 0)} />
            <GlassTooltip
              rows={(row) => [
                { label: "Score", value: num(row.score) },
                { label: "Repaid", value: pct(row.repaid), color: C.blue },
                { label: "Defaulted", value: pct(row.defaulted), color: C.orange },
              ]}
            />
            <Line type="stepAfter" dataKey="repaid" stroke={C.blue} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="stepAfter" dataKey="defaulted" stroke={C.orange} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ChartCard>
      </div>
      <p className="small muted">
        Train AUC {dec(metrics.train.roc_auc)}, validation AUC {dec(metrics.valid.roc_auc)}, test AUC{" "}
        {dec(metrics.test.roc_auc)}. Early stopping on the validation set picked {num(training.best_iteration + 1)} trees.
      </p>
    </div>
  );
}

function Calibration({ report }) {
  const max = Math.max(...report.calibration.map((r) => Math.max(r.mean_pd, r.default_rate))) * 1.05;
  return (
    <div className="stack">
      <div className="split">
        <ChartCard
          title="Predicted vs observed default rate"
          height={360}
          legend={[{ label: "Test deciles", color: C.blue }, { label: "Perfect calibration", color: C.muted, dash: true }]}
        >
          <LineChart data={report.calibration} margin={{ left: 4, right: 20, top: 8, bottom: 16 }}>
            <Grid />
            <X dataKey="mean_pd" type="number" domain={[0, max]} tickFormatter={(v) => pct(v, 0)}
               label={{ value: "Mean predicted PD", position: "insideBottom", offset: -8, fill: C.ink2, fontSize: 12 }} />
            <Y domain={[0, max]} tickFormatter={(v) => pct(v, 0)} />
            <GlassTooltip
              rows={(row) => [
                { label: "Decile", value: row.bin },
                { label: "Predicted", value: pct(row.mean_pd, 2) },
                { label: "Observed", value: pct(row.default_rate, 2), color: C.blue },
                { label: "Applicants", value: num(row.n) },
              ]}
            />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke={C.muted} strokeDasharray="4 4" />
            <Line dataKey="default_rate" stroke={C.blue} strokeWidth={2} isAnimationActive={false}
                  dot={{ r: 4, fill: C.blue, stroke: "#0b0b0d", strokeWidth: 2 }} />
          </LineChart>
        </ChartCard>

        <ChartCard title="Observed default rate by risk band" height={360}>
          <BarChart data={report.bands} margin={{ left: 4, right: 20, top: 8, bottom: 4 }}>
            <Grid />
            <X dataKey="band" />
            <Y tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
            <GlassTooltip
              rows={(row) => [
                { label: row.label, value: `${row.min_score}–${row.max_score}` },
                { label: "Default rate", value: pct(row.default_rate, 2), color: C.blue },
                { label: "Applicants", value: `${num(row.n)} (${pct(row.share, 0)})` },
              ]}
            />
            <Bar dataKey="default_rate" fill={C.blue} barSize={48} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ChartCard>
      </div>
      <p className="small muted">
        No resampling was used in training, so predicted probabilities match observed default rates. The credit score
        and the decision cut-offs rely on that.
      </p>
      <Disclosure label="View calibration as table">
        <Table
          columns={[
            { key: "bin", label: "Decile" },
            { key: "mean_pd", label: "Mean predicted PD", align: "right", render: (v) => pct(v, 2) },
            { key: "default_rate", label: "Observed default rate", align: "right", render: (v) => pct(v, 2) },
            { key: "n", label: "Applicants", align: "right", render: (v) => num(v) },
          ]}
          rows={report.calibration.map((r) => ({ ...r, id: r.bin }))}
        />
      </Disclosure>
    </div>
  );
}

function Policy({ report, policy }) {
  const gains = [{ cum_population: 0, cum_defaulters_captured: 0 }, ...report.gains];
  return (
    <div className="stack">
      <Card hover={false}>
        <p className="small">
          Applicants with PD below <b className="ink">{pct(policy.approve_below)}</b> are approved, those at or above{" "}
          <b className="ink">{pct(policy.decline_at)}</b> are declined, and everyone in between goes to manual review.
          The cut-offs were set on the validation set to approve about 70% and decline the riskiest 10%.
        </p>
      </Card>

      <div className="grid grid-3">
        {report.policy.outcomes.map((row) => (
          <Card key={row.decision} className="stack" style={{ gap: 8 }}>
            <div className="row" style={{ gap: 10 }}>
              <span className="badge-icon" style={{ background: DECISION[row.decision].color, width: 26, height: 26, fontSize: "0.8rem" }}>
                {DECISION[row.decision].icon}
              </span>
              <b className="ink">{DECISION[row.decision].label}</b>
            </div>
            <div className="stat-value mono-num">{pct(row.share, 1)}</div>
            <p className="tiny muted">
              {num(row.n)} applicants · default rate {pct(row.default_rate, 1)} · holds {pct(row.share_of_defaulters, 1)} of
              all defaulters
            </p>
          </Card>
        ))}
      </div>

      <div className="split">
        <ChartCard
          title="Cumulative gains"
          height={340}
          legend={[{ label: "Model", color: C.blue }, { label: "Random", color: C.muted, dash: true }]}
          note="Riskiest applicants first: what share of all defaulters is captured."
        >
          <LineChart data={gains} margin={{ left: 4, right: 20, top: 8, bottom: 16 }}>
            <Grid />
            <X dataKey="cum_population" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)}
               label={{ value: "Share of applicants", position: "insideBottom", offset: -8, fill: C.ink2, fontSize: 12 }} />
            <Y domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} />
            <GlassTooltip
              rows={(row) => [
                { label: "Riskiest", value: pct(row.cum_population, 0) },
                { label: "Defaulters caught", value: pct(row.cum_defaulters_captured), color: C.blue },
              ]}
            />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.muted} strokeDasharray="4 4" />
            <Line dataKey="cum_defaulters_captured" stroke={C.blue} strokeWidth={2} isAnimationActive={false}
                  dot={{ r: 3, fill: C.blue, stroke: "#0b0b0d", strokeWidth: 1.5 }} />
          </LineChart>
        </ChartCard>

        <Card hover={false} className="stack pad-0">
          <div className="card" style={{ paddingBottom: 0 }}>
            <h3>Risk deciles (1 = riskiest)</h3>
          </div>
          <Table
            maxHeight={300}
            columns={[
              { key: "decile", label: "Decile" },
              { key: "default_rate", label: "Default rate", align: "right", render: (v) => pct(v, 1) },
              { key: "lift", label: "Lift", align: "right", render: (v) => (v == null ? "—" : `${v.toFixed(2)}×`) },
              { key: "cum_defaulters_captured", label: "Cum. defaulters", align: "right", render: (v) => pct(v, 1) },
            ]}
            rows={report.gains.map((r) => ({ ...r, id: r.decile }))}
          />
        </Card>
      </div>
    </div>
  );
}

function Explainability({ report, info }) {
  const top = [...report.importance].slice(0, 20).reverse();
  return (
    <div className="split">
      <ChartCard title="Top 20 features by mean |SHAP|" height={640} note="Measured on a sample of the test set.">
        <BarChart data={top} layout="vertical" margin={{ left: 4, right: 20, top: 8, bottom: 4 }}>
          <Grid />
          <X type="number" domain={[0, "auto"]} />
          <Y type="category" dataKey="label" width={210} tick={{ fill: C.ink2, fontSize: 11 }} />
          <GlassTooltip
            rows={(row) => [
              { label: row.feature, value: dec(row.mean_abs_shap) },
              { label: "Gain share", value: pct(row.gain_share, 1) },
            ]}
          />
          <Bar dataKey="mean_abs_shap" fill={C.blue} barSize={14} radius={4} isAnimationActive={false} />
        </BarChart>
      </ChartCard>

      <div className="stack">
        <Card hover={false} className="stack">
          <h3>How explanations work</h3>
          <p className="small">
            Every score is explained with exact TreeSHAP values computed by XGBoost itself. Each feature's contribution
            moves the applicant's log-odds of default up or down from the portfolio baseline. This chart shows which
            features move scores the most on average.
          </p>
        </Card>
        <Card hover={false} className="stack">
          <h3>Deliberately excluded inputs</h3>
          <ul className="reason-list">
            {info.excluded_features?.map((item) => (
              <li key={item.feature}>
                <code className="ink">{item.feature}</code> — <span className="small">{item.reason}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}

function Fairness({ report }) {
  const groups = [
    ["gender", "By gender"],
    ["age", "By age band"],
  ];
  return (
    <div className="stack">
      <Card hover={false}>
        <p className="small">
          Gender is <b className="ink">not</b> a model input; it is kept only to monitor outcomes. Differences in
          approval rates should follow differences in observed default rates, and ranking quality (AUC) should be
          similar across groups.
        </p>
      </Card>
      <div className="split">
        {groups.map(([key, title]) =>
          report.fairness[key]?.length ? (
            <div className="stack" key={key}>
              <ChartCard
                title={title}
                height={320}
                legend={[
                  { label: "Approval rate", color: C.blue },
                  { label: "Observed default rate", color: C.orange },
                ]}
              >
                <BarChart data={report.fairness[key]} margin={{ left: 4, right: 16, top: 8, bottom: 4 }}>
                  <Grid />
                  <X dataKey="group" />
                  <Y tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
                  <GlassTooltip
                    rows={(row) => [
                      { label: "Applicants", value: num(row.n) },
                      { label: "Approval rate", value: pct(row.approval_rate), color: C.blue },
                      { label: "Default rate", value: pct(row.default_rate), color: C.orange },
                      { label: "ROC AUC", value: dec(row.roc_auc) },
                    ]}
                  />
                  <Bar dataKey="approval_rate" fill={C.blue} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="default_rate" fill={C.orange} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ChartCard>
              <Disclosure label={`${title} — table`}>
                <Table
                  columns={[
                    { key: "group", label: "Group" },
                    { key: "n", label: "Applicants", align: "right", render: (v) => num(v) },
                    { key: "default_rate", label: "Default rate", align: "right", render: (v) => pct(v, 1) },
                    { key: "approval_rate", label: "Approval rate", align: "right", render: (v) => pct(v, 1) },
                    { key: "roc_auc", label: "ROC AUC", align: "right", render: (v) => dec(v) },
                  ]}
                  rows={report.fairness[key].map((r) => ({ ...r, id: r.group }))}
                />
              </Disclosure>
            </div>
          ) : (
            <Card hover={false} key={key}>
              <p className="muted small">{title}: not enough test applicants per group to report.</p>
            </Card>
          ),
        )}
      </div>
    </div>
  );
}
