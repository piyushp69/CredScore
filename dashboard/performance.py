"""Model performance: discrimination, calibration, policy economics, explainability, fairness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import (
    BLUE, DECISION, ORANGE, dec, diagonal, figure, num, page_header, pct, require_service, show, targets_note,
)

PCT_COL = st.column_config.NumberColumn(format="percent")


def render() -> None:
    service = require_service()
    info = service.model_info()
    report = service.performance_report()
    if report is None:
        st.error("No evaluation report found; retrain the model.")
        return

    metrics, training = info["metrics"], info["training"]
    test, baseline = metrics["test"], metrics["baseline_test"]
    auc_gain = test["roc_auc"] - baseline["roc_auc"]

    page_header("Model performance",
                f"Model {info['version']} · {info['algorithm']} · trained on {num(training.get('n_train'))} "
                f"applications, evaluated on {num(training.get('n_test'))} held-out applications the model never saw.")

    cols = st.columns(6)
    # Delta chips stay short (six columns cut long ones off); the detail goes in the tooltip.
    cols[0].metric("ROC AUC", dec(test["roc_auc"]), f"{auc_gain:+.3f}",
                   help=f"Change vs the baseline that ranks by the average external score alone "
                        f"(AUC {dec(baseline['roc_auc'])}).")
    cols[1].metric("Gini", dec(test["gini"]))
    cols[2].metric("KS statistic", dec(test["ks"]), help="Max separation between good and bad score distributions.")
    cols[3].metric("PR AUC", dec(test["pr_auc"]), f"random {dec(test['base_rate'])}",
                   delta_color="off", delta_arrow="off",
                   help="A random ranking scores a PR AUC equal to the default rate, shown below the value.")
    cols[4].metric("Brier score", dec(test["brier"], 4), help="Mean squared error of the predicted PD (lower is better).")
    cols[5].metric("Mean PD", pct(test["mean_pd"], 2), f"actual {pct(test['base_rate'], 2)}",
                   delta_color="off", delta_arrow="off")

    tabs = st.tabs(["Discrimination", "Calibration", "Decision policy", "Explainability", "Fairness"])
    with tabs[0]:
        _discrimination(report, metrics, training)
    with tabs[1]:
        _calibration(report)
    with tabs[2]:
        _policy(report, info)
    with tabs[3]:
        _explainability(report, info)
    with tabs[4]:
        _fairness(report)


def _discrimination(report: dict, metrics: dict, training: dict) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        model, baseline = report["roc"]["model"], report["roc"]["baseline"]
        fig = figure(height=380, title="ROC curve")
        diagonal(fig)
        fig.add_trace(go.Scatter(x=baseline["fpr"], y=baseline["tpr"], mode="lines", line=dict(color=ORANGE, width=2),
                                 name=f"External scores only (AUC {dec(metrics['baseline_test']['roc_auc'])})"))
        fig.add_trace(go.Scatter(x=model["fpr"], y=model["tpr"], mode="lines", line=dict(color=BLUE, width=2),
                                 name=f"CredScore model (AUC {dec(metrics['test']['roc_auc'])})"))
        fig.update_xaxes(title="False positive rate", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="True positive rate", tickformat=".0%", range=[0, 1])
        show(fig)
    with right:
        hist = report["score_histogram"]
        centres = [start + hist["bin_width"] / 2 for start in hist["bin_start"]]
        fig = figure(height=380, title="Score distribution by actual outcome")
        for key, name, color in (("repaid", "Repaid", BLUE), ("defaulted", "Defaulted", ORANGE)):
            fig.add_trace(go.Scatter(x=centres, y=hist[key], mode="lines", name=name,
                                     line=dict(color=color, width=2, shape="hvh"),
                                     hovertemplate=f"Score %{{x:.0f}}<br>{name} %{{y:.1%}}<extra></extra>"))
        fig.update_xaxes(title="Credit score", range=[300, 850])
        fig.update_yaxes(tickformat=".0%")
        show(fig)
    best = training.get("best_iteration")
    st.caption(f"Train AUC {dec(metrics['train']['roc_auc'])}, validation AUC {dec(metrics['valid']['roc_auc'])}, "
               f"test AUC {dec(metrics['test']['roc_auc'])}."
               + (f" Early stopping on the validation set picked {num(best + 1)} trees." if best is not None else ""))


def _calibration(report: dict) -> None:
    calibration = pd.DataFrame(report["calibration"])
    top = float(max(calibration["mean_pd"].max(), calibration["default_rate"].max()) * 1.05)
    left, right = st.columns(2, gap="large")
    with left:
        fig = figure(height=380, title="Predicted vs observed default rate")
        diagonal(fig, top, name="Perfect calibration")
        fig.add_trace(go.Scatter(
            x=calibration["mean_pd"], y=calibration["default_rate"], mode="lines+markers", name="Test deciles",
            line=dict(color=BLUE, width=2), marker=dict(size=8),
            customdata=np.stack([calibration["bin"].astype(str), calibration["n"]], axis=1),
            hovertemplate="Decile %{customdata[0]}<br>Predicted %{x:.2%}<br>Observed %{y:.2%}"
                          "<br>%{customdata[1]:,} applicants<extra></extra>",
        ))
        fig.update_xaxes(title="Mean predicted PD", tickformat=".0%", range=[0, top])
        fig.update_yaxes(title="Observed default rate", tickformat=".0%", range=[0, top])
        show(fig)
    with right:
        bands = report["bands"]
        fig = figure(height=380, title="Observed default rate by risk band", showlegend=False)
        fig.add_trace(go.Bar(
            x=[b["band"] for b in bands], y=[b["default_rate"] for b in bands], marker_color=BLUE,
            customdata=[[b["label"], b["min_score"], b["max_score"], b["n"], b["share"]] for b in bands],
            hovertemplate="%{customdata[0]} (%{customdata[1]}–%{customdata[2]})<br>Default rate %{y:.2%}"
                          "<br>%{customdata[3]:,} applicants (%{customdata[4]:.0%})<extra></extra>",
        ))
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        show(fig)
    st.caption("No resampling was used in training, so predicted probabilities match observed default rates. "
               "The credit score and the decision cut-offs rely on that.")
    with st.expander("View calibration as table"):
        st.dataframe(
            calibration.rename(columns={"bin": "Decile", "mean_pd": "Mean predicted PD",
                                        "default_rate": "Observed default rate", "n": "Applicants"}),
            hide_index=True, column_config={"Mean predicted PD": PCT_COL, "Observed default rate": PCT_COL},
        )


def _policy(report: dict, info: dict) -> None:
    # The outcomes were measured at the trained cut-offs; live scoring may override them via the environment.
    cutoffs, live = report["policy"]["cutoffs"], info["policy"]
    st.info(f"Applicants with PD below **{pct(cutoffs['approve_below'])}** are approved, those at or above "
            f"**{pct(cutoffs['decline_at'])}** are declined, and everyone in between goes to manual review. "
            + targets_note(info))
    if live != cutoffs:
        st.warning(f"Live scoring uses overridden cut-offs (approve below {pct(live['approve_below'])}, decline at "
                   f"{pct(live['decline_at'])} or above). The outcomes below are for the trained cut-offs.")

    cols = st.columns(len(report["policy"]["outcomes"]), border=True)
    for col, row in zip(cols, report["policy"]["outcomes"]):
        decision = DECISION[row["decision"]]
        col.metric(f"{decision['icon']} {decision['label']}", pct(row["share"], 1))
        col.caption(f"{num(row['n'])} applicants · default rate {pct(row['default_rate'], 1)} · "
                    f"holds {pct(row['share_of_defaulters'], 1)} of all defaulters")

    gains = pd.DataFrame(report["gains"])
    left, right = st.columns(2, gap="large")
    with left:
        fig = figure(height=360, title="Cumulative gains")
        diagonal(fig)
        fig.add_trace(go.Scatter(
            x=[0, *gains["cum_population"]], y=[0, *gains["cum_defaulters_captured"]], mode="lines+markers",
            name="Model", line=dict(color=BLUE, width=2),
            hovertemplate="Riskiest %{x:.0%}<br>Defaulters caught %{y:.1%}<extra></extra>",
        ))
        fig.update_xaxes(title="Share of applicants", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        show(fig)
        st.caption("Riskiest applicants first: what share of all defaulters is captured.")
    with right:
        st.markdown("**Risk deciles (1 = riskiest)**")
        st.dataframe(
            gains[["decile", "default_rate", "lift", "cum_defaulters_captured"]].rename(columns={
                "decile": "Decile", "default_rate": "Default rate", "lift": "Lift",
                "cum_defaulters_captured": "Cum. defaulters"}),
            hide_index=True, height=360,
            column_config={"Default rate": PCT_COL, "Cum. defaulters": PCT_COL,
                           "Lift": st.column_config.NumberColumn(format="%.2f×")},
        )


def _explainability(report: dict, info: dict) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        top = report["importance"][:20][::-1]
        fig = figure(height=660, title="Top 20 features by mean |SHAP|", showlegend=False)
        fig.add_trace(go.Bar(
            x=[r["mean_abs_shap"] for r in top], y=[r["label"] for r in top], orientation="h", marker_color=BLUE,
            customdata=[[r["feature"], r.get("gain_share") or 0] for r in top],
            hovertemplate="%{customdata[0]}<br>mean |SHAP| %{x:.3f}<br>Gain share %{customdata[1]:.1%}<extra></extra>",
        ))
        show(fig)
        st.caption("Measured on a sample of the test set.")
    with right:
        with st.container(border=True):
            st.markdown("**How explanations work**")
            st.write("Every score is explained with exact TreeSHAP values computed by XGBoost itself. Each feature's "
                     "contribution moves the applicant's log-odds of default up or down from the portfolio baseline. "
                     "This chart shows which features move scores the most on average.")
        with st.container(border=True):
            st.markdown("**Deliberately excluded inputs**")
            excluded = info.get("excluded_features") or []
            st.markdown("\n".join(f"- `{item['feature']}` — {item['reason']}" for item in excluded) or "None.")


def _fairness(report: dict) -> None:
    st.info("Gender is **not** a model input; it is kept only to monitor outcomes. Differences in approval rates "
            "should follow differences in observed default rates, and ranking quality (AUC) should be similar "
            "across groups.")
    columns = st.columns(2, gap="large")
    for col, (key, title) in zip(columns, (("gender", "By gender"), ("age", "By age band"))):
        rows = report.get("fairness", {}).get(key) or []
        with col:
            if not rows:
                st.caption(f"{title}: not enough test applicants per group to report.")
                continue
            frame = pd.DataFrame(rows)
            fig = figure(height=340, title=title, barmode="group")
            for metric, name, color in (("approval_rate", "Approval rate", BLUE),
                                        ("default_rate", "Observed default rate", ORANGE)):
                fig.add_trace(go.Bar(x=frame["group"], y=frame[metric], name=name, marker_color=color,
                                     hovertemplate=f"%{{x}}<br>{name} %{{y:.1%}}<extra></extra>"))
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            show(fig)
            with st.expander(f"{title} — table"):
                st.dataframe(
                    frame[["group", "n", "default_rate", "approval_rate", "roc_auc"]].rename(columns={
                        "group": "Group", "n": "Applicants", "default_rate": "Default rate",
                        "approval_rate": "Approval rate", "roc_auc": "ROC AUC"}),
                    hide_index=True,
                    column_config={"Default rate": PCT_COL, "Approval rate": PCT_COL,
                                   "ROC AUC": st.column_config.NumberColumn(format="%.3f")},
                )
