"""Underwriting: score one applicant, explain the decision, and explore what-if changes."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from credscore.profile import ApplicantProfile

from .common import (
    BAND_COLOR, BLUE, DECISION, MUTED, RISK_DOWN, RISK_UP, RISKY, STRONG, figure, num, page_header, pct,
    require_service, show, targets_note,
)

# (key, label, kind, options). Kinds: num | int | optional | optional_int | select | optional_select | toggle
SECTIONS: dict[str, list[tuple]] = {
    "Applicant": [
        ("age_years", "Age (years)", "num", dict(min_value=18.0, max_value=100.0)),
        ("family_status", "Family status", "select", {}),
        ("education", "Education", "select", {}),
        ("children", "Children", "int", dict(min_value=0, max_value=20)),
        ("family_members", "Household size", "int", dict(min_value=1, max_value=25)),
        ("housing_type", "Housing", "select", {}),
        ("owns_realty", "Owns real estate", "toggle", {}),
        ("owns_car", "Owns a car", "toggle", {}),
        ("car_age_years", "Car age (years)", "optional", dict(min_value=0.0, max_value=100.0)),
        ("region_rating", "Region rating (1 = best)", "int", dict(min_value=1, max_value=3)),
    ],
    "Employment & income": [
        ("income_type", "Income type", "select", {}),
        ("occupation", "Occupation", "optional_select", {}),
        ("organization_type", "Employer type", "optional_select", {}),
        ("annual_income", "Annual income", "num", dict(min_value=1.0, step=5000.0)),
        ("years_employed", "Years in current job", "optional",
         dict(min_value=0.0, max_value=60.0, step=0.5, help="Leave empty for pensioners or unemployed applicants.")),
    ],
    "Loan request": [
        ("contract_type", "Contract type", "select", {}),
        ("credit_amount", "Credit amount", "num", dict(min_value=1.0, step=10000.0)),
        ("annuity_amount", "Annuity (periodic payment)", "num", dict(min_value=1.0, step=500.0)),
        ("goods_price", "Goods price", "optional",
         dict(min_value=1.0, step=10000.0, help="Price of the goods financed. Leave empty to use the credit amount.")),
    ],
    "External scores": [
        ("ext_source_1", "External score 1", "optional",
         dict(min_value=0.0, max_value=1.0, step=0.01,
              help="Normalized external bureau scores (0–1, higher is safer). Leave empty if unavailable.")),
        ("ext_source_2", "External score 2", "optional", dict(min_value=0.0, max_value=1.0, step=0.01)),
        ("ext_source_3", "External score 3", "optional", dict(min_value=0.0, max_value=1.0, step=0.01)),
    ],
    "Credit bureau": [
        ("bureau_loans", "Loans on record (0 = none)", "int", dict(min_value=0, max_value=500)),
        ("bureau_active_loans", "Active loans", "int", dict(min_value=0, max_value=500)),
        ("bureau_total_credit", "Total credit", "num", dict(min_value=0.0, step=10000.0)),
        ("bureau_total_debt", "Outstanding debt", "num", dict(min_value=0.0, step=10000.0)),
        ("bureau_overdue_amount", "Largest overdue amount", "num", dict(min_value=0.0, step=1000.0)),
        ("bureau_days_overdue", "Days currently overdue", "int", dict(min_value=0, max_value=3000)),
        ("bureau_years_since_last_loan", "Years since latest loan", "optional",
         dict(min_value=0.0, max_value=50.0, step=0.5)),
        ("bureau_enquiries_last_year", "Bureau enquiries (last year)", "optional_int", dict(min_value=0, max_value=100)),
    ],
    "Home Credit history": [
        ("prev_applications", "Previous applications (0 = new client)", "int", dict(min_value=0, max_value=200)),
        ("prev_approved", "Approved", "int", dict(min_value=0, max_value=200)),
        ("prev_refused", "Refused", "int", dict(min_value=0, max_value=200)),
        ("installments_paid", "Installments on record", "int", dict(min_value=0, max_value=2000)),
        ("late_payment_share", "Share paid late", "num", dict(min_value=0.0, max_value=1.0, step=0.01)),
        ("avg_days_past_due", "Average days past due", "num", dict(min_value=0.0, max_value=3000.0)),
        ("max_days_past_due", "Worst days past due", "num", dict(min_value=0.0, max_value=3000.0)),
        ("payment_ratio", "Share of each installment paid", "num", dict(min_value=0.0, max_value=5.0, step=0.05)),
    ],
}
FIELDS = {key: (label, kind, opts) for fields in SECTIONS.values() for key, label, kind, opts in fields}

WHAT_IF = {
    "ext_source_2": ("External score 2", lambda v: (0.02, 0.85)),
    "ext_source_3": ("External score 3", lambda v: (0.02, 0.9)),
    "annual_income": ("Annual income", lambda v: (max(20000, v * 0.3), v * 3)),
    "credit_amount": ("Credit amount", lambda v: (max(20000, v * 0.2), v * 2.5)),
    "annuity_amount": ("Annuity", lambda v: (max(1000, v * 0.4), v * 2)),
    "age_years": ("Age (years)", lambda v: (20, 70)),
    "years_employed": ("Years in current job", lambda v: (0, 30)),
    "late_payment_share": ("Share of installments paid late", lambda v: (0, 0.6)),
}


def _widget_value(kind: str, value):
    """Coerce a profile value to the type its widget expects (Streamlit is strict about int vs float)."""
    if value is None:
        return None
    if kind in ("int", "optional_int"):
        return int(value)
    if kind in ("num", "optional"):
        return float(value)
    return value


def _load_profile(profile: dict) -> None:
    for key, (_, kind, _) in FIELDS.items():
        st.session_state[f"uw_{key}"] = _widget_value(kind, profile.get(key))
    # Streamlit discards widget state while another page is shown; this copy outlives it.
    st.session_state["uw_inputs"] = dict(profile)


def _field(key: str, options: dict[str, list[str]]):
    label, kind, opts = FIELDS[key]
    state_key = f"uw_{key}"
    if kind == "toggle":
        return st.toggle(label, key=state_key)
    if kind in ("select", "optional_select"):
        choices = list(options.get(key, []))
        if kind == "optional_select":
            choices = [None, *choices]
        if st.session_state.get(state_key) not in choices:
            st.session_state[state_key] = choices[0] if choices else None
        return st.selectbox(label, choices, key=state_key,
                            format_func=lambda v: "— not specified —" if v is None else v)
    step = 1 if kind in ("int", "optional_int") else opts.get("step")
    fmt = "%d" if kind in ("int", "optional_int") else None
    kwargs = {k: v for k, v in opts.items() if k != "step"}
    if kind.startswith("optional"):
        # value=None makes the input clearable; without it an emptied field snaps back to min_value.
        kwargs |= {"value": None, "placeholder": "not provided"}
    return st.number_input(label, key=state_key, step=step, format=fmt, **kwargs)


def render() -> None:
    service = require_service()
    schema, info = service.schema(), service.model_info()
    defaults = schema["defaults"]

    page_header("Applicant underwriting",
                "Score a loan applicant, see why the model decided what it did, and test what would change it.")

    if "uw_age_years" not in st.session_state:
        # First visit, or the form's widget state was discarded while another page was open.
        _load_profile(st.session_state.get("uw_inputs", defaults))

    presets = {"Typical applicant": defaults, "Strong applicant": STRONG, "Risky applicant": RISKY}
    cols = st.columns([1.2, 1, 1, 1, 2], vertical_alignment="center")
    cols[0].caption("Start from an example")
    for col, (name, preset) in zip(cols[1:], presets.items()):
        if col.button(name, width="stretch"):
            _load_profile(preset)
            st.session_state.pop("uw_scored", None)
            st.rerun()

    with st.form("applicant"):
        values = {}
        for tab, (section, fields) in zip(st.tabs(list(SECTIONS)), SECTIONS.items()):
            with tab:
                grid = st.columns(3)
                for i, (key, *_rest) in enumerate(fields):
                    with grid[i % 3]:
                        values[key] = _field(key, schema["options"])
        submitted = st.form_submit_button("⚡ Score applicant", type="primary", width="stretch")

    if submitted:
        st.session_state["uw_inputs"] = values
        _score(service, values)

    scored = st.session_state.get("uw_scored")
    if scored is None:
        st.info("Fill in the applicant details (or start from an example) and press **Score applicant**.")
        return
    _result(service, info, scored["profile"], scored["result"])


def _score(service, values: dict) -> None:
    try:
        profile = ApplicantProfile.model_validate(values)
    except ValidationError as exc:
        messages = [f"{'.'.join(map(str, e['loc'])) or 'profile'}: {e['msg']}" for e in exc.errors()]
        st.error("**Please fix the applicant details:**\n\n" + "\n".join(f"- {m}" for m in messages))
        return
    errors = service.profile_errors(profile)
    if errors:
        st.error("\n".join(f"- {m}" for m in errors))
        return
    result = service.score_profiles([profile], top_k=12)[0]
    profile_dict = profile.model_dump()
    st.session_state["uw_scored"] = {"profile": profile_dict, "result": result}
    history = st.session_state.setdefault("uw_history", [])
    history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Score": result["credit_score"],
        "Band": result["risk_band"]["code"],
        "PD": result["probability_of_default"],
        "Decision": DECISION[result["decision"]]["label"],
        "Income": profile.annual_income,
        "Credit": profile.credit_amount,
    })


def _score_scale(score: int, bands: list[dict]) -> go.Figure:
    fig = figure(height=130, showlegend=False, margin=dict(l=8, r=8, t=28, b=8))
    for band in bands:
        width = band["max_score"] - band["min_score"] + 1
        fig.add_trace(go.Bar(
            x=[width], y=[""], base=band["min_score"], orientation="h",
            marker_color=BAND_COLOR.get(band["status"], MUTED), opacity=0.35 if not (
                band["min_score"] <= score <= band["max_score"]) else 0.95,
            text=band["code"], textposition="inside", insidetextanchor="middle",
            hovertemplate=f"Band {band['code']} · {band['label']}<br>{band['min_score']}–{band['max_score']}<extra></extra>",
        ))
    fig.add_vline(x=score, line_width=3, line_color="white")
    fig.add_annotation(x=score, y=1, yref="paper", text=f"<b>{score}</b>", showarrow=False, yanchor="bottom")
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_xaxes(range=[bands[0]["min_score"] if bands else 300, bands[-1]["max_score"] if bands else 850])
    fig.update_yaxes(showgrid=False, showticklabels=False)
    return fig


def _result(service, info: dict, profile: dict, result: dict) -> None:
    band = result["risk_band"]
    decision = DECISION[result["decision"]]
    base_rate = info.get("training", {}).get("base_rate")
    bands = sorted(info["scorecard"]["bands"], key=lambda b: b["min_score"])

    st.divider()
    c1, c2, c3 = st.columns(3, border=True)
    with c1:
        st.metric("Credit score", result["credit_score"])
        color = {"good": "green", "warning": "orange", "serious": "orange"}.get(band["status"], "red")
        st.markdown(f":{color}-badge[Band {band['code']} · {band['label']}]")
    with c2:
        st.metric("Probability of default", pct(result["probability_of_default"]))
        if base_rate:
            st.caption(f"{result['probability_of_default'] / base_rate:.1f}× the portfolio average of {pct(base_rate)}")
    with c3:
        st.metric("Recommended decision", f"{decision['icon']} {decision['label']}")
        st.caption(result["decision_reason"])

    show(_score_scale(result["credit_score"], bands))

    metric_cols = st.columns(len(result["key_metrics"]) or 1)
    for col, metric in zip(metric_cols, result["key_metrics"].values()):
        col.metric(metric["label"], metric["display_value"])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("What drove this score")
        explanations = sorted(result["explanations"], key=lambda e: abs(e["contribution"]))
        fig = figure(height=max(300, len(explanations) * 32 + 60), showlegend=False)
        fig.add_trace(go.Bar(
            x=[e["contribution"] for e in explanations],
            y=[f"{e['label']} = {e['display_value']}" for e in explanations],
            orientation="h",
            marker_color=[RISK_UP if e["contribution"] > 0 else RISK_DOWN for e in explanations],
            customdata=[e["source"] for e in explanations],
            hovertemplate="%{y}<br>Impact %{x:+.3f}<br>Source: %{customdata}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        show(fig)
        st.caption("SHAP contributions in log-odds. :red[Red bars raise] the default risk, :green[green bars lower] it.")

    with right:
        with st.container(border=True):
            st.subheader("Main risk factors")
            if result["reasons"]:
                st.markdown("\n".join(f"- {reason}" for reason in result["reasons"]))
            else:
                st.caption("No factor materially raises this applicant's risk.")
            defaulted = [e["label"] for e in result["explanations"] if e["source"] == "default"]
            if defaulted:
                st.caption(f"Not provided, so a typical applicant's value was used: {', '.join(defaulted)}.")
            report = {"generated_at": datetime.now(timezone.utc).isoformat(), "profile": profile, "result": result}
            st.download_button("⤓ Download decision report", json.dumps(report, indent=2),
                               file_name=f"credscore_decision_{datetime.now():%Y%m%d_%H%M%S}.json",
                               mime="application/json")
        with st.container(border=True):
            st.subheader("Decision policy")
            policy, trained = info["policy"], service.model.metadata["policy"]
            overridden = any(policy[k] != trained[k] for k in policy)
            note = "Set with CREDSCORE_APPROVE_PD / CREDSCORE_DECLINE_PD." if overridden else targets_note(info)
            st.caption(f"Approve below {pct(policy['approve_below'])} · decline at {pct(policy['decline_at'])} or above. "
                       + note)

    _what_if(service, profile, info["policy"])

    history = st.session_state.get("uw_history", [])
    if len(history) > 1:
        st.subheader("This session")
        st.dataframe(pd.DataFrame(history), hide_index=True, column_config={
            "PD": st.column_config.NumberColumn(format="percent"),
            "Income": st.column_config.NumberColumn(format="localized"),
            "Credit": st.column_config.NumberColumn(format="localized"),
        })


def _what_if(service, profile: dict, policy: dict) -> None:
    st.subheader("What-if analysis")
    choices = [f for f in WHAT_IF if f != "late_payment_share" or profile.get("installments_paid")]
    head, pick = st.columns([3, 2], vertical_alignment="bottom")
    head.caption("Vary one input, keep everything else fixed, and watch the probability of default respond.")
    field = pick.selectbox("Input to vary", choices, format_func=lambda f: WHAT_IF[f][0], key="uw_whatif")
    label = WHAT_IF[field][0]

    current = profile.get(field)  # None when not provided: there is no current point to mark
    values = _sweep(field, current)
    response = service.score_batch([{**profile, field: v} for v in values], n_reasons=0)
    curve = pd.DataFrame({
        label: values,
        "Probability of default": [(item["result"] or {}).get("probability_of_default") for item in response["results"]],
    })

    fig = figure(height=360)
    fig.add_trace(go.Scatter(x=curve[label], y=curve["Probability of default"], mode="lines", name="Probability of default",
                             line=dict(color=BLUE, width=2.5, shape="spline"),
                             hovertemplate=f"{label} %{{x:,.3~f}}<br>PD %{{y:.2%}}<extra></extra>"))
    fig.add_hline(y=policy["approve_below"], line_dash="dash", line_color=MUTED,
                  annotation_text=f"Approve below {pct(policy['approve_below'])}", annotation_position="bottom right")
    fig.add_hline(y=policy["decline_at"], line_dash="dash", line_color=MUTED,
                  annotation_text=f"Decline at {pct(policy['decline_at'])}", annotation_position="top right")
    if current is not None:
        fig.add_vline(x=current, line_color="rgba(200,200,200,0.6)", annotation_text="current",
                      annotation_position="top")
    fig.update_layout(title=f"Probability of default vs {label.lower()}", showlegend=False)
    fig.update_xaxes(title=label)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    show(fig)
    if current is None:
        st.caption(f"{label} is not provided for this applicant, so the curve has no current point.")
    with st.expander("View as table"):
        st.dataframe(curve, hide_index=True, column_config={
            "Probability of default": st.column_config.NumberColumn(format="percent")})


def _sweep(field: str, current: float | None) -> list[float]:
    """25 evenly spaced values to try for a what-if input, spanning the applicant's own value."""
    lo, hi = WHAT_IF[field][1](float(current or 0))
    if current is not None:
        lo, hi = min(lo, current), max(hi, current)
    return [round(lo + (hi - lo) * i / 24, 3) for i in range(25)]
